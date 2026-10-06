#!/usr/bin/env python3
"""Diagnose saved P1 references; native stages load solutions, never solve anew."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
ROOT = Path(os.environ.get('P1_ORIGINAL_ROOT', Path(__file__).resolve().parent)).resolve()
sys.path.insert(1, str(ROOT))


import numpy as np

from p1_design import file_sha256, write_json
from p1_geometry import honeycomb_layout, physical_parameters
from p1_pairing import assign, load_run, normalized
from run_hf_batches import _supervise

DATA = ROOT / 'p1_data_five'
FOLLOW = DATA / 'hf_followup_20261003'
EXPRESSIONS = ('u', 'v', 'w', 'arx', 'ary', 'arz')
SHELL_REFERENCES = (
    'https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.032.html',
    'https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.131.html',
    'https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/application_programming_guide.15.75.html',
)


def clipped_wall_segments(p):
    """The builder's two lattices, clipped to the plate, shared walls counted once."""
    layout = honeycomb_layout(p)
    length = p['L']
    unique = {}
    for ox, oy in layout['origins']:
        for ix in range(layout['nx']):
            for iy in range(layout['ny']):
                vertices = np.asarray(layout['vertices'], dtype=float)
                vertices = vertices + (ox + ix*layout['dx'], oy + iy*layout['dy'])
                for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                    delta = b-a
                    lo, hi = 0., 1.
                    for axis in range(2):
                        if abs(delta[axis]) < 1e-15:
                            if a[axis] < -1e-12 or a[axis] > length+1e-12:
                                hi = -1.
                                break
                        else:
                            bounds = sorted((-a[axis]/delta[axis], (length-a[axis])/delta[axis]))
                            lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
                    if hi-lo <= 1e-12:
                        continue
                    endpoints = np.clip(np.stack((a+lo*delta, a+hi*delta)), 0, length)
                    key = tuple(sorted(tuple(np.round(row, 12)) for row in endpoints))
                    unique.setdefault(key, np.asarray(key, dtype=float))
    if not unique:
        raise ValueError('No physical core walls')
    return np.stack([unique[key] for key in sorted(unique)])


def shell_quadrature(p, face_order, wall_order):
    """Common physical shell surfaces; thickness integration handled separately."""
    nodes, weight = np.polynomial.legendre.leggauss(face_order)
    coordinate = .5*p['L']*(nodes+1)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing='xy')
    face_weight = (.25*p['L']**2*np.outer(weight, weight)).ravel()
    result = []
    for selection, z, thickness, offset in (
        ('sel_top', p['hc'], p['h_top'], p['h_top']/2),
        ('sel_bot', 0., p['h_bottom'], -p['h_bottom']/2),
    ):
        points = np.vstack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
        result.append((selection, points, face_weight, thickness, offset))
    walls = clipped_wall_segments(p)
    nodes, weight = np.polynomial.legendre.leggauss(wall_order)
    along = .5*(nodes+1)
    xy = walls[:, 0, None, :] + (walls[:, 1]-walls[:, 0])[:, None, :]*along[None, :, None]
    xy = np.repeat(xy[:, :, None, :], wall_order, axis=2)
    zz = np.broadcast_to(.5*p['hc']*(nodes+1), xy.shape[:-1])
    points = np.vstack((xy[..., 0].ravel(), xy[..., 1].ravel(), zz.ravel()))
    lengths = np.linalg.norm(walls[:, 1]-walls[:, 0], axis=1)
    surface_weights = (.25*p['hc']*lengths[:, None, None]
                       *weight[None, :, None]*weight[None, None, :]).ravel()
    result.append(('sel_core', points, surface_weights, p['tc'], 0.))
    return result


def mass_vectors(values, area_weights, thickness, offset, density=2710.):
    """Integrate rho|u + z*ar|^2 through thickness, including offset coupling.

    ar is the dimensionless displacement of the shell director, not an angle
    approximated or inferred from top-surface w. Two-point Gauss integration
    is exact for this linear director kinematics through each thickness.
    """
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] != 6 or values.shape[2] != len(area_weights):
        raise ValueError('Expected six native components by modes by surface points')
    displacement = np.moveaxis(values[:3], 0, -1)
    director = np.moveaxis(values[3:], 0, -1)
    positions = offset + .5*thickness*np.array([-1., 1.])/np.sqrt(3.)
    particle = displacement[:, :, None, :] + director[:, :, None, :]*positions[None, None, :, None]
    particle *= np.sqrt(density*area_weights*.5*thickness)[None, :, None, None]
    return particle.reshape(values.shape[1], -1)


def span_diagnostics(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError('Common quadrature and matching cluster dimensions required')
    norms = np.linalg.norm(reference, axis=1)
    current_norms = np.linalg.norm(current, axis=1)
    if np.any(norms <= 0) or np.any(current_norms <= 0):
        raise ValueError('Zero weighted modal state')
    a, b = reference/norms[:, None], current/current_norms[:, None]
    qa = np.linalg.qr(a.T)[0]
    qb = np.linalg.qr(b.T)[0]
    singular = np.linalg.svd(qa.conj().T@qb, compute_uv=False)
    return {
        'principal_angles_degrees': np.degrees(np.arccos(np.clip(singular, 0, 1))).tolist(),
        'individual_mac': np.abs(np.sum(a.conj()*b, axis=1))**2,
        'reference_span_condition': float(np.linalg.cond(a.T)),
        'current_span_condition': float(np.linalg.cond(b.T)),
        'reference_normalized_gram': np.abs(a.conj()@a.T).tolist(),
        'current_normalized_gram': np.abs(b.conj()@b.T).tolist(),
    }


def references(run):
    if run == 2209:
        return {'mesh4': DATA/'hf/run_2209.npz',
                **{f'mesh{m}': FOLLOW/f'mesh{m}/hf/run_2209.npz' for m in (3, 2)},
                'mesh1': FOLLOW/'mesh1_scratch/hf/run_2209.npz'}
    return {'mesh4': DATA/'hf/run_3408.npz',
            'mesh3_original': DATA/'mesh_convergence_check/extreme_angle_3408_mesh3/hf/run_3408.npz',
            'mesh3_compressed': FOLLOW/'mumps_blr3408_control_mesh3/hf/run_3408.npz',
            'mesh2_compressed': FOLLOW/'resume3408_mesh2/hf/run_3408.npz'}


def top_diagnosis(output):
    protocol = json.loads((output/'protocol.json').read_text())
    jobs, results = [], {}
    with (DATA/'corrections_five/pairing.csv').open(newline='', encoding='utf-8') as stream:
        pairs = list(csv.DictReader(stream))
    for run in (2209, 3408):
        accepted = protocol['reference']['runs'][str(run)]['accepted_mode']
        pair = next(p for p in pairs if int(p['run']) == run and int(p['mode']) == accepted and p['status'] == 'accepted')
        anchor = int(pair['hf_mode'])-1
        bundles = {key: load_run(path, 'hf') for key, path in references(run).items()}
        campaign = bundles['mesh4']
        gaps = abs(campaign['f']-campaign['f'][anchor])/np.maximum(campaign['f'], campaign['f'][anchor])
        gaps[anchor] = np.inf
        neighbor = int(np.argmin(gaps))
        cluster = np.array([anchor, neighbor])
        original = normalized(campaign['w'][cluster]).reshape(2, -1)
        rows = {}
        for label, bundle in bundles.items():
            if bundle['meta']['parameters'] != campaign['meta']['parameters']:
                raise ValueError('Reference geometry changed')
            mapping, mac = assign(campaign['w'], bundle['w'])
            indices = mapping[cluster]
            current = normalized(bundle['w'][indices]).reshape(2, -1)
            metrics = span_diagnostics(original, current)
            metrics['individual_mac'] = metrics['individual_mac'].tolist()
            rows[label] = {'matched_hf_modes': (indices+1).tolist(),
                           'frequency_hz': bundle['f'][indices].tolist(),
                           'frequency_gap_pct': float(100*abs(np.diff(bundle['f'][indices])[0])/max(bundle['f'][indices])),
                           'bundle_sha256': file_sha256(bundle['path']), **metrics}
            native = Path(bundle['path']).with_suffix('.mph')
            if native.is_file():
                with np.load(bundle['path'], allow_pickle=False) as archive:
                    solnums = archive['solnums'][indices].astype(int).tolist()
                jobs.append({'run': run, 'label': label, 'model': str(native),
                             'bundle': bundle['path'], 'solnums': solnums,
                             'parameters': bundle['meta']['parameters'],
                             'campaign_cluster': (cluster+1).tolist()})
        results[str(run)] = {'accepted_mode': accepted, 'campaign_hf_cluster': (cluster+1).tolist(),
                             'selection': 'Nearest campaign frequency, fixed before finer-span fitting',
                             'quantity': 'Extracted top-transverse fields; not full shell state or mass metric',
                             'references': rows}
    write_json(output/'reference_top_diagnosis.json', {'source_sha256': file_sha256(__file__), 'runs': results})
    write_json(output/'reference_native_jobs.json', {'jobs': jobs, 'protocol_sha256': file_sha256(output/'protocol.json')})
    print(json.dumps(results, indent=2), flush=True)


def interpolate(model, selection, points, solnums, tag):
    numerical = model.java.result().numerical()
    feature = numerical.create(tag, 'Interp')
    try:
        feature.set('data', 'dset1')
        feature.selection().named(selection)
        feature.set('expr', list(EXPRESSIONS))
        feature.set('solnum', ','.join(map(str, solnums)))
        feature.set('coord', points.tolist())
        feature.set('coorderr', 'on')
        feature.set('matherr', 'on')
        feature.set('ext', .1)
        values = np.asarray(feature.getData(), dtype=float)
        if feature.isComplex():
            values = values+1j*np.asarray(feature.getImagData(), dtype=float)
        if values.shape != (6, len(solnums), points.shape[1]) or not np.isfinite(values).all():
            raise ValueError(f'Incomplete native shell state: {values.shape}')
        units = [str(unit) for unit in feature.getStringArray('unit')]
        if units[:3] != ['m']*3 or any(unit not in ('', '1') for unit in units[3:]) or len(units) != 6:
            raise ValueError(f'Unexpected shell displacement/director units: {units}')
        return values, units
    finally:
        numerical.remove(tag)


def native_mass(model, solnum):
    numerical = model.java.result().numerical()
    feature = numerical.create('p1_reference_mass', 'IntSurface')
    try:
        feature.set('data', 'dset1')
        feature.set('solnum', str(solnum))
        mass = 0.
        # Use the actual ThicknessOffset expressions and named selections;
        # a physics tag is not necessarily its postprocessing variable prefix.
        physics = model.java.component('comp1').physics('sh')
        for tag, selection in (('th_top', 'sel_top'), ('th_bot', 'sel_bot'), ('th_core', 'sel_core')):
            thickness = str(physics.feature(tag).getString('d'))
            feature.selection().named(selection)
            feature.set('expr', [f'2710[kg/m^3]*({thickness})'])
            mass += float(np.asarray(feature.getReal()).reshape(-1)[0])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('Nonpositive native shell mass')
        return mass
    finally:
        numerical.remove('p1_reference_mass')


def native_worker(args):
    import mph
    import hc_HighFidelity_LHS as hf
    jobs = json.loads((args.output/'reference_native_jobs.json').read_text())['jobs']
    job = jobs[args.job]
    protocol = json.loads((args.output/'protocol.json').read_text())
    levels = protocol['reference']['quadrature_levels']
    if args.probe:
        job = {'run': 4352, 'label': 'native_api_probe',
               'model': str(FOLLOW/'mesh3/hf/run_4352.mph'),
               'bundle': str(FOLLOW/'mesh3/hf/run_4352.npz')}
        with np.load(job['bundle'], allow_pickle=False) as archive:
            job['solnums'] = archive['solnums'][:2].astype(int).tolist()
            job['parameters'] = json.loads(str(archive['metadata'].item()))['parameters']
        levels = [{'face_order': 12, 'wall_order': 1}]
    p = physical_parameters(job['parameters'])
    destination = args.output/'reference_native'/f"run{job['run']}_{job['label']}.npz"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace native export: {destination}')
    mph.option('session', 'stand-alone')
    client = None
    try:
        started = time.monotonic()
        client = mph.start(cores=args.cores)
        client.java.showProgress(os.environ['P1_NATIVE_PROGRESS'])
        print(f"PHASE loading saved model {job['model']}; no new solve", flush=True)
        model = client.load(job['model'])
        print(f'PHASE loaded in {time.monotonic()-started:.1f}s', flush=True)
        total_mass = native_mass(model, job['solnums'][0])
        print(f'PHASE native shell mass: {total_mass:.12g} kg; dependent-variable units required before export', flush=True)
        arrays, metadata = {}, {'job': job, 'native_mass_kg': total_mass,
                               'quantity': 'Full displacement/director mass-weighted common-shell quadrature; not a full-DOF matrix eigenspace certificate',
                               'kinematics': 'rho * integral_surface integral_thickness |u + z*ar|^2; relative shell offsets included',
                               'source_sha256': file_sha256(__file__), 'native_model_sha256': file_sha256(job['model']),
                               'bundle_sha256': file_sha256(job['bundle']), 'documentation': SHELL_REFERENCES,
                               'levels': []}
        for level, spec in enumerate(levels):
            vectors = []
            mass = 0.
            coordinate_hash = hashlib.sha256()
            for selection, points, weights, thickness, offset in shell_quadrature(p, **spec):
                mass += 2710*thickness*weights.sum()
                coordinate_hash.update(points.tobytes())
                coordinate_hash.update(weights.tobytes())
                for chunk, start in enumerate(range(0, points.shape[1], 16000)):
                    end = min(start+16000, points.shape[1])
                    values, units = interpolate(model, selection, points[:, start:end], job['solnums'], f'p1_state_{level}_{chunk}')
                    vectors.append(mass_vectors(values, weights[start:end], thickness, offset))
                print(f'PHASE level{level} {selection}: {points.shape[1]} surface points; units={units}', flush=True)
            relative_mass_error = abs(mass-total_mass)/total_mass
            if relative_mass_error > 1e-6:
                raise ValueError(f'Common geometry mass disagrees with native integral: {mass} vs {total_mass}')
            vector = np.concatenate(vectors, axis=1)
            gram = vector.conj()@vector.T
            arrays[f'level{level}_vectors'] = vector
            metadata['levels'].append({'quadrature': spec, 'surface_mass_kg': float(mass),
                                       'relative_mass_error': float(relative_mass_error),
                                       'coordinate_weights_sha256': coordinate_hash.hexdigest(),
                                       'state_components': EXPRESSIONS, 'units': units,
                                       'mass_norm_squared': np.real(np.diag(gram)).tolist(),
                                       'complex_gram_real': gram.real.tolist(), 'complex_gram_imag': gram.imag.tolist()})
        arrays['metadata'] = np.asarray(json.dumps(metadata, sort_keys=True))
        hf.atomic_npz(destination, {k: v for k, v in arrays.items() if k != 'metadata'}, metadata)
        write_json(destination.with_suffix('.json'), metadata)
        print(f'PHASE saved {destination}; elapsed={time.monotonic()-started:.1f}s', flush=True)
        return 0
    finally:
        hf._close_client(client)


def native_controller(args):
    protocol = json.loads((args.output/'protocol.json').read_text())
    settings = protocol['resources']
    log_root = args.output/'reference_worker_logs'
    log_root.mkdir(exist_ok=True)
    jobs = json.loads((args.output/'reference_native_jobs.json').read_text())['jobs']
    indices = [0] if args.probe else list(range(len(jobs)))
    outcomes = []
    for index in indices:
        label = 'probe' if args.probe else f"run{jobs[index]['run']}_{jobs[index]['label']}"
        version = file_sha256(__file__)[:12]
        log = log_root/f'{label}_{version}.log'
        if log.exists():
            raise FileExistsError('A native attempt exists; no unchanged automatic retry')
        with TemporaryDirectory(prefix='p1_reference_', dir=settings['scratch_root']) as scratch:
            env = os.environ.copy()
            env['TEMP'] = env['TMP'] = scratch
            options = re.sub(r'(?<!\S)-Xmx\S+', '', env.get('JAVA_TOOL_OPTIONS', '')).strip()
            env['JAVA_TOOL_OPTIONS'] = f'{options} -Xmx{settings["java_heap_gib"]}g -Djava.io.tmpdir="{Path(scratch).as_posix()}"'.strip()
            env['P1_NATIVE_PROGRESS'] = str(log.with_suffix('.progress.log'))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'native', '--output', str(args.output),
                       '--worker', '--job', str(index), '--cores', str(settings['native_cores'])]
            if args.probe:
                command.append('--probe')
            started = datetime.now(timezone.utc).isoformat()
            status = _supervise(command, ROOT, env, log, settings['native_worker_deadline_minutes']*60, settings['heartbeat_seconds'])
        outcomes.append({'label': label, 'exit_code': status, 'started_at_utc': started,
                         'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                         'owned_scratch_removed': not Path(scratch).exists()})
        stem = 'reference_probe_execution' if args.probe else 'reference_native_execution'
        write_json(args.output/f'{stem}_{version}.json',
                   {'attempts_per_job_per_source_version': 1, 'source_sha256': file_sha256(__file__), 'outcomes': outcomes}, overwrite=True)
        if status:
            return status
    return 0


def native_analysis(output):
    from itertools import combinations
    protocol = json.loads((output/'protocol.json').read_text())
    jobs = json.loads((output/'reference_native_jobs.json').read_text())['jobs']
    exports = {}
    for job in jobs:
        path = output/'reference_native'/f"run{job['run']}_{job['label']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            exports[(job['run'], job['label'])] = {key: archive[key] for key in archive.files}
    results = {}
    for run in (2209, 3408):
        labels = [job['label'] for job in jobs if job['run'] == run]
        metrics = {}
        for reference, label in combinations(labels, 2):
            first, current = exports[(run, reference)], exports[(run, label)]
            first_meta = json.loads(str(first['metadata'].item()))
            current_meta = json.loads(str(current['metadata'].item()))
            levels = []
            for level in range(len(protocol['reference']['quadrature_levels'])):
                if first_meta['levels'][level]['coordinate_weights_sha256'] != current_meta['levels'][level]['coordinate_weights_sha256']:
                    raise ValueError('Native exports do not use a common physical quadrature')
                measurement = span_diagnostics(first[f'level{level}_vectors'], current[f'level{level}_vectors'])
                measurement['individual_mac'] = measurement['individual_mac'].tolist()
                levels.append(measurement)
            angle_change = float(np.max(np.abs(np.array(levels[0]['principal_angles_degrees'])-levels[1]['principal_angles_degrees'])))
            metrics[f'{reference}_vs_{label}'] = {'levels': levels, 'quadrature_angle_change_degrees': angle_change,
                'quadrature_angle_target_passed': angle_change <= protocol['reference']['quadrature_angle_target_degrees']}
        results[str(run)] = {'quantity': 'Mass-weighted full shell displacement/director state sampled on all physical shell surfaces',
                             'not_claimed': 'Full-DOF eigenvector convergence or an exact mass-matrix certificate',
                             'comparisons': metrics}
    write_json(output/'reference_mass_diagnosis_all_pairs.json', {'source_sha256': file_sha256(__file__), 'runs': results})
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('top', 'native', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == 'top':
        top_diagnosis(args.output)
    elif args.stage == 'native':
        return native_worker(args) if args.worker else native_controller(args)
    else:
        native_analysis(args.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
