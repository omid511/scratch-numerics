"""P1 explicit honeycomb shell model and top-surface Interp exports.

COMSOL Interp API: https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_results.52.082.html
The top reference surface is z=hc; transverse shell displacement is constant
through its thickness.  Legacy bundles lacking the quality fields are
incompatible with the revised HF export contract.  The entrypoint isolates
MPh/JVM state by running each batch in a fresh Python worker process.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from p1_geometry import (PARAM_NAMES, L_FIXED, EIGEN_SHIFT_HZ,
                         physical_parameters, honeycomb_layout,
                         geometry_tolerance, preflight)

N_EIGS, GRID_RES = 16, 80
HF_CANDIDATE_MULTIPLIER = 2
MIN_TRANSVERSE_FRACTION = 0.5
MODEL_VERSION = 'p1-v2'

def _set_box_bounds(box, x_bounds, y_bounds, z_bounds):
    for axis, bounds in (('x', x_bounds), ('y', y_bounds), ('z', z_bounds)):
        box.set(axis + 'min', str(bounds[0]))
        box.set(axis + 'max', str(bounds[1]))


def build_model(client, p, mesh_size=4, n_eigs=N_EIGS,
                candidate_eigs=None, eigen_shift_hz=EIGEN_SHIFT_HZ):
    if mesh_size < 1 or n_eigs < 1:
        raise ValueError('mesh_size and n_eigs must be positive')
    candidate_eigs = max(n_eigs, candidate_eigs or HF_CANDIDATE_MULTIPLIER * n_eigs)
    model = client.create('p1_honeycomb')
    m = model.java
    for name in ('tc', 'hc', 'h_top', 'h_bottom', 'L'):
        m.param().set(name, f'{p[name]}[m]')
    m.component().create('comp1', True)
    comp = m.component('comp1')
    g = comp.geom().create('geom1', 3)
    g.create('wp1', 'WorkPlane').set('quickplane', 'xy')
    wp = g.feature('wp1').geom()
    layout = honeycomb_layout(p)
    for i, (ox, oy) in enumerate(layout['origins'], 1):
        pts = [(x + ox, y + oy) for x, y in layout['vertices']]
        pts.append(pts[0])
        polygon = wp.create(f'hex{i}', 'Polygon')
        polygon.set('type', 'open')
        polygon.set('x', ','.join(f'{x:.16g}' for x, y in pts))
        polygon.set('y', ','.join(f'{y:.16g}' for x, y in pts))
        array = wp.create(f'arr{i}', 'Array')
        array.selection('input').set(f'hex{i}')
        array.set('type', 'rectangular')
        for axis, key in enumerate(('nx', 'ny')):
            array.setIndex('fullsize', str(layout[key]), axis)
        for axis, key in enumerate(('dx', 'dy')):
            array.setIndex('displ', str(layout[key]), axis)
    union = wp.create('uni1', 'Union')
    union.selection('input').set('arr1', 'arr2')
    # The two sublattices intentionally share walls.  Merge coincident curves
    # before extrusion so a shared wall contributes one shell, not two.
    union.set('intbnd', 'off')
    rectangle = wp.create('r1', 'Rectangle')
    rectangle.set('size', ['L', 'L'])
    rectangle.set('pos', ['0', '0'])
    wp.create('int1', 'Intersection').selection('input').set('uni1', 'r1')
    g.create('ext1', 'Extrude').selection('input').set('wp1')
    g.feature('ext1').setIndex('distance', 'hc', 0)
    for tag, z in [('wp_bot', '0'), ('wp_top', 'hc')]:
        plane = g.create(tag, 'WorkPlane')
        plane.set('quickplane', 'xy')
        plane.set('quickz', z)
        face = plane.geom().create('r1', 'Rectangle')
        face.set('size', ['L', 'L'])
        face.set('pos', ['0', '0'])
    g.run()
    mat = comp.material().create('mat1', 'Common')
    for name, value in [('density', '2710'), ('youngsmodulus', '70e9'),
                        ('poissonsratio', '0.33')]:
        mat.propertyGroup('def').set(name, value)
    mat.selection().all()
    sel = comp.selection()
    tol = geometry_tolerance(p)
    xy_bounds = (-tol, p['L'] + tol)
    z_bounds = (-tol, p['hc'] + tol)
    for tag, z in [('sel_bot', 0), ('sel_top', p['hc'])]:
        box = sel.create(tag, 'Box')
        box.set('entitydim', '2')
        _set_box_bounds(box, xy_bounds, xy_bounds, (z - tol, z + tol))
        box.set('condition', 'inside')
    core = sel.create('sel_core', 'Complement')
    core.set('entitydim', '2')
    core.set('input', ['sel_bot', 'sel_top'])
    for tag, axis, value in [
            ('edg_L', 'x', 0), ('edg_R', 'x', p['L']),
            ('edg_B', 'y', 0), ('edg_T', 'y', p['L'])]:
        edge = sel.create(tag, 'Box')
        edge.set('entitydim', '1')
        bounds = dict(x=xy_bounds, y=xy_bounds, z=z_bounds)
        bounds[axis] = (value - tol, value + tol)
        _set_box_bounds(edge, bounds['x'], bounds['y'], bounds['z'])
        edge.set('condition', 'inside')
    outer = sel.create('sel_outer_edges', 'Union')
    outer.set('entitydim', '1')
    outer.set('input', ['edg_L', 'edg_R', 'edg_B', 'edg_T'])
    for name in ('sel_top', 'sel_bot', 'sel_core', 'sel_outer_edges'):
        if len(comp.selection(name).entities()) == 0:
            raise RuntimeError(f'Empty geometry selection: {name}')
    physics = comp.physics().create('sh', 'Shell', 'geom1')
    for tag, selection, thickness, offset in [
            ('th_top', 'sel_top', 'h_top', '1'),
            ('th_bot', 'sel_bot', 'h_bottom', '-1'),
            ('th_core', 'sel_core', 'tc', None)]:
        feature = physics.create(tag, 'ThicknessOffset')
        feature.selection().named(selection)
        feature.set('d', thickness)
        if offset is not None:
            feature.set('OffsetDefinition', 'RelativeDistance')
            feature.set('z_offset_rel', offset)
    physics.create('fix1', 'Fixed', 1).selection().named('sel_outer_edges')
    mesh = comp.mesh().create('mesh1')
    tri = mesh.create('ftri1', 'FreeTri')
    tri.selection().all()
    tri.create('size1', 'Size').set('hauto', str(mesh_size))
    mesh.run()
    eig = m.study().create('std1').create('eig', 'Eigenfrequency')
    eig.set('neigs', str(candidate_eigs))
    eig.set('shift', f'{eigen_shift_hz}[Hz]')
    return model


def _extract_frequencies(numerical):
    feature = numerical.create('p1_freq', 'EvalGlobal')
    try:
        feature.set('data', 'dset1')
        feature.set('expr', ['freq'])
        feature.set('unit', ['Hz'])
        values = np.asarray(feature.getReal(), dtype=float).reshape(-1)
        if feature.isComplex():
            imag = np.asarray(feature.getImag(), dtype=float).reshape(-1)
            if not np.all(np.abs(imag) <= 1e-8 * np.maximum(1, np.abs(values))):
                raise ValueError('Significantly complex undamped eigenfrequencies')
        if not np.all(np.isfinite(values)):
            raise ValueError('Nonfinite eigenfrequencies')
        positive = np.flatnonzero(values > 0)
        indices = positive[np.argsort(values[positive], kind='stable')]
        return values[indices], indices + 1
    finally:
        numerical.remove('p1_freq')


def _extract_expression(numerical, tag, expression, solnums, coordinates,
                        candidate_count, point_count):
    feature = numerical.create(tag, 'Interp')
    try:
        feature.set('data', 'dset1')
        feature.selection().named('sel_top')
        feature.set('expr', [expression])
        feature.set('solnum', ','.join(str(int(i)) for i in solnums))
        feature.set('coord', coordinates.tolist())
        feature.set('coorderr', 'on')
        feature.set('matherr', 'on')
        feature.set('ext', 0.1)
        values = np.asarray(feature.getData(), dtype=float)
        if feature.isComplex():
            values = values + 1j * np.asarray(feature.getImagData(), dtype=float)
        if values.shape != (1, candidate_count, point_count):
            raise ValueError(f'Unexpected {expression} Interp shape {values.shape}')
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Nonfinite/missing top-face {expression} values; export rejected')
        return values[0]
    finally:
        numerical.remove(tag)


def solve_and_extract(model, p, solve=True, n_eigs=N_EIGS, grid_res=GRID_RES):
    if n_eigs < 1 or grid_res < 3:
        raise ValueError('n_eigs must be positive and grid_res must be at least 3')
    if solve:
        model.java.study('std1').run()
    numerical = model.java.result().numerical()
    frequencies, solnums = _extract_frequencies(numerical)
    if len(frequencies) < n_eigs:
        raise ValueError(f'Expected at least {n_eigs} positive modes, received {len(frequencies)}')
    # Exact outer-edge points can belong to edge entities rather than the top
    # shell boundary in COMSOL's geometric search.  Evaluate just inside the
    # rectangle and restore the known clamped-edge value explicitly.
    x = np.linspace(0, p['L'], grid_res)
    y = x.copy()
    eps = max(1e-9, 1e-6 * p['L'])
    x_eval = np.clip(x, eps, p['L'] - eps)
    y_eval = np.clip(y, eps, p['L'] - eps)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='xy')
    coordinates = np.vstack((X.ravel(), Y.ravel(), np.full(X.size, p['hc'])))
    candidate_count, point_count = len(solnums), X.size
    components = {
        name: _extract_expression(numerical, f'p1_interp_{name}', name, solnums,
                                  coordinates, candidate_count, point_count)
        for name in ('u', 'v', 'w')
    }
    energies = sum(np.abs(components[name]) ** 2 for name in ('u', 'v', 'w'))
    transverse_fraction = (
        np.sum(np.abs(components['w']) ** 2, axis=1)
        / np.maximum(np.sum(energies, axis=1), np.finfo(float).tiny))
    if not np.all(np.isfinite(transverse_fraction)):
        raise ValueError('Nonfinite transverse-energy fractions; export rejected')
    selected = np.flatnonzero(transverse_fraction >= MIN_TRANSVERSE_FRACTION)
    if len(selected) < n_eigs:
        raise ValueError(
            f'Only {len(selected)} flexural modes among {candidate_count}; '
            f'requested {n_eigs}')
    selected = selected[:n_eigs]
    raw_w = components['w'][selected].reshape(n_eigs, grid_res, grid_res).copy()
    w_peak_abs = np.max(np.abs(raw_w), axis=(1, 2))
    if np.any(w_peak_abs <= 0):
        raise ValueError('Zero transverse mode; export rejected')
    w = raw_w.copy()
    # CCCC boundary condition: w=0 on the perimeter.  These are exact
    # analytical values, not missing-data imputation.
    w[:, 0, :] = w[:, -1, :] = 0
    w[:, :, 0] = w[:, :, -1] = 0
    return dict(x=x, y=y, w=w, frequencies=frequencies[selected],
                solnums=solnums[selected], w_peak_abs=w_peak_abs,
                transverse_fraction=transverse_fraction[selected])


def metadata(parameters, run, mesh_size, n_eigs=N_EIGS,
             candidate_eigs=None, eigen_shift_hz=EIGEN_SHIFT_HZ):
    candidate_eigs = max(n_eigs, candidate_eigs or HF_CANDIDATE_MULTIPLIER * n_eigs)
    source = (Path(__file__).read_bytes()
              + Path(__file__).with_name('p1_geometry.py').read_bytes())
    config = dict(
        mesh_size=mesh_size, n_eigs=n_eigs, candidate_eigs=candidate_eigs,
        grid_res=GRID_RES, eigen_shift_hz=eigen_shift_hz, L=L_FIXED,
        material='Al:70e9,2710,0.33', boundary='CCCC',
        mode_selection=f'transverse_energy_fraction>={MIN_TRANSVERSE_FRACTION}')
    return dict(schema_version=2, fidelity='hf', parameters=parameters, run_id=run,
                model_version=MODEL_VERSION, surface='top', extraction='comsol-interp',
                source_hash=hashlib.sha256(source).hexdigest(), config=config,
                config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())


def compatible(path, meta):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if (json.loads(str(data['metadata'].item())) != meta
                    or data['w'].shape != (meta['config']['n_eigs'],
                                            meta['config']['grid_res'],
                                            meta['config']['grid_res'])):
                return False
            for axis in ('x', 'y'):
                if not np.array_equal(data[axis], np.linspace(0, L_FIXED, meta['config']['grid_res'])):
                    return False
            f = data['frequencies']
            solnums = data['solnums']
            peak = data['w_peak_abs']
            fraction = data['transverse_fraction']
            w = data['w']
            return bool(
                f.shape == (meta['config']['n_eigs'],)
                and solnums.shape == f.shape
                and peak.shape == f.shape
                and fraction.shape == f.shape
                and np.all(f > 0) and np.all(np.diff(f) >= 0)
                and np.all(np.isfinite(f)) and np.all(np.isfinite(solnums))
                and np.all(np.isfinite(peak)) and np.all(peak > 0)
                and np.all(np.isfinite(fraction))
                and np.all((fraction >= 0) & (fraction <= 1))
                and np.all(np.isfinite(w))
                and np.all(np.max(np.abs(w), axis=(1, 2)) > 0)
                and np.allclose(w[:, 0, :], 0) and np.allclose(w[:, -1, :], 0)
                and np.allclose(w[:, :, 0], 0) and np.allclose(w[:, :, -1], 0))
    except (OSError, ValueError, KeyError, TypeError):
        return False




def atomic_npz(path, payload, meta):
    fd,temp = tempfile.mkstemp(prefix=path.stem+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**payload,metadata=json.dumps(meta,sort_keys=True))
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def _close_client(client):
    if client is None:
        return
    for method in ('clear', 'disconnect'):
        try:
            getattr(client, method)()
        except Exception:
            pass


def _run_worker(args, samples, runs):
    """Process one batch in this interpreter; the parent replaces it per batch."""
    import mph

    out = Path(args.output) / 'hf'
    out.mkdir(parents=True, exist_ok=True)
    client = None
    failed = []
    try:
        client = mph.start(cores=args.cores)
        for run in runs:
            params = samples[run]
            physical = physical_parameters(params)
            meta = metadata(params, run, args.mesh_size, N_EIGS,
                            args.candidate_eigs, args.eigen_shift_hz)
            path = out / f'run_{run:04d}.npz'
            mph_path = path.with_suffix('.mph')
            sidecar = path.with_suffix('.model.json')
            model = None
            try:
                if not args.reexport and compatible(path, meta):
                    print(f'Run {run}: compatible export exists', flush=True)
                    continue
                if path.exists() and not compatible(path, meta):
                    if not args.replace:
                        raise ValueError(
                            f'{path} is incompatible. Use a new --output '
                            'or explicitly --replace.')
                    path.unlink()
                if args.reexport:
                    if (not mph_path.exists() or not sidecar.exists()
                            or json.loads(sidecar.read_text()) != meta):
                        raise ValueError(
                            f'Run {run}: no compatible p1-v2 model for reexport')
                    model = client.load(str(mph_path))
                else:
                    sidecar.unlink(missing_ok=True)
                    model = build_model(
                        client, physical, args.mesh_size, n_eigs=N_EIGS,
                        candidate_eigs=args.candidate_eigs,
                        eigen_shift_hz=args.eigen_shift_hz)
                    model.java.study('std1').run()
                    # Save only as an optional re-export artifact. NPZ is
                    # written atomically after extraction succeeds.
                    model.save(str(mph_path))
                    sidecar.write_text(
                        json.dumps(meta, sort_keys=True, indent=2),
                        encoding='utf-8')
                payload = solve_and_extract(
                    model, physical, solve=False,
                    n_eigs=N_EIGS, grid_res=GRID_RES)
                atomic_npz(path, payload, meta)
                print(f'Run {run}: saved {path}', flush=True)
            except Exception as error:
                failed.append(run)
                print(f'Run {run}: FAILED: {error}', flush=True)
            finally:
                if model is not None:
                    try:
                        client.remove(model)
                    except Exception as cleanup_error:
                        print(f'Run {run}: COMSOL cleanup skipped: '
                              f'{cleanup_error}', flush=True)
    except Exception as error:
        print(f'COMSOL worker startup failed: {error}', flush=True)
        failed.extend(run for run in runs if run not in failed)
    finally:
        _close_client(client)
    if failed:
        print(f'Worker failed runs: {sorted(set(failed))}', flush=True)
        return 1
    return 0


def _worker_command(args, batch, sample_path, output_path):
    command = [
        sys.executable, str(Path(__file__).resolve()),
        '--samples', str(sample_path), '--output', str(output_path),
        '--runs', ','.join(str(run) for run in batch),
        '--mesh-size', str(args.mesh_size), '--cores', str(args.cores),
        '--worker',
    ]
    if args.candidate_eigs is not None:
        command.extend(['--candidate-eigs', str(args.candidate_eigs)])
    command.extend(['--eigen-shift-hz', str(args.eigen_shift_hz)])
    if args.reexport:
        command.append('--reexport')
    if args.replace:
        command.append('--replace')
    return command


def _run_orchestrator(args, samples, runs, root):
    """Run each batch in a fresh Python process to isolate MPh/JVM state."""
    sample_path = Path(args.samples)
    output_path = Path(args.output)
    failed = []
    for start in range(0, len(runs), args.batch_size):
        batch = runs[start:start + args.batch_size]
        success = False
        command = _worker_command(args, batch, sample_path, output_path)
        for attempt in range(1, args.max_attempts + 1):
            print(f'Worker batch {batch}: process attempt '
                  f'{attempt}/{args.max_attempts}', flush=True)
            try:
                result = subprocess.run(command, cwd=str(root))
            except OSError as error:
                print(f'Worker process failed to start: {error}', flush=True)
                continue
            if result.returncode == 0:
                success = True
                break
        if not success:
            failed.extend(batch)
            print(f'Worker batch {batch}: failed after '
                  f'{args.max_attempts} fresh processes', flush=True)
    if failed:
        raise RuntimeError(f'Unfinished HF runs after retries: '
                           f'{sorted(set(failed))}')
    print(f'All {len(runs)} requested runs have valid or newly-created '
          'NPZ bundles.', flush=True)
    return 0


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=root / 'lhs_samples_v2.csv')
    parser.add_argument('--output', type=Path, default=root / 'p1_data')
    parser.add_argument('--runs', help='1-based sample IDs, e.g. 1 or 1,2')
    parser.add_argument('--mesh-size', type=int, choices=range(1, 10), default=4)
    parser.add_argument('--candidate-eigs', type=int,
                        help='Positive eigenmodes to solve before flexural filtering')
    parser.add_argument('--eigen-shift-hz', type=float, default=EIGEN_SHIFT_HZ,
                        help='COMSOL eigenfrequency shift; validated default is 1000 Hz')
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Runs per fresh Python/COMSOL worker process')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='Maximum fresh worker processes per batch')
    parser.add_argument('--reexport', action='store_true')
    parser.add_argument('--replace', action='store_true',
                        help='Replace incompatible p1 NPZ exports')
    parser.add_argument('--preflight', action='store_true',
                        help='CPU checks; no COMSOL or output writes')
    parser.add_argument('--worker', action='store_true',
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.cores < 1 or args.batch_size < 1 or args.max_attempts < 1
            or (args.candidate_eigs is not None and args.candidate_eigs < N_EIGS)
            or not np.isfinite(args.eigen_shift_hz)):
        parser.error('Invalid COMSOL resource, candidate-eigenvalue, or shift setting')
    args.samples = args.samples.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.samples.is_file():
        parser.error(f'Sample table not found: {args.samples}')
    with args.samples.open(newline='', encoding='utf-8-sig') as stream:
        samples = {}
        for index, row in enumerate(csv.DictReader(stream), 1):
            run = int(row.get('run_id', row.get('run', index)))
            if run < 1 or run in samples:
                parser.error('Sample IDs must be unique positive integers')
            samples[run] = {k: float(row[k]) for k in PARAM_NAMES}
    try:
        runs = (list(dict.fromkeys(int(v) for v in args.runs.split(',')))
                if args.runs else list(samples))
    except ValueError:
        parser.error('--runs must contain comma-separated integers')
    if not runs or any(run not in samples for run in runs):
        parser.error('Invalid sample IDs')
    for run in runs:
        preflight(samples[run])
    if args.preflight:
        print(f'Geometry preflight passed for {len(runs)} samples. '
              'No COMSOL execution.')
        return 0
    if args.worker:
        return _run_worker(args, samples, runs)
    return _run_orchestrator(args, samples, runs, root)


if __name__ == '__main__':
    raise SystemExit(main())
