"""Frozen-model prospective P1 acquisition, calibration, evaluation and screening.

No prospective geometry is generated without model_freeze.json. LF-only
predictions and calibrated screening decisions precede evaluation HF solves.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from p1_improve import ROOT, DATA, SEED, preserve_stage_source, utc
from p1_design import file_sha256, read_design_table, write_design_table, write_json


@dataclass
class LFInputs:
    """Real LF inputs only; unavailable HF labels are not represented by zeros."""
    x: object
    y: object
    boundary_mask: object
    interior_mask: object
    lf: object
    run_ids: object
    mode_ids: object
    parameters: object
    parameter_names: tuple[str, ...]
    f_lf: object

    @property
    def n_rows(self):
        return len(self.run_ids)

    @property
    def grid_shape(self):
        return len(self.y), len(self.x)

    @classmethod
    def load(cls, path):
        import numpy as np
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        values['parameter_names'] = tuple(str(name) for name in values['parameter_names'])
        return cls(**values)


def physical_source_hashes():
    paths = [ROOT/name for name in ('hc_HighFidelity_LHS.py', 'p1_geometry.py',
        'fsdt_mode_shapes.py', 'honeycomb.py', 'p1_surrogate.py', 'p1_pairing.py',
        'p1_design.py', 'p1_holdout.py', 'run_hf_batches.py')]
    paths += sorted((ROOT/'plate').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def freeze_reference(output, parameter_names):
    import numpy as np
    from p1_pairing import align_real, load_run
    destination = output/'frozen_reference.npz'
    if destination.exists():
        raise FileExistsError('Frozen mode reference already exists')
    with (DATA/'corrections_five/pairing.csv').open(newline='', encoding='utf-8') as stream:
        audit = list(csv.DictReader(stream))
    manifest = json.loads((DATA/'corrections_five/manifest.json').read_text())
    shapes, frequencies, reference_runs, indices, details = [], [], [], [], []
    for mode in range(1, manifest['requested_modes']+1):
        identities = {(int(row['reference_run']), int(row['reference_lf_mode'])-1)
                      for row in audit if int(row['mode']) == mode}
        if len(identities) != 1:
            raise ValueError(f'Historical label{mode} has inconsistent reference identity')
        run, index = identities.pop()
        source = DATA/'lf'/f'run_{run:04d}.npz'
        bundle = load_run(source, 'lf')
        if shapes:
            np.testing.assert_array_equal(bundle['x'], x)
            np.testing.assert_array_equal(bundle['y'], y)
        x, y = bundle['x'], bundle['y']
        shapes.append(align_real(bundle['w'][index]))
        frequencies.append(bundle['f'])
        reference_runs.append(run); indices.append(index)
        details.append({'mode': mode, 'reference_run': run, 'reference_lf_mode': index+1,
                        'source_bundle_sha256': file_sha256(source)})
    np.savez_compressed(destination, x=x, y=y, reference_shapes=np.stack(shapes),
        reference_frequency_vectors=np.stack(frequencies), reference_run_ids=reference_runs,
        reference_indices=indices, parameter_names=np.array(parameter_names))
    rules = {'requested_modes': manifest['requested_modes'], 'min_mac': manifest['min_mac'],
        'min_margin': .05, 'degeneracy_gap': .005,
        'min_hf_transverse_fraction': manifest['hf_quality_policy']['min_transverse_energy_fraction'],
        'source': 'Original target-builder defaults verified by the byte-identical audited721-row rebuild; thresholds missing from the old manifest are now explicit.',
        'gauge': 'Reuse each original stable LF reference and peak/sign convention; do not select a reference from prospective cases.',
        'references': details, 'parameter_names': list(parameter_names),
        'pairing_csv_sha256': file_sha256(DATA/'corrections_five/pairing.csv'),
        'historical_manifest_sha256': file_sha256(DATA/'corrections_five/manifest.json'),
        'physical_source_hashes': physical_source_hashes()}
    write_json(output/'frozen_reference.json', rules)
    return {**rules, 'artifact_hashes': {'frozen_reference.npz': file_sha256(destination),
                                       'frozen_reference.json': file_sha256(output/'frozen_reference.json')}}


def check_freeze(output, verify_artifacts=True):
    frozen = json.loads((output/'model_freeze.json').read_text())
    if file_sha256(output/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    for name, expected in frozen['method_source_hashes'].items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Prospective method source changed after freeze: {name}')
    if physical_source_hashes() != frozen['reference']['physical_source_hashes']:
        raise ValueError('Original physical solver/pairing sources changed after freeze')
    if verify_artifacts:
        for name, expected in frozen['artifact_hashes'].items():
            if file_sha256(output/name) != expected:
                raise ValueError(f'Frozen artifact changed: {name}')
    return frozen


def plan_rows(output):
    with (output/'prospective/plan.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    from p1_geometry import PARAM_NAMES
    return [{'run': int(row['run_id']), 'group': row['group'], 'role': row['role'],
             'parameters': {name: float(row[name]) for name in PARAM_NAMES}}
            for row in rows]


def generate_plan(output):
    import numpy as np
    from lhs_sampling import PARAM_RANGES
    from p1_geometry import PARAM_NAMES
    frozen = check_freeze(output)
    directory = output/'prospective'
    directory.mkdir(exist_ok=False)
    protocol = json.loads((output/'protocol.json').read_text())['prospective']
    old = read_design_table(DATA/'lhs_five.csv', PARAM_NAMES)
    old_geometries = {tuple(round(params[name], 12) for name in PARAM_NAMES) for _, params in old}
    seen = set(old_geometries)
    lower = np.array([PARAM_RANGES[name][0] for name in PARAM_NAMES])
    upper = np.array([PARAM_RANGES[name][1] for name in PARAM_NAMES])
    next_run = max(run for run, _ in old)+1
    design, plan = [], []
    for channel, (group, count, role) in enumerate((
        ('calibration', protocol['calibration_count'], 'calibration'),
        ('iid', protocol['iid_test_count'], 'test'),
        ('challenge', protocol['excluded_region_challenge_count'], 'test'))):
        bounds = lower.copy()
        if group == 'challenge':
            bounds[PARAM_NAMES.index('theta_c')] = 60.
        values = np.random.default_rng(np.random.SeedSequence([SEED, 610, channel])).uniform(bounds, upper, (count, len(PARAM_NAMES)))
        for vector in values:
            params = dict(zip(PARAM_NAMES, map(float, vector)))
            signature = tuple(round(params[name], 12) for name in PARAM_NAMES)
            if signature in seen:
                raise ValueError('Prospective geometry duplicates an old or new design; not silently redrawing')
            seen.add(signature)
            design.append((next_run, params))
            plan.append({'run_id': next_run, 'role': role, 'group': group, **params})
            next_run += 1
    write_design_table(directory/'design.csv', design, PARAM_NAMES)
    with (directory/'plan.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('run_id', 'role', 'group', *PARAM_NAMES))
        writer.writeheader(); writer.writerows(plan)
    write_json(directory/'generation.json', {'generated_at_utc': utc(),
        'model_freeze_sha256': file_sha256(output/'model_freeze.json'),
        'model_frozen_at_utc': frozen['frozen_at_utc'], 'old_geometries_checked': len(old_geometries),
        'duplicate_identity_rounding_digits': 12, 'groups': {g: sum(r['group'] == g for r in plan) for g in ('calibration', 'iid', 'challenge')},
        'random_streams': {'calibration': [SEED, 610, 0], 'iid': [SEED, 610, 1], 'challenge': [SEED, 610, 2]},
        'distribution': 'Independent physical-uniform draws; IID calibration/test identical, unlike the historical density-transformed LHS.',
        'design_sha256': file_sha256(directory/'design.csv'), 'plan_sha256': file_sha256(directory/'plan.csv')})
    print(json.dumps({'prospective_geometries_generated_after_freeze': len(plan),
                      'groups': {g: sum(r['group'] == g for r in plan) for g in ('calibration', 'iid', 'challenge')}}), flush=True)


def lf_campaign(output):
    from run_hf_batches import _supervise
    check_freeze(output)
    directory = output/'prospective'
    command = [sys.executable, '-u', str(ROOT/'fsdt_mode_shapes.py'), '--samples', str(directory/'design.csv'),
               '--output', str(directory), '--modes', '16', '--order', '15', '--grid', '80']
    return _supervise(command, ROOT, os.environ.copy(), directory/'lf_worker.log', 3600, 60)


def reference_arrays(output):
    import numpy as np
    with np.load(output/'frozen_reference.npz', allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def make_lf_inputs(output, runs, destination):
    import numpy as np
    from p1_pairing import align_real, assign, assignment_margin, boundary_mask, load_run, repeated
    reference = reference_arrays(output)
    rules = json.loads((output/'frozen_reference.json').read_text())
    names = tuple(str(name) for name in reference['parameter_names'])
    fields, ids, modes, parameters, frequencies, audit = [], [], [], [], [], []
    for run in runs:
        low = load_run(output/'prospective/lf'/f'run_{run:04d}.npz', 'lf')
        np.testing.assert_array_equal(low['x'], reference['x'])
        np.testing.assert_array_equal(low['y'], reference['y'])
        mapping, scores = assign(reference['reference_shapes'], low['w'])
        for index, low_mode in enumerate(mapping):
            reasons = []
            mac = float(scores[index, low_mode])
            margin = float(assignment_margin(scores, index, low_mode))
            if mac < rules['min_mac']: reasons.append('low_MAC')
            if margin < rules['min_margin']: reasons.append('ambiguous_assignment')
            if (repeated(reference['reference_frequency_vectors'][index], reference['reference_indices'][index], rules['degeneracy_gap'])
                    or repeated(low['f'], low_mode, rules['degeneracy_gap'])):
                reasons.append('near_repeated_eigenvalue_use_subspace_target')
            row = {'run': int(run), 'mode': index+1, 'lf_mode': int(low_mode)+1,
                   'tracking_mac': mac, 'tracking_margin': margin, 'f_lf': float(low['f'][low_mode])}
            try:
                field = align_real(low['w'][low_mode], reference['reference_shapes'][index])
            except ValueError as error:
                reasons.append(str(error))
            else:
                fields.append(field); ids.append(run); modes.append(index+1)
                parameters.append([low['meta']['parameters'][name] for name in names])
                frequencies.append(low['f'][low_mode])
            row.update(trackable=not reasons, reasons=reasons)
            audit.append(row)
    if not fields:
        raise ValueError('No real LF input can be phase-aligned to the frozen references')
    mask = boundary_mask(reference['x'], reference['y'])
    np.savez_compressed(destination, x=reference['x'], y=reference['y'], boundary_mask=mask,
        interior_mask=mask > 0, lf=np.stack(fields), run_ids=np.array(ids), mode_ids=np.array(modes),
        parameters=np.array(parameters), parameter_names=np.array(names), f_lf=np.array(frequencies))
    write_json(destination.with_suffix('.json'), {'created_at_utc': utc(), 'audit': audit,
        'frozen_reference_sha256': file_sha256(output/'frozen_reference.npz'),
        'lf_only': True, 'hf_labels_available_to_predictor': False,
        'input_sha256': file_sha256(destination)})
    return LFInputs.load(destination)


def blind_predictions(output):
    import numpy as np
    from p1_improved_models import load_model
    frozen = check_freeze(output)
    directory = output/'prospective'
    if any((directory/'hf').glob('run_*.npz')):
        raise ValueError('HF exports exist before blinded LF-only predictions')
    target = directory/'blinded_predictions'
    target.mkdir(exist_ok=False)
    inputs = make_lf_inputs(output, [row['run'] for row in plan_rows(output)], directory/'lf_inputs.npz')
    rows = np.arange(inputs.n_rows)
    timings = {}
    for name in frozen['models']:
        model = load_model(output/'frozen_models'/name)
        if set(map(int, model.training_runs)) & set(map(int, inputs.run_ids)):
            raise ValueError('Prospective geometry leaked into fitted model')
        started = time.monotonic()
        prediction = model.predict_rows(inputs, rows)
        elapsed = time.monotonic()-started
        np.savez_compressed(target/f'{name}.npz', run_ids=inputs.run_ids, mode_ids=inputs.mode_ids, **prediction)
        timings[name] = {'prediction_seconds': elapsed, 'input_rows': inputs.n_rows,
                         'prediction_sha256': file_sha256(target/f'{name}.npz')}
        print(f'BLIND {name}: {inputs.n_rows} LF-only rows; no HF labels; {elapsed:.3f}s', flush=True)
    write_json(target/'manifest.json', {'predicted_at_utc': utc(), 'models': timings,
        'lf_input_sha256': file_sha256(directory/'lf_inputs.npz'),
        'model_freeze_sha256': file_sha256(output/'model_freeze.json'),
        'before_any_prospective_HF_export': True})


def native_metadata(job):
    import hc_HighFidelity_LHS as hf
    config = job['native_settings']
    metadata = hf.metadata(job['parameters'], job['run'], config['mesh_size'],
                           config['n_eigs'], config['candidate_eigs'], config['eigen_shift_hz'])
    metadata['study'] = {'group': job['group'], 'model_freeze_sha256': job['model_freeze_sha256'],
        'plan_sha256': job['plan_sha256'], 'worker_source_sha256': file_sha256(__file__),
        'full_native_model_saved': False}
    return metadata


def hf_worker(job_path, destination, cores):
    import mph
    import hc_HighFidelity_LHS as hf
    from p1_geometry import physical_parameters
    from simulations.scripts.hc_mesh_convergence import _element_count
    job = json.loads(job_path.read_text())
    path = destination/'hf'/f'run_{job["run"]:04d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = native_metadata(job)
    if path.exists():
        raise FileExistsError('Native worker refuses to replace an HF export')
    mph.option('session', 'stand-alone')
    client = None
    total_started = time.monotonic()
    try:
        client = mph.start(cores=cores)
        client.java.showProgress(os.environ['P1_NATIVE_PROGRESS'])
        parameters = physical_parameters(job['parameters'])
        config = job['native_settings']
        started = time.monotonic()
        print(f'PHASE geometry-and-mesh run{job["run"]} started', flush=True)
        model = hf.build_model(client, parameters, config['mesh_size'], config['n_eigs'],
                               config['candidate_eigs'], config['eigen_shift_hz'])
        geometry_seconds = time.monotonic()-started
        elements = _element_count(model)
        print(f'PHASE geometry-and-mesh finished {geometry_seconds:.3f}s; elements={elements}; eigensolve started', flush=True)
        started = time.monotonic()
        model.java.study('std1').run()
        solve_seconds = time.monotonic()-started
        print(f'PHASE eigensolve finished {solve_seconds:.3f}s; field-extraction started', flush=True)
        started = time.monotonic()
        payload = hf.solve_and_extract(model, parameters, solve=False, n_eigs=config['n_eigs'], grid_res=80)
        extraction_seconds = time.monotonic()-started
        hf.atomic_npz(path, payload, metadata)
        if not hf.compatible(path, metadata):
            raise ValueError('Completed native export failed the existing full bundle contract')
        write_json(path.with_suffix('.timing.json'), {'run': job['run'], 'group': job['group'],
            'geometry_mesh_seconds': geometry_seconds, 'eigensolve_seconds': solve_seconds,
            'field_extraction_seconds': extraction_seconds, 'worker_total_seconds': time.monotonic()-total_started,
            'elements': elements, 'cores': cores, 'npz_sha256': file_sha256(path),
            'reference': 'Original nominal hauto4, 32 candidates/16 flexural exports, shift1000; no new physics or solver override.'})
        print(f'PHASE saved {path}; no full MPH saved', flush=True)
        return 0
    finally:
        hf._close_client(client)


def supervise_native_job(output, job, directory, settings):
    from run_hf_batches import _supervise
    directory.mkdir(parents=True, exist_ok=True)
    log_directory = directory/'worker_logs'
    log_directory.mkdir(exist_ok=True)
    path = log_directory/f'run_{job["run"]}.job.json'
    log = log_directory/f'run_{job["run"]}.log'
    if path.exists() or log.exists():
        raise FileExistsError(f'Native attempt already exists for run{job["run"]}; no unchanged automatic retry')
    write_json(path, job)
    if shutil.disk_usage(settings['scratch_root']).free < 8*2**30:
        raise OSError('Less than8GiB free on the declared native scratch drive')
    with TemporaryDirectory(prefix='p1_prospective_', dir=settings['scratch_root']) as scratch:
        environment = os.environ.copy()
        environment['TEMP'] = environment['TMP'] = scratch
        options = re.sub(r'(?<!\S)-Xmx\S+', '', environment.get('JAVA_TOOL_OPTIONS', '')).strip()
        environment['JAVA_TOOL_OPTIONS'] = f'{options} -Xmx{settings["java_heap_gib"]}g -Djava.io.tmpdir="{Path(scratch).as_posix()}"'.strip()
        environment['P1_NATIVE_PROGRESS'] = str(log.with_suffix('.progress.log'))
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'hf', '--output', str(output),
                   '--worker', '--job', str(path), '--destination', str(directory), '--cores', str(settings['native_cores'])]
        started = utc()
        status = _supervise(command, ROOT, environment, log,
                            settings['native_worker_deadline_minutes']*60, settings['heartbeat_seconds'])
    return {'run': job['run'], 'group': job['group'], 'exit_code': status,
            'started_at_utc': started, 'finished_at_utc': utc(), 'owned_scratch_removed': not Path(scratch).exists()}


def hf_campaign(output, group):
    import hc_HighFidelity_LHS as hf
    frozen = check_freeze(output)
    directory = output/'prospective'
    if not (directory/'blinded_predictions/manifest.json').is_file():
        raise ValueError('Blinded LF-only predictions must precede any HF acquisition')
    if group == 'evaluation' and not (directory/'screening_decisions.json').is_file():
        raise ValueError('Calibrated decisions must be stored before evaluation HF acquisition')
    rows = [row for row in plan_rows(output) if (row['group'] == 'calibration') == (group == 'calibration')]
    settings = json.loads((output/'protocol.json').read_text())['resources']
    outcomes = []
    manifest = directory/f'hf_{group}_execution.json'
    if manifest.exists():
        raise FileExistsError('Native campaign already has outcomes; no automatic retry')
    for row in rows:
        job = {**row, 'native_settings': frozen['native_reference_settings'],
            'model_freeze_sha256': file_sha256(output/'model_freeze.json'),
            'plan_sha256': file_sha256(directory/'plan.csv')}
        path = directory/'hf'/f'run_{row["run"]:04d}.npz'
        if path.exists():
            raise FileExistsError('A prospective HF export exists without this campaign manifest')
        outcome = supervise_native_job(output, job, directory, settings)
        outcomes.append(outcome)
        write_json(manifest, {'group': group, 'one_attempt_per_geometry': True,
            'completed': len(outcomes), 'requested': len(rows), 'outcomes': outcomes,
            'model_freeze_sha256': job['model_freeze_sha256']}, overwrite=True)
    return 1 if any(outcome['exit_code'] for outcome in outcomes) else 0


def native_smoke(output, run):
    import numpy as np
    from p1_pairing import align_real, assign, load_run
    from p1_geometry import PARAM_NAMES
    directory = output/f'native_method_control_{run}'
    if directory.exists():
        raise FileExistsError('A native method control already exists')
    parameters = dict(read_design_table(DATA/'lhs_five.csv', PARAM_NAMES))[run]
    config = {'mesh_size': 4, 'n_eigs': 16, 'candidate_eigs': 32, 'eigen_shift_hz': 1000., 'solver_override': None}
    job = {'run': run, 'parameters': parameters, 'group': 'historical_method_control',
           'native_settings': config, 'model_freeze_sha256': None, 'plan_sha256': None}
    settings = json.loads((output/'protocol.json').read_text())['resources']
    outcome = supervise_native_job(output, job, directory, settings)
    if outcome['exit_code']:
        write_json(directory/'control.json', {'outcome': outcome, 'reference_verified': False})
        return outcome['exit_code']
    baseline = load_run(DATA/'hf'/f'run_{run:04d}.npz', 'hf')
    current = load_run(directory/'hf'/f'run_{run:04d}.npz', 'hf')
    mapping, scores = assign(baseline['w'], current['w'])
    records = []
    for index, matched in enumerate(mapping):
        difference = align_real(current['w'][matched], align_real(baseline['w'][index]))-align_real(baseline['w'][index])
        records.append({'mode': index+1, 'matched_mode': int(matched)+1,
            'mac': float(scores[index, matched]),
            'frequency_error_pct': float(100*abs(current['f'][matched]-baseline['f'][index])/baseline['f'][index]),
            'field_rms': float(np.sqrt(np.mean(difference**2)))})
    passed = max(r['frequency_error_pct'] for r in records) <= .1 and max(r['field_rms'] for r in records) <= .00075
    write_json(directory/'control.json', {'outcome': outcome, 'reference_verified': bool(passed),
        'acceptance': {'frequency_pct': .1, 'field_rms': .00075}, 'rows': records,
        'scope': 'One historical geometry verifies the new no-MPH acquisition path against the old nominal export; not universal discretization certification.'})
    print(json.dumps({'native_acquisition_control_passed': bool(passed),
        'maximum_frequency_error_pct': max(r['frequency_error_pct'] for r in records),
        'maximum_field_rms': max(r['field_rms'] for r in records)}), flush=True)
    return 0 if passed else 1


def paired_targets(output, group):
    import numpy as np
    from p1_pairing import (align_real, assign, assignment_margin, boundary_mask,
                            check_pair, load_run, repeated)
    from p1_surrogate import P1Dataset, write_records
    directory = output/'prospective'
    target = directory/f'{group}_targets'
    target.mkdir(exist_ok=False)
    reference = reference_arrays(output)
    rules = json.loads((output/'frozen_reference.json').read_text())
    names = tuple(str(name) for name in reference['parameter_names'])
    inputs = LFInputs.load(directory/'lf_inputs.npz')
    input_index = {(int(run), int(mode)): index for index, (run, mode) in enumerate(zip(inputs.run_ids, inputs.mode_ids))}
    rows = [row for row in plan_rows(output) if (row['group'] == 'calibration') == (group == 'calibration')]
    accepted, audit = [], []
    for plan in rows:
        run = plan['run']
        path = directory/'hf'/f'run_{run:04d}.npz'
        if not path.exists():
            audit.extend({'run': run, 'mode': mode, 'group': plan['group'],
                          'status': 'missing_HF', 'reason': 'Native acquisition failed; no fabricated label'}
                         for mode in range(1, rules['requested_modes']+1))
            continue
        low, high = load_run(directory/'lf'/f'run_{run:04d}.npz', 'lf'), load_run(path, 'hf')
        check_pair(low, high)
        low_map, tracking = assign(reference['reference_shapes'], low['w'])
        high_map, pairing = assign(low['w'][low_map], high['w'])
        for label, (li, hi) in enumerate(zip(low_map, high_map), 1):
            li, hi = int(li), int(hi)
            row = {'run': run, 'mode': label, 'group': plan['group'], 'lf_mode': li+1, 'hf_mode': hi+1,
                'tracking_mac': float(tracking[label-1, li]), 'pair_mac': float(pairing[label-1, hi]),
                'tracking_margin': float(assignment_margin(tracking, label-1, li)),
                'pair_margin': float(assignment_margin(pairing, label-1, hi)),
                'f_fsdt': float(low['f'][li]), 'f_comsol': float(high['f'][hi]),
                'hf_transverse_fraction': float(high['transverse_fraction'][hi]),
                'hf_w_peak_ratio_diagnostic': float(high['w_peak_ratio'][hi])}
            reasons = []
            if min(row['tracking_mac'], row['pair_mac']) < rules['min_mac']: reasons.append('low_MAC')
            if min(row['tracking_margin'], row['pair_margin']) < rules['min_margin']: reasons.append('ambiguous_assignment')
            index = label-1
            if (repeated(reference['reference_frequency_vectors'][index], reference['reference_indices'][index], rules['degeneracy_gap'])
                    or repeated(low['f'], li, rules['degeneracy_gap']) or repeated(high['f'], hi, rules['degeneracy_gap'])):
                reasons.append('near_repeated_eigenvalue_use_subspace_target')
            if row['hf_transverse_fraction'] < rules['min_hf_transverse_fraction']: reasons.append('low_HF_transverse_energy_fraction')
            key = (run, label)
            if key not in input_index:
                reasons.append('No phase-valid blinded LF input')
            if not reasons:
                lshape = align_real(low['w'][li], reference['reference_shapes'][index])
                np.testing.assert_allclose(lshape, inputs.lf[input_index[key]], rtol=0, atol=1e-13)
                hshape = align_real(high['w'][hi], lshape)
                accepted.append((row, lshape, hshape, [plan['parameters'][name] for name in names]))
            row.update(status='quarantined' if reasons else 'accepted', reason=';'.join(reasons))
            audit.append(row)
    write_records(target/'pairing.csv', audit)
    if not accepted:
        write_json(target/'manifest.json', {'accepted': 0, 'requested_geometries': len(rows), 'audit': audit})
        raise ValueError('No prospective accepted HF labels; cannot calibrate or evaluate')
    mask = boundary_mask(reference['x'], reference['y'])
    lf, hf = np.stack([a[1] for a in accepted]), np.stack([a[2] for a in accepted])
    np.savez_compressed(target/'training.npz', x=reference['x'], y=reference['y'], boundary_mask=mask,
        interior_mask=mask > 0, lf=lf, hf=hf, correction=hf-lf,
        run_ids=np.array([a[0]['run'] for a in accepted]), mode_ids=np.array([a[0]['mode'] for a in accepted]),
        parameters=np.array([a[3] for a in accepted]), parameter_names=np.array(names),
        f_lf=np.array([a[0]['f_fsdt'] for a in accepted]), f_hf=np.array([a[0]['f_comsol'] for a in accepted]))
    write_json(target/'manifest.json', {'created_at_utc': utc(), 'role': group,
        'accepted': len(accepted), 'quarantined_or_missing': len(audit)-len(accepted),
        'requested_geometries': len(rows), 'accepted_geometries': len({a[0]['run'] for a in accepted}),
        'frozen_reference_sha256': file_sha256(output/'frozen_reference.npz'),
        'training_filename_is_an_inherited_data_contract_not_training_permission': True})
    return P1Dataset.load(target)


def saved_prediction(output, model_name, dataset):
    import numpy as np
    with np.load(output/'prospective/blinded_predictions'/f'{model_name}.npz', allow_pickle=False) as saved:
        index = {(int(run), int(mode)): i for i, (run, mode) in enumerate(zip(saved['run_ids'], saved['mode_ids']))}
        selected = np.array([index[(int(run), int(mode))] for run, mode in zip(dataset.run_ids, dataset.mode_ids)])
        return {key: saved[key][selected] for key in ('correction', 'correction_std', 'frequency', 'frequency_std')}


def scaled_prediction(output, name, inputs, rows, prediction, frozen, recipe=None):
    from p1_improvement_study import LocalResidualScale, raw_scale
    regional = name == 'region_selected'
    selected = frozen['region_uncertainty'] if regional else frozen['selected_uncertainty']
    chosen = recipe or (selected if regional or name == frozen['selected_mean'] else 'raw')
    if chosen == 'raw':
        return raw_scale(prediction, inputs.interior_mask)
    scale_directory = output/('region_uncertainty_selection' if regional else 'uncertainty_selection')/'final_local_scale'
    return LocalResidualScale.load(scale_directory).predict(inputs, rows, prediction)


def mode1_calibration(dataset, rows, prediction):
    import numpy as np
    from p1_holdout import _finite_quantile
    rows = np.asarray(rows, dtype=int)
    local = np.flatnonzero(dataset.mode_ids[rows] == 1)
    errors = np.abs(prediction['frequency'][local]-dataset.f_hf[rows[local]])
    sigma = prediction['frequency_std'][local]
    if not np.isfinite(errors).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Invalid frequency calibration errors/scales')
    scores = errors/sigma
    multipliers, unsupported = {}, {}
    for coverage, minimum in ((90, 9), (95, 19)):
        if len(scores) < minimum:
            unsupported[str(coverage)] = f'Need{minimum} first-mode calibration geometries; have{len(scores)}'
        else:
            multipliers[str(coverage)] = _finite_quantile(scores, coverage)
    return {'quantity': 'Only the predeclared stable LF label1 frequency; not the complete structural spectrum',
        'runs': dataset.run_ids[rows[local]].astype(int).tolist(), 'scores': scores.tolist(),
        'multipliers': multipliers, 'unsupported': unsupported}


def calibrate(output):
    import numpy as np
    from p1_improvement_study import conformal_state
    frozen = check_freeze(output)
    directory = output/'prospective'
    dataset = paired_targets(output, 'calibration')
    all_rows = np.arange(dataset.n_rows)
    theta = dataset.parameter_names.index('theta_c')
    report = {'calibrated_at_utc': utc(), 'models': {}, 'uncertainty_recipe_fixed_before_calibration': True}
    for name in frozen['models']:
        rows = np.flatnonzero(dataset.parameters[:, theta] < 60.) if name == 'region_selected' else all_rows
        base = saved_prediction(output, name, dataset)
        prediction = {key: values[rows] for key, values in base.items()}
        prediction = scaled_prediction(output, name, dataset, rows, prediction, frozen)
        report['models'][name] = {'run_simultaneous': conformal_state(dataset, rows, prediction),
                                 'screening_mode1': mode1_calibration(dataset, rows, prediction)}
        if name == frozen['selected_mean']:
            report['selected_mean_recipe_diagnostics'] = {}
            for recipe in ('raw', 'local_residual'):
                alternate = scaled_prediction(output, name, dataset, all_rows, base, frozen, recipe)
                report['selected_mean_recipe_diagnostics'][recipe] = conformal_state(dataset, all_rows, alternate)
    lf_prediction = {'frequency': dataset.f_lf, 'frequency_std': dataset.f_lf}
    report['lf_only_screening'] = mode1_calibration(dataset, all_rows, lf_prediction)
    write_json(directory/'calibration.json', report)
    write_screening_decisions(output, frozen, report)
    return report


def screening_action(mean, half_width, threshold):
    if mean-half_width >= threshold:
        return 'accept'
    if mean+half_width < threshold:
        return 'reject'
    return 'HF'


def write_screening_decisions(output, frozen, calibration):
    import numpy as np
    directory = output/'prospective'
    plan = [row for row in plan_rows(output) if row['role'] == 'test']
    if any((directory/'hf'/f'run_{row["run"]:04d}.npz').exists() for row in plan):
        raise ValueError('Evaluation HF exists before engineering decisions are frozen')
    inputs = LFInputs.load(directory/'lf_inputs.npz')
    index = {(int(run), int(mode)): i for i, (run, mode) in enumerate(zip(inputs.run_ids, inputs.mode_ids))}
    audit = json.loads((directory/'lf_inputs.json').read_text())['audit']
    trackable = {(row['run'], row['mode']): row['trackable'] for row in audit}
    threshold = frozen['frequency_screen_threshold_hz']
    decisions = []
    for name in [*frozen['models'], 'lf_only']:
        if name == 'lf_only':
            mean = inputs.f_lf
            sigma = inputs.f_lf
            state = calibration['lf_only_screening']
        else:
            with np.load(directory/'blinded_predictions'/f'{name}.npz', allow_pickle=False) as saved:
                base = {key: saved[key] for key in ('correction', 'correction_std', 'frequency', 'frequency_std')}
            predicted = scaled_prediction(output, name, inputs, np.arange(inputs.n_rows), base, frozen)
            mean, sigma = predicted['frequency'], predicted['frequency_std']
            state = calibration['models'][name]['screening_mode1']
        q = state['multipliers'].get('95')
        for row in plan:
            key = (row['run'], 1)
            valid = key in index and trackable.get(key, False) and q is not None
            i = index.get(key)
            half_width = float(q*sigma[i]) if valid else None
            action = screening_action(float(mean[i]), half_width, threshold) if valid else 'HF'
            decisions.append({'run': row['run'], 'group': row['group'], 'policy': name,
                'decision': action, 'predicted_frequency_hz': float(mean[i]) if i is not None else None,
                'half_width_hz': half_width, 'lf_identity_trackable': trackable.get(key, False),
                'finite95_supported': q is not None})
    write_json(directory/'screening_decisions.json', {'recorded_at_utc': utc(),
        'before_evaluation_HF': True, 'threshold_hz': threshold, 'confidence': 95,
        'task': 'Stable reference mode1 frequency screening; benchmark threshold, not universal service certification',
        'decision_rule': 'Lower>=threshold accept; upper<threshold reject; otherwise HF',
        'calibration_sha256': file_sha256(directory/'calibration.json'), 'decisions': decisions})
    print(json.dumps({'calibration_finished': True, 'engineering_decisions_frozen_before_evaluation_HF': len(decisions)}), flush=True)


def engineering_results(output, dataset, frozen):
    import numpy as np
    from p1_improvement_study import coverage_interval
    directory = output/'prospective'
    frozen_decisions = json.loads((directory/'screening_decisions.json').read_text())
    threshold = frozen['frequency_screen_threshold_hz']
    truth = {int(run): float(freq) for run, mode, freq in zip(dataset.run_ids, dataset.mode_ids, dataset.f_hf) if mode == 1}
    timing = {}
    for row in plan_rows(output):
        path = directory/'hf'/f'run_{row["run"]:04d}.timing.json'
        if path.exists(): timing[row['run']] = json.loads(path.read_text())
    calibration = json.loads((directory/'calibration.json').read_text())
    calibration_runs = [row['run'] for row in plan_rows(output) if row['group'] == 'calibration']
    inference = json.loads((directory/'blinded_predictions/manifest.json').read_text())['models']
    reports, verification = {}, []
    for group in ('iid', 'challenge'):
        group_decisions = [row for row in frozen_decisions['decisions'] if row['group'] == group]
        policies = sorted({row['policy'] for row in group_decisions})
        group_runs = [row['run'] for row in plan_rows(output) if row['group'] == group]
        for policy in policies:
            rows = [row for row in group_decisions if row['policy'] == policy]
            false_accepts = false_rejects = verified_saved = unverified_automatic = accepts = verified_accepts = 0
            for row in rows:
                frequency = truth.get(row['run'])
                correct = None if frequency is None or row['decision'] == 'HF' else ((frequency >= threshold) == (row['decision'] == 'accept'))
                false_accept = frequency is not None and row['decision'] == 'accept' and frequency < threshold
                false_reject = frequency is not None and row['decision'] == 'reject' and frequency >= threshold
                false_accepts += int(false_accept); false_rejects += int(false_reject)
                accepts += int(row['decision'] == 'accept')
                verified_accepts += int(row['decision'] == 'accept' and frequency is not None)
                verified_saved += int(correct is True)
                unverified_automatic += int(row['decision'] != 'HF' and frequency is None)
                verification.append({**row, 'hf_frequency_hz': frequency, 'correct_automatic_decision': correct,
                                     'false_accept': bool(false_accept), 'false_reject': bool(false_reject)})
            calls = sum(row['decision'] == 'HF' for row in rows)
            baseline_seconds = sum(timing[run]['worker_total_seconds'] for run in group_runs if run in timing)
            referral_seconds = sum(timing[row['run']]['worker_total_seconds'] for row in rows
                                   if row['decision'] == 'HF' and row['run'] in timing)
            calibration_state = (calibration['lf_only_screening'] if policy == 'lf_only'
                                 else calibration['models'][policy]['screening_mode1'])
            used_calibration = calibration_state['runs']
            trained = [] if policy == 'lf_only' else frozen['models'][policy]['training_geometries']
            reports[f'{group}/{policy}'] = {'cases': len(rows), 'HF_referrals': calls,
                'referral_fraction': calls/len(rows), 'false_acceptances': false_accepts,
                'false_rejections': false_rejects, 'automatic_acceptances': accepts,
                'verified_automatic_acceptances': verified_accepts,
                'verified_correct_saved_HF_calls': verified_saved,
                'unverified_automatic_decisions': unverified_automatic,
                'false_acceptance_rate_95_CP': coverage_interval(false_accepts, verified_accepts),
                'observed_operational_benefit_criterion': calls < len(rows) and false_accepts == 0 and unverified_automatic == 0,
                'HF_only_measured_reference_seconds': baseline_seconds,
                'counterfactual_referral_HF_seconds': referral_seconds,
                'timed_reference_cases': sum(run in timing for run in group_runs),
                'policy_setup_cost': {
                    'training_HF_geometries': len(trained),
                    'calibration_HF_geometries_used_for_screening': len(used_calibration),
                    'calibration_HF_geometries_acquired_in_campaign': len(calibration_runs),
                    'used_calibration_reference_seconds': sum(timing[run]['worker_total_seconds']
                                                              for run in used_calibration if run in timing),
                    'model_fit_seconds': 0. if policy == 'lf_only' else frozen['models'][policy]['fit_seconds'],
                    'all80_geometry_full_field_inference_seconds': 0. if policy == 'lf_only' else inference[policy]['prediction_seconds'],
                    'historical_training_solve_seconds': None,
                    'historical_training_cost_note': 'HF labels are counted; original per-run solve timings are not available in this matched campaign.'},
                'claim': 'HF-verified counterfactual operational saving; this validation campaign actually paid for every HF reference.'}
    result = {'screening_task': frozen_decisions['task'], 'threshold_hz': threshold,
        'results': reports, 'decision_verification': verification,
        'shared_study_setup_cost': {'original_training_HF_geometries_available': 48,
            'independent_calibration_HF_geometries_acquired': len(calibration_runs),
            'calibration_measured_reference_seconds': sum(timing[run]['worker_total_seconds'] for run in calibration_runs if run in timing),
            'frozen_model_fit_seconds': sum(model['fit_seconds'] for model in frozen['models'].values()),
            'LF_worker_log': 'prospective/lf_worker.log',
            'blinded_all_model_inference': inference},
        'limits': ['Zero observed false acceptances does not establish zero physical risk.',
            'Threshold is a predeclared benchmark, not an external design requirement.',
            'Stable label1 is not a guarantee of the global lowest structural frequency or complete spectrum.',
            'Discretization is nominal hauto4; uncertain individual-field references remain disclosed.',
            'No extrapolative/OOD coverage guarantee; challenge results are diagnostics.',
            'Upfront training/calibration and full validation costs are not erased by counterfactual saved HF calls.']}
    write_json(directory/'engineering_results.json', result)
    return result


def evaluate(output):
    import numpy as np
    from p1_improve import error_rows, summarize_rows
    from p1_improvement_study import uncertainty_metrics
    from p1_surrogate import write_records
    frozen = check_freeze(output)
    directory = output/'prospective'
    calibration = json.loads((directory/'calibration.json').read_text())
    dataset = paired_targets(output, 'evaluation')
    plan = {row['run']: row for row in plan_rows(output)}
    results, records = {}, []
    for group in ('iid', 'challenge'):
        rows = np.flatnonzero([plan[int(run)]['group'] == group for run in dataset.run_ids])
        if not len(rows):
            results[group] = {'status': 'No accepted HF evaluation labels; no accuracy claim'}
            continue
        results[group] = {}
        for name in frozen['models']:
            all_prediction = saved_prediction(output, name, dataset)
            base = {key: value[rows] for key, value in all_prediction.items()}
            prediction = scaled_prediction(output, name, dataset, rows, base, frozen)
            current = error_rows(dataset, rows, prediction, group=group, model=name)
            records.extend(current)
            model_result = {'point_metrics': summarize_rows(current), 'uncertainty': {}}
            state = calibration['models'][name]['run_simultaneous']
            for coverage in (90, 95):
                if str(coverage) in state['multipliers']:
                    model_result['uncertainty'][str(coverage)] = {
                        'marginal': uncertainty_metrics(dataset, rows, prediction, state, coverage, joint=False),
                        'joint': uncertainty_metrics(dataset, rows, prediction, state, coverage, joint=True)}
                else:
                    model_result['uncertainty'][str(coverage)] = {'unsupported': state['unsupported'][str(coverage)]}
            results[group][name] = model_result
        lf_prediction = {'correction': np.zeros_like(dataset.correction[rows]), 'frequency': dataset.f_lf[rows]}
        current = error_rows(dataset, rows, lf_prediction, group=group, model='uncorrected_lf')
        records.extend(current)
        results[group]['uncorrected_lf'] = {'point_metrics': summarize_rows(current)}
    diagnostics = {}
    for recipe in ('raw', 'local_residual'):
        name = frozen['selected_mean']
        base = saved_prediction(output, name, dataset)
        rows = np.flatnonzero([plan[int(run)]['group'] == 'iid' for run in dataset.run_ids])
        prediction = scaled_prediction(output, name, dataset, rows,
            {key: value[rows] for key, value in base.items()}, frozen, recipe)
        state = calibration['selected_mean_recipe_diagnostics'][recipe]
        diagnostics[recipe] = {str(coverage): uncertainty_metrics(dataset, rows, prediction, state, coverage, joint=True)
                              for coverage in (90, 95) if str(coverage) in state['multipliers']}
    write_records(directory/'evaluation_mode_rows.csv', records)
    report = {'evaluated_at_utc': utc(), 'groups': results,
        'selected_mean_uncertainty_diagnostics': diagnostics,
        'no_evaluation_label_tuning': True, 'model_freeze_sha256': file_sha256(output/'model_freeze.json'),
        'accuracy_scope': 'New independent physical-uniform IID test and separately drawn region-exclusion challenge; stable accepted mode identities only.'}
    write_json(directory/'evaluation_results.json', report)
    engineering_results(output, dataset, frozen)
    plot_prospective(output, report)
    print(json.dumps({'prospective_evaluation_finished': True,
        'iid_selected_mean': results['iid'][frozen['selected_mean']]['point_metrics'],
        'challenge_region_model': results['challenge']['region_selected']['point_metrics']}), flush=True)
    return report


def plot_prospective(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen = json.loads((output/'model_freeze.json').read_text())
    directory = output/'prospective'
    name = frozen['selected_mean']
    for quantity in ('field', 'frequency'):
        fig, ax = plt.subplots(figsize=(8, 5))
        for recipe, entry in report['selected_mean_uncertainty_diagnostics'].items():
            if '95' not in entry: continue
            curve = entry['95']['referral_curves'][quantity]
            ax.plot([row['retained_fraction'] for row in curve], [row['mean_retained_error'] for row in curve],
                    marker='o', label=recipe)
            ax.axhline(curve[-1]['random_referral_expected_mean'], linestyle=':', alpha=.4)
        ax.set(xlabel='Retained geometry fraction (remaining geometries referred to HF)',
               ylabel='Mean retained error', title=f'Prospective IID uncertainty discrimination: {quantity}; frozen mean={name}')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(directory/f'{quantity}_referral_curve.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'lf', 'blind', 'hf', 'calibrate', 'evaluate', 'native-smoke'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=('calibration', 'evaluation'), default='calibration')
    parser.add_argument('--run', type=int, default=9828)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--destination', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        if args.job is None or args.destination is None:
            parser.error('Native worker requires job and destination')
        return hf_worker(args.job, args.destination, args.cores)
    preserve_stage_source(output, f'prospective-{args.stage}' if args.stage != 'hf' else f'prospective-hf-{args.group}')
    if args.stage == 'plan': generate_plan(output)
    elif args.stage == 'lf': return lf_campaign(output)
    elif args.stage == 'blind': blind_predictions(output)
    elif args.stage == 'hf': return hf_campaign(output, args.group)
    elif args.stage == 'calibrate': calibrate(output)
    elif args.stage == 'evaluate': evaluate(output)
    else: return native_smoke(output, args.run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
