#!/usr/bin/env python3
"""Staged P1 improvement; original audited artifacts are never overwritten."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
import zipfile
ROOT = Path(os.environ.get('P1_ORIGINAL_ROOT', Path(__file__).resolve().parent)).resolve()
sys.path.insert(1, str(ROOT))


from p1_design import file_sha256, write_json

DATA = ROOT / 'p1_data_five'
SEED = 20261005
WORKFLOW_SOURCES = ('p1_improve.py', 'p1_improved_models.py',
                    'p1_reference_diagnosis.py', 'p1_improvement_study.py',
                    'p1_prospective.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'audited_baseline.zip'
    manifest = output / 'baseline_manifest.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError('Baseline snapshot already exists; refusing replacement')
    # Include the actual raw data, native models, saved predictions and models,
    # not only hashes or a promise that a caller can find the original files.
    sources = [p for p in ROOT.glob('*.py') if (
        p.name.startswith(('p1_', 'test_p1_', 'run_p1_', 'make_p1_'))
        or p.name in {'compute_corrections.py', 'run_hf_batches.py',
                     'run_lf_batches.py', 'hc_HighFidelity_LHS.py',
                     'fsdt_mode_shapes.py', 'lhs_sampling.py', 'honeycomb.py',
                     'plot_p1_results.py', 'test_regressions.py'})
        and p.name not in {*WORKFLOW_SOURCES, 'test_p1_improvement.py'}]
    sources += [p for p in (ROOT / 'plate').rglob('*.py')]
    sources += [ROOT / name for name in ('P1_DATA_README.md', 'P1_RESULTS_REVIEW.md',
                                       'proposals_updated(2).md')]
    sources += [p for p in (ROOT / 'simulations/scripts').glob('hc_mesh_convergence.py')]
    files = sorted(set(sources + [p for p in DATA.rglob('*') if p.is_file()]),
                   key=lambda p: p.relative_to(ROOT).as_posix())
    total = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(output).free < total + 8 * 2**30:
        raise OSError('Baseline snapshot must leave at least 8GiB working space')
    partial = output / 'audited_baseline.partial.zip'
    if partial.exists():
        raise FileExistsError(f'Incomplete snapshot exists: {partial}')
    records = []
    started = time.monotonic()
    print(f'BASELINE snapshot started: {len(files)} files, {total/2**30:.2f}GiB', flush=True)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for index, path in enumerate(files, 1):
            before = path.stat()
            relative = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256()
            with path.open('rb') as source, bundle.open(relative, 'w', force_zip64=True) as target:
                while block := source.read(2**20):
                    digest.update(block)
                    target.write(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Baseline changed while being preserved: {path}')
            records.append({'path': relative, 'bytes': before.st_size,
                            'sha256': digest.hexdigest()})
            if index % 250 == 0 or path.suffix == '.mph':
                print(f'BASELINE {index}/{len(files)}; elapsed={time.monotonic()-started:.1f}s; {relative}', flush=True)
    partial.rename(archive)
    os.chmod(archive, stat.S_IREAD)
    write_json(manifest, {'recorded_at_utc': utc(), 'root': str(ROOT),
                         'archive': archive.name, 'archive_bytes': archive.stat().st_size,
                         'archive_sha256': file_sha256(archive), 'files': records,
                         'policy': 'Read-only byte snapshot; original source/data/model artifacts remain untouched.'})
    protocol = {
        'version': 1, 'recorded_at_utc': utc(), 'seed': SEED,
        'baseline_manifest_sha256': file_sha256(manifest),
        'physical_scope': 'Five variables; fixed Al/CCCC; original FSDT order15 and COMSOL shell builder; no aerodynamic or damage extension.',
        'isolation': {'model_selection': 'Original 48 train geometries only; geometry-grouped CV; every scaler/decoder fitted within its training fold.',
                      'historical_calibration_test': 'Diagnostics and historical benchmarks only; not untouched evidence or model-selection labels.',
                      'prospective': 'Freeze fitted models/configuration/metrics before new evaluation geometries and HF labels; independent calibration; no final-label tuning.'},
        'reference': {'hypothesis': 'Close-mode numerical basis mixing explains individual-shape drift better than a change in the modal subspace.',
                      'runs': {'2209': {'accepted_mode': 7}, '3408': {'accepted_mode': 10}},
                      'neighbor_selection': 'Nearest campaign-reference frequency; fixed before projecting any finer fields.',
                      'diagnostics': ['Individual MAC/RMS/frequency gaps', 'Top-transverse span', 'Common-shell mass-weighted full displacement/director span if native variables are available', 'Quadrature sensitivity'],
                      'acceptance': 'Retain individual modes regardless of outcome. Accept subspace interpretation only with stable quadrature and observed span stability; never call a sampled-field span full-DOF convergence.',
                      'quadrature_levels': [{'face_order': 40, 'wall_order': 2}, {'face_order': 80, 'wall_order': 4}],
                      'quadrature_angle_target_degrees': 0.05,
                      'mesh_targets': {'frequency_pct': 0.1, 'field_rms': 0.00075},
                      'additional_solve_policy': 'Choose a specific numerical experiment only after diagnosis; no unchanged automatic retry or blind largest-mesh escalation.'},
        'decomposition': {'hypothesis': 'Regression dominates typical field errors; difficult tails can additionally contain representation and reference sensitivity.',
                          'identity': 'Total error = regression-in-field-space + oracle representation error; report the MSE cross term, not a false additive variance attribution.',
                          'reference_limit': 'Finer-reference drift is observed sensitivity, not an exact error bar or converged truth.'},
        'selection': {'folds': 4, 'seed': SEED, 'primary': 'Geometry-level median field RMS',
                      'secondary': ['Geometry-level frequency relative error', 'Field P95 and maximum', 'Decoder oracle error'],
                      'promotion': 'At least5% primary improvement over original PCA48/RBF with no more than5% degradation in frequency median or field P95 on identical train-only folds; retain negative results.',
                      'candidate_policy': 'Small, finite targeted GP/representation changes chosen from training-only decomposition; no automatic nonlinear-decoder upgrade.',
                      'gp_max_iterations': 80, 'neural_steps': 1500,
                      'neural_scope': 'Required by proposal: genuine coordinate-conditioned MLP with design, mode and LF conditioning, structural boundary factor; Fourier ridge remains labeled as a proxy.'},
        'learning_curves': {'hf_train_budgets': [8, 16, 24, 32, 48], 'repeats': 3,
                            'families': ['Direct-HF', 'Original PCA-GP correction', 'Co-kriging', 'Genuine neural INR'],
                            'policy': 'Same nested geometry subsets and historical evaluation cases for all families; fit representation on each subset; report tuning and runtime, and do not select models from historical-test curves.'},
        'uncertainty': {'hypothesis': 'Training-only error-informed uncertainty can provide sharper useful referral decisions than raw GP scales.',
                        'metrics': ['Marginal/simultaneous/joint coverage', 'Width', 'Proper interval score', 'Geometry error discrimination', 'Risk versus referral/retained fraction'],
                        'selection': 'Train-only validation; any residual-based scale fitted without the validation geometry; final calibration changes widths only.',
                        'coverage_levels': [90, 95],
                        'caution': 'Finite conformal ranks do not establish sharpness, stable estimated coverage or conditional/shift coverage.'},
        'prospective': {'calibration_count': 32, 'iid_test_count': 32,
                        'excluded_region_challenge_count': 16,
                        'distribution': 'Independent uniform draws within the five original physical bounds; IID calibration/test use the same distribution.',
                        'region': 'theta_c>=60 degrees; separate exclusion model fitted/tuned only below60; calibrate its in-region-of-training scale below60 and report challenge coverage as shift diagnostics, not a guarantee.',
                        'geometry_generation': 'Only after model freeze; deterministic independent RNG streams, new run IDs above10000, no duplicate old designs.'},
        'engineering': {'task': 'Screen a predeclared first-mode frequency threshold; confident accept/reject or HF referral.',
                        'threshold': 'Median first-mode HF frequency among original train geometries, fixed before prospective labels; a benchmark threshold, not an external service safety requirement.',
                        'metrics': ['False acceptances', 'False rejections', 'HF referral fraction', 'Verified saved HF calls', 'Training/LF/HF/inference cost'],
                        'acceptance': 'Positive benefit only with fewer HF calls and zero observed false acceptances; quantify finite-sample uncertainty; ambiguous/unverified identities cannot count as successes.',
                        'claim_limit': 'Decisions verified against the declared COMSOL reference; no universal physical safety guarantee.'},
        'resources': {'native_cores': 4, 'java_heap_gib': 8, 'scratch_root': 'D:/',
                      'native_worker_deadline_minutes': 360, 'heartbeat_seconds': 60,
                      'concurrent_native_jobs': 1, 'timeout_retries': 0,
                      'prospective_save_full_model': False,
                      'policy': 'Durable stage/native progress; retain failures; all owned worker trees stopped on parent loss or deadline.'},
        'delivery': 'Reproducible matched source/raw data/models/predictions/protocols/figures; claim-by-claim outcomes including failures; no upload without user authorization.'
    }
    write_json(output / 'protocol.json', protocol)
    print(json.dumps({'baseline_snapshot_complete': True, 'files': len(records),
                      'bytes': total, 'elapsed_seconds': time.monotonic()-started,
                      'manifest': str(manifest), 'protocol': str(output/'protocol.json')}, indent=2), flush=True)


def training_data():
    from p1_holdout import read_roles
    from p1_surrogate import P1Dataset
    data = P1Dataset.load(DATA/'corrections_five')
    roles = read_roles(DATA/'hf_plan/hf_plan.csv')
    return data.select_runs([run for run, role in roles.items() if role == 'train']), data, roles


def error_rows(dataset, rows, prediction, **labels):
    import numpy as np
    rows = np.asarray(rows, dtype=int)
    error = prediction['correction']-dataset.correction[rows]
    rms = np.sqrt(np.mean(error[:, dataset.interior_mask]**2, axis=1))
    frequency = 100*np.abs(prediction['frequency']-dataset.f_hf[rows])/dataset.f_hf[rows]
    return [{'run': int(dataset.run_ids[row]), 'mode': int(dataset.mode_ids[row]),
             'field_rms': float(rms[index]), 'frequency_error_pct': float(frequency[index]), **labels}
            for index, row in enumerate(rows)]


def summarize_rows(records):
    import numpy as np
    from p1_recheck import distribution
    runs = sorted({int(row['run']) for row in records})
    geometry_field = [np.median([row['field_rms'] for row in records if int(row['run']) == run]) for run in runs]
    geometry_frequency = [np.median([row['frequency_error_pct'] for row in records if int(row['run']) == run]) for run in runs]
    return {'rows': len(records), 'geometries': len(runs),
            'field_rms': distribution([row['field_rms'] for row in records]),
            'frequency_error_pct': distribution([row['frequency_error_pct'] for row in records]),
            'geometry_median_field_rms': distribution(geometry_field),
            'geometry_median_frequency_error_pct': distribution(geometry_frequency)}


def decompose(output):
    import numpy as np
    from p1_improved_models import OutputMetricDecoder
    from p1_pairing import align_real, assign, load_run
    from p1_recheck import distribution
    from p1_surrogate import LatentGPModel, grouped_run_splits, write_records
    target = output/'decomposition'
    if target.exists():
        raise FileExistsError('Decomposition already started; use its saved results')
    target.mkdir()
    protocol = json.loads((output/'protocol.json').read_text())
    train, complete, roles = training_data()
    splits = grouped_run_splits(train.run_ids, protocol['selection']['folds'], protocol['selection']['seed'])
    records, folds = [], []
    predictions = np.full_like(train.correction, np.nan)
    prediction_std = np.full_like(train.correction, np.nan)
    frequencies, frequency_std = np.full(train.n_rows, np.nan), np.full(train.n_rows, np.nan)
    for fold, (training, validation, training_runs, validation_runs) in enumerate(splits):
        started = time.monotonic()
        model = LatentGPModel(48, 42, protocol['selection']['gp_max_iterations']).fit(train, training)
        prediction = model.predict_rows(train, validation)
        oracle = model.decoder.reconstruction(train.correction[validation])
        physical = OutputMetricDecoder(48, 42).fit(train, training).reconstruction(train.correction[validation])
        represent = oracle-train.correction[validation]
        regression = prediction['correction']-oracle
        total = prediction['correction']-train.correction[validation]
        mask = train.interior_mask
        representation_mse = np.mean(represent[:, mask]**2, axis=1)
        regression_mse = np.mean(regression[:, mask]**2, axis=1)
        cross = 2*np.mean(represent[:, mask]*regression[:, mask], axis=1)
        total_mse = np.mean(total[:, mask]**2, axis=1)
        np.testing.assert_allclose(total_mse, representation_mse+regression_mse+cross, rtol=1e-10, atol=1e-14)
        physical_rms = np.sqrt(np.mean((physical-train.correction[validation])[:, mask]**2, axis=1))
        current = error_rows(train, validation, prediction, fold=fold)
        for index, row in enumerate(current):
            row.update(representation_rms=float(np.sqrt(representation_mse[index])),
                       regression_rms=float(np.sqrt(regression_mse[index])),
                       mse_cross_term=float(cross[index]), physical_metric_oracle_rms=float(physical_rms[index]))
        records.extend(current)
        predictions[validation], prediction_std[validation] = prediction['correction'], prediction['correction_std']
        frequencies[validation], frequency_std[validation] = prediction['frequency'], prediction['frequency_std']
        model.save(target/f'fold{fold}_current_model', train)
        folds.append({'fold': fold, 'training_runs': training_runs.tolist(), 'validation_runs': validation_runs.tolist(),
                      'elapsed_seconds': time.monotonic()-started})
        print(f'DECOMPOSITION fold{fold}: {len(training_runs)}train/{len(validation_runs)}validation geometries; {folds[-1]["elapsed_seconds"]:.1f}s', flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError('A training geometry was not evaluated out of fold')
    np.savez_compressed(target/'current_oof_predictions.npz', run_ids=train.run_ids, mode_ids=train.mode_ids,
                        correction=predictions, correction_std=prediction_std,
                        frequency=frequencies, frequency_std=frequency_std)
    write_records(target/'train_mode_rows.csv', records)
    summary = {'source_sha256': file_sha256(__file__), 'scope': 'Training-only geometry-grouped out-of-fold decomposition',
               'folds': folds, 'total': summarize_rows(records),
               'components': {key: distribution([row[key] for row in records])
                              for key in ('representation_rms', 'regression_rms', 'mse_cross_term', 'physical_metric_oracle_rms')},
               'by_mode': {str(mode): {key: distribution([row[key] for row in records if row['mode'] == mode])
                            for key in ('field_rms', 'representation_rms', 'regression_rms', 'physical_metric_oracle_rms')}
                           for mode in train.modes},
               'by_geometry': {str(run): {key: distribution([row[key] for row in records if row['run'] == run])
                                for key in ('field_rms', 'representation_rms', 'regression_rms')}
                               for run in train.runs},
               'identity': 'MSE(total)=MSE(representation)+MSE(regression)+2*mean(representation*regression); not additive independent error bars.'}
    write_json(target/'training_results.json', summary)
    # Separately diagnose the previously inspected test references. These
    # values do not choose kernels, hyperparameters or promotion outcomes.
    frozen = LatentGPModel.load(DATA/'holdout_corrected/latent_gp_train')
    historical = []
    finest_paths = {2209: DATA/'hf_followup_20261003/mesh1_scratch/hf/run_2209.npz',
                    3408: DATA/'hf_followup_20261003/resume3408_mesh2/hf/run_3408.npz'}
    with (DATA/'corrections_five/pairing.csv').open(newline='', encoding='utf-8') as stream:
        accepted = {(int(row['run']), int(row['mode'])): int(row['hf_mode'])-1
                    for row in csv.DictReader(stream) if row['status'] == 'accepted'}
    for run, path in finest_paths.items():
        nominal, fine = load_run(DATA/f'hf/run_{run:04d}.npz', 'hf'), load_run(path, 'hf')
        mapping, _ = assign(nominal['w'], fine['w'])
        rows = np.flatnonzero(complete.run_ids == run)
        prediction = frozen.predict_rows(complete, rows)
        for index, row in enumerate(rows):
            mode = int(complete.mode_ids[row])
            candidate = int(mapping[accepted[(run, mode)]])
            refined = align_real(fine['w'][candidate], complete.hf[row])-complete.lf[row]
            oracle = frozen.decoder.reconstruction(refined[None])[0]
            reference_delta = refined-complete.correction[row]
            historical.append({'run': run, 'mode': mode,
                'reference_sensitivity_rms': float(np.sqrt(np.mean(reference_delta[complete.interior_mask]**2))),
                'fine_reference_representation_rms': float(np.sqrt(np.mean((oracle-refined)[complete.interior_mask]**2))),
                'fine_reference_regression_rms': float(np.sqrt(np.mean((prediction['correction'][index]-oracle)[complete.interior_mask]**2))),
                'fine_reference_total_rms': float(np.sqrt(np.mean((prediction['correction'][index]-refined)[complete.interior_mask]**2)))})
    write_records(target/'historical_reference_components.csv', historical)
    write_json(target/'historical_reference_results.json', {
        'scope': 'Previously inspected runs2209/3408 only; not tuning or prospectively blinded evaluation.',
        'caution': 'Neither finest saved reference certifies individual-shape convergence; drift is sensitivity, not an exact error bar.',
        'by_run': {str(run): {key: distribution([row[key] for row in historical if row['run'] == run])
                    for key in historical[0] if key not in ('run', 'mode')} for run in finest_paths}})
    print(json.dumps({'training_only': summary['total'], 'components': summary['components']}, indent=2), flush=True)


def candidate_configurations(output):
    decomposition = json.loads((output/'decomposition/training_results.json').read_text())
    protocol = json.loads((output/'protocol.json').read_text())
    common = {'seed': 42, 'max_iterations': protocol['selection']['gp_max_iterations']}
    candidates = [{'name': 'current_pca48_rbf', 'family': 'current_pca', 'latent_dim': 48, **common}]
    # This is a single finite targeted improvement comparison, not an open
    # architecture search. The actual field-metric oracle diagnostic supplies
    # the reason for changing the representation's training inner product.
    for family in ('output_pca', 'direct_hf', 'cokriging'):
        for kernel in ('rbf', 'matern32'):
            candidates.append({'name': f'{family}_{kernel}', 'family': family, 'kernel': kernel, 'latent_dim': 48, **common})
    for rate in (1e-3, 3e-4):
        candidates.append({'name': f'neural_inr_lr{rate:g}', 'family': 'neural_inr', 'learning_rate': rate,
                           'steps': protocol['selection']['neural_steps'], 'width': 64, **common})
    payload = {'registered_at_utc': utc(), 'candidates': candidates,
               'diagnosis_sha256': file_sha256(output/'decomposition/training_results.json'),
               'diagnosis': decomposition['components'],
               'rationale': 'Align PCA with reported physical RMS; compare RBF smoothness with a single Matern3/2 alternative where regression dominates. Two bounded neural settings satisfy the mandatory true-MLP baseline, not a neural-decoder promotion.',
               'selection_labels': 'Original train geometries only; no historical-calibration/test model selection.'}
    write_json(output/'candidate_protocol.json', payload)
    return candidates


def select_models(output, excluded_region=False):
    import numpy as np
    from p1_improved_models import make_model, save_model
    from p1_surrogate import grouped_run_splits, write_records
    protocol = json.loads((output/'protocol.json').read_text())
    path = output/('region_selection' if excluded_region else 'selection')
    if path.exists():
        raise FileExistsError(f'Selection already started: {path}')
    path.mkdir()
    configs = (json.loads((output/'candidate_protocol.json').read_text())['candidates']
               if (output/'candidate_protocol.json').exists() else candidate_configurations(output))
    train, _, _ = training_data()
    if excluded_region:
        theta = train.parameter_names.index('theta_c')
        keep = [run for run in train.runs if train.parameters[np.flatnonzero(train.run_ids == run)[0], theta] < 60.]
        train = train.select_runs(keep)
    splits = grouped_run_splits(train.run_ids, protocol['selection']['folds'], protocol['selection']['seed'])
    all_records, reports = [], {}
    for config in configs:
        name = config['name']
        directory = path/name
        directory.mkdir()
        if name == 'current_pca48_rbf' and not excluded_region:
            source = output/'decomposition'
            with np.load(source/'current_oof_predictions.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['run_ids'], train.run_ids)
                np.testing.assert_array_equal(saved['mode_ids'], train.mode_ids)
            with (source/'train_mode_rows.csv').open(newline='', encoding='utf-8') as stream:
                records = [{'candidate': name, 'run': int(row['run']), 'mode': int(row['mode']),
                            'fold': int(row['fold']), 'field_rms': float(row['field_rms']),
                            'frequency_error_pct': float(row['frequency_error_pct'])}
                           for row in csv.DictReader(stream)]
            summary = json.loads((source/'training_results.json').read_text())
            for fold, (_, _, training_runs, validation_runs) in enumerate(splits):
                np.testing.assert_array_equal(summary['folds'][fold]['training_runs'], training_runs)
                np.testing.assert_array_equal(summary['folds'][fold]['validation_runs'], validation_runs)
                destination = directory/f'fold{fold}'
                shutil.copytree(source/f'fold{fold}_current_model', destination)
                write_json(destination/'configuration.json', config)
            shutil.copyfile(source/'current_oof_predictions.npz', directory/'oof.npz')
            reports[name] = {'configuration': config, 'metrics': summarize_rows(records),
                             'timings': summary['folds'],
                             'timing_scope': 'Reused decomposition fits; timings also include oracle diagnostics.'}
            all_records.extend(records)
            write_json(directory/'results.json', reports[name])
            print('SELECTION current_pca48_rbf: reusing completed identical training-only OOF predictions', flush=True)
            continue
        field, field_std = np.full_like(train.correction, np.nan), np.full_like(train.correction, np.nan)
        freq, freq_std = np.full(train.n_rows, np.nan), np.full(train.n_rows, np.nan)
        records, timings = [], []
        for fold, (training, validation, training_runs, validation_runs) in enumerate(splits):
            started = time.monotonic()
            model = make_model(config).fit(train, training)
            prediction = model.predict_rows(train, validation)
            field[validation], field_std[validation] = prediction['correction'], prediction['correction_std']
            freq[validation], freq_std[validation] = prediction['frequency'], prediction['frequency_std']
            current = error_rows(train, validation, prediction, candidate=name, fold=fold)
            records.extend(current)
            save_model(model, directory/f'fold{fold}', train, config)
            timings.append({'fold': fold, 'seconds': time.monotonic()-started,
                            'training_runs': training_runs.tolist(), 'validation_runs': validation_runs.tolist()})
            print(f'SELECTION {name} fold{fold}: {timings[-1]["seconds"]:.1f}s', flush=True)
        if not np.isfinite(field).all() or not np.isfinite(freq).all():
            raise ValueError(f'Incomplete OOF prediction: {name}')
        np.savez_compressed(directory/'oof.npz', run_ids=train.run_ids, mode_ids=train.mode_ids,
                            correction=field, correction_std=field_std, frequency=freq, frequency_std=freq_std)
        reports[name] = {'configuration': config, 'metrics': summarize_rows(records), 'timings': timings}
        all_records.extend(records)
        write_json(directory/'results.json', reports[name])
    baseline = reports['current_pca48_rbf']['metrics']
    admissible = []
    for name, report in reports.items():
        metrics = report['metrics']
        primary = metrics['geometry_median_field_rms']['median']
        promoted = (primary <= .95*baseline['geometry_median_field_rms']['median']
                    and metrics['geometry_median_frequency_error_pct']['median'] <= 1.05*baseline['geometry_median_frequency_error_pct']['median']
                    and metrics['field_rms']['p95'] <= 1.05*baseline['field_rms']['p95'])
        report['passes_predeclared_promotion'] = bool(promoted)
        if promoted:
            admissible.append((primary, name))
    selected = min(admissible)[1] if admissible else 'current_pca48_rbf'
    family_winners = {}
    for family in ('current_pca', 'output_pca', 'direct_hf', 'cokriging', 'neural_inr'):
        members = [(r['metrics']['geometry_median_field_rms']['median'], name)
                   for name, r in reports.items() if r['configuration']['family'] == family]
        family_winners[family] = min(members)[1]
    result = {'selected': selected, 'promoted': selected != 'current_pca48_rbf',
              'family_winners': family_winners, 'candidates': reports,
              'training_geometries': train.runs.tolist(), 'scope': 'Train-only grouped CV; guidance for prospectively frozen evaluation, not unbiased post-selection final evidence.',
              'excluded_region': 'theta_c>=60 absent from fitting AND model-selection labels' if excluded_region else None,
              'source_sha256': file_sha256(__file__), 'model_source_sha256': file_sha256(ROOT/'p1_improved_models.py')}
    write_records(path/'mode_rows.csv', all_records)
    write_json(path/'results.json', result)
    print(json.dumps({'selected': selected, 'promoted': result['promoted'], 'family_winners': family_winners}, indent=2), flush=True)


def preserve_stage_source(output, stage):
    directory = output/'source_versions'
    directory.mkdir(exist_ok=True)
    records = {}
    for name in WORKFLOW_SOURCES:
        source = Path(__file__).with_name(name)
        if not source.exists():
            continue
        digest = file_sha256(source)
        snapshot = directory/f'{source.stem}_{digest}.py'
        if not snapshot.exists():
            shutil.copyfile(source, snapshot)
        records[name] = {'sha256': digest, 'snapshot': snapshot.relative_to(output).as_posix()}
    write_json(output/f'{stage}_started.json', {'recorded_at_utc': utc(), 'sources': records})


def prepare_execution_source(output, stage):
    """Run from immutable canonical filenames while later workflow code evolves."""
    if not stage or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for character in stage):
        raise ValueError('Execution stage must contain only lowercase letters, digits, underscores and hyphens')
    directory = output/'execution_sources'/stage
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in WORKFLOW_SOURCES:
        source = Path(__file__).with_name(name)
        if not source.exists():
            continue
        shutil.copyfile(source, directory/name)
        hashes[name] = file_sha256(directory/name)
    write_json(directory/'execution.json', {'original_root': str(ROOT), 'sources': hashes,
               'environment': {'P1_ORIGINAL_ROOT': str(ROOT)}, 'prepared_at_utc': utc()})
    print(json.dumps({'execution_source': str(directory), 'sources': hashes}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'decompose', 'select', 'select-region', 'prepare-source'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execution-stage')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == 'prepare-source':
        if args.execution_stage is None:
            parser.error('prepare-source requires --execution-stage')
        prepare_execution_source(output, args.execution_stage)
        return 0
    if args.stage != 'freeze':
        preserve_stage_source(output, args.stage)
    if args.stage == 'freeze':
        freeze(output)
    elif args.stage == 'decompose':
        decompose(output)
    else:
        select_models(output, excluded_region=args.stage == 'select-region')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
