"""Report, replay and package the completed P1 improvement study without refitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

from p1_improve import ROOT, SEED, utc
from p1_design import file_sha256, write_json


SOURCE_FILES = (
    'p1_improve.py', 'p1_improved_models.py', 'p1_reference_diagnosis.py',
    'p1_improvement_study.py', 'p1_prospective.py', 'p1_delivery.py',
    'test_p1_improvement.py', 'P1_DATA_README.md', 'P1_RESULTS_REVIEW.md',
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require_completed(output):
    for group, count in (('calibration', 32), ('evaluation', 48)):
        execution = read_json(output/'prospective'/f'hf_{group}_execution.json')
        if execution['completed'] != count or execution['requested'] != count:
            raise ValueError(f'{group} acquisition is incomplete; do not package a partial study')
    for name in ('evaluation_results.json', 'engineering_results.json'):
        if not (output/'prospective'/name).is_file():
            raise ValueError(f'Missing actual prospective result: {name}')


def paired_comparisons(output):
    """Geometry-paired descriptive bootstrap; negative means candidate is worse."""
    import numpy as np
    with (output/'prospective/evaluation_mode_rows.csv').open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    selected = read_json(output/'model_freeze.json')['selected_mean']
    results = {}
    for group in ('iid', 'challenge'):
        models = sorted({r['model'] for r in records if r['group'] == group})
        for baseline in ('uncorrected_lf', selected):
            for candidate in models:
                if candidate == baseline:
                    continue
                for quantity in ('field_rms', 'frequency_error_pct'):
                    by_model = {}
                    for model in (baseline, candidate):
                        runs = sorted({int(r['run']) for r in records if r['group'] == group and r['model'] == model})
                        by_model[model] = {run: float(np.median([float(r[quantity]) for r in records
                            if r['group'] == group and r['model'] == model and int(r['run']) == run])) for run in runs}
                    if by_model[baseline].keys() != by_model[candidate].keys():
                        raise ValueError('Comparison models do not share identical accepted geometries')
                    runs = sorted(by_model[baseline])
                    if not runs:
                        continue
                    difference = np.array([by_model[baseline][run]-by_model[candidate][run] for run in runs])
                    draw = np.random.default_rng(SEED).integers(0, len(runs), (10000, len(runs)))
                    results[f'{group}/{baseline}-minus-{candidate}/{quantity}'] = {
                        'geometries': len(runs), 'median_paired_difference': float(np.median(difference)),
                        'bootstrap_95_percentile_interval': np.quantile(np.median(difference[draw], axis=1), [.025, .975]).tolist(),
                        'positive_direction': 'Candidate has lower error',
                        'scope': 'Descriptive geometry bootstrap, not simultaneous multiple-comparison inference or a model-selection criterion.'}
    return results


def lf_reported_cost(output):
    with (output/'prospective/plan.csv').open(newline='', encoding='utf-8') as stream:
        groups = {int(row['run_id']): row['group'] for row in csv.DictReader(stream)}
    logged = {}
    for line in (output/'prospective/lf_worker.log').read_text().splitlines():
        match = re.fullmatch(r'Run (\d+): .* \(([\d.]+)s\)', line)
        if match:
            run = int(match[1])
            if run in logged:
                raise ValueError('Duplicate prospective LF timing record')
            logged[run] = float(match[2])
    return {
        'groups': {group: {'timed_runs': sum(run in logged for run, assigned in groups.items() if assigned == group),
                          'sum_logged_seconds': sum(logged[run] for run, assigned in groups.items()
                                                    if assigned == group and run in logged),
                          'maximum_sum_rounding_error_seconds': .05*sum(run in logged for run, assigned in groups.items() if assigned == group)}
                   for group in ('calibration', 'iid', 'challenge')},
        'missing_runs': sorted(set(groups)-logged.keys()),
        'meaning': 'Original LF CLI per-run solve/save elapsed time printed to0.1s; sums retain that rounding, and omit process startup. No LF solves are repeated.'}


def make_report(output):
    import numpy as np
    require_completed(output)
    model_replay = read_json(output/'delivery_model_replay.json')
    frozen = read_json(output/'model_freeze.json')
    generation = read_json(output/'prospective/generation.json')
    blind = read_json(output/'prospective/blinded_predictions/manifest.json')
    calibration = read_json(output/'prospective/calibration.json')
    evaluation = read_json(output/'prospective/evaluation_results.json')
    engineering = read_json(output/'prospective/engineering_results.json')
    selection = read_json(output/'selection/results.json')
    regional = read_json(output/'region_selection/results.json')
    decomposition = read_json(output/'decomposition/training_results.json')
    curves = read_json(output/'learning_curves/results.json')
    uq = {name: read_json(output/directory/'results.json') for name, directory in
          (('global', 'uncertainty_selection'), ('regional', 'region_uncertainty_selection'))}
    mass = read_json(output/'reference_mass_diagnosis_all_pairs.json')
    assert frozen['frozen_at_utc'] < generation['generated_at_utc'] < blind['predicted_at_utc']
    cal_execution = read_json(output/'prospective/hf_calibration_execution.json')
    test_execution = read_json(output/'prospective/hf_evaluation_execution.json')
    decisions = read_json(output/'prospective/screening_decisions.json')
    assert blind['predicted_at_utc'] < min(r['started_at_utc'] for r in cal_execution['outcomes'])
    assert decisions['recorded_at_utc'] < min(r['started_at_utc'] for r in test_execution['outcomes'])
    aggregates = {}
    for family in sorted({r['family'] for r in curves['trials']}):
        for budget in (8, 16, 24, 32, 48):
            trials = [r for r in curves['trials'] if r['family'] == family and r['hf_budget'] == budget]
            successful = [r for r in trials if r['status'] == 'complete']
            entry = {'complete': len(successful), 'attempted': len(trials)}
            for quantity in ('geometry_median_field_rms', 'geometry_median_frequency_error_pct'):
                values = [r['metrics'][quantity]['median'] for r in successful]
                if values:
                    entry[quantity] = {'median_over_repeats': float(np.median(values)), 'repeat_min': min(values), 'repeat_max': max(values)}
            aggregates[f'{family}/HF{budget}'] = entry
    selected_uq = evaluation['groups']['iid'][frozen['selected_mean']]['uncertainty']
    selected_engineering = engineering['results'][f'iid/{frozen["selected_mean"]}']
    claims = [
        {'stage': 1, 'claim': 'Numerical basis mixing contributes to the individual-field sensitivity',
         'status': 'Supported for the measured two-mode shell-state spans',
         'evidence': 'reference_mass_diagnosis_all_pairs.json and reference_top_diagnosis.json',
         'limit': 'Sampled full shell displacement/director state with physical mass quadrature, not assembled full-DOF convergence; individual 2209/3408 field tolerance failures remain.'},
        {'stage': 2, 'claim': 'A targeted global model improves the predeclared training-only gate',
         'status': 'Not supported; retain original PCA48/RBF', 'selected': selection['selected'],
         'promoted': selection['promoted'], 'regional_selected': regional['selected'],
         'regional_promoted': regional['promoted'],
         'limit': 'A physical-output PCA oracle can improve representation without improving full predictions. Regional training-CV promotion is not an OOD guarantee.'},
        {'stage': 3, 'claim': 'Matched HF-budget comparisons include a genuine neural INR',
         'status': 'Completed with actual fitted weights and predictions',
         'trials': len(curves['trials']), 'completed': sum(r['status'] == 'complete' for r in curves['trials']),
         'limit': 'Historical evaluation cases; each two-configuration family receives the same tuning splits, not identical optimizer dynamics. Bands are three-repeat extrema, not confidence intervals; do not infer a universal sample-efficiency winner.'},
        {'stage': 4, 'claim': 'Local residual scaling improves useful uncertainty',
         'status': 'Rejected by both predeclared training-only promotion gates',
         'selected_recipes': {name: result['selected_recipe'] for name, result in uq.items()},
         'prospective_selected_mean': selected_uq,
         'limit': 'Finite conformal ranks and observed coverage do not imply sharpness, conditional coverage, epistemic meaning, or OOD validity. Calibration is over accepted-mode populations.'},
        {'stage': 5, 'claim': 'Models were frozen before independent prospective geometry and label generation',
         'status': 'Authenticated ordering', 'frozen_at_utc': frozen['frozen_at_utc'],
         'generated_at_utc': generation['generated_at_utc'], 'blind_predictions_at_utc': blind['predicted_at_utc'],
         'screening_decisions_at_utc': decisions['recorded_at_utc'], 'groups': generation['groups'],
         'limit': 'Independent draws within the same original physical bounds; challenge theta>=60 is excluded only for the regional model, not a new physical family.'},
        {'stage': 6, 'claim': 'Frequency screening provides HF-verified operational benefit',
         'status': 'Observed criterion met' if selected_engineering['observed_operational_benefit_criterion'] else 'Observed criterion not met',
         'selected_IID_policy': selected_engineering,
         'limit': 'Stable reference label1 and a benchmark threshold, not a universal fundamental-frequency or service certificate. Full validation paid all HF costs; savings are counterfactual operation, with training/calibration cost disclosed.'},
        {'stage': 7, 'claim': 'A matched local source/data/model release is reproducible',
         'status': 'LF-only fitted-model/source/data replay authenticated',
         'limit': 'Local release only; archive transport is authenticated by package_manifest.json and package_verification.json. No commit, push, or external upload; native reruns require licensed COMSOL. Float32 neural replay preserves the original batch layout; the retained subset-layout attempt differed by1.05e-8.'},
    ]
    report = {'reported_at_utc': utc(), 'claims': claims, 'decomposition': decomposition,
              'mass_weighted_reference_diagnosis': mass, 'matched_budget_aggregates': aggregates,
              'training_only_uncertainty_selection': uq,
              'prospective': evaluation, 'geometry_paired_prospective_comparisons': paired_comparisons(output),
              'engineering': engineering,
              'native_failures': [r for execution in (cal_execution, test_execution) for r in execution['outcomes'] if r['exit_code']],
              'prospective_LF_reported_cost': lf_reported_cost(output),
              'actual_fitted_model_replay': model_replay,
              'no_test_label_retuning': True}
    write_json(output/'improvement_report.json', report, overwrite=True)
    lines = ['# P1 improvement study — claim-by-claim results', '',
             'The audited baseline and its nominal predictions remain unchanged. New models, independent raw references, negative outcomes and provenance are retained separately.', '',
             '## Claim disposition', '', '| Stage | Claim | Outcome |', '| --- | --- | --- |']
    lines += [f'| {c["stage"]} | {c["claim"]} | {c["status"]} |' for c in claims]
    lines += ['', '## Numerical reference interpretation', '',
              'All prescribed native state extractions completed. The state includes displacement and dimensionless shell-director displacement on both faces and every clipped core wall, with thickness, offsets, coupling and rotary inertia. Native mass integrals validate the common geometry. Principal angles measure a two-mode span; no individual shape or complete spectrum is declared converged.', '',
              '| Run / mesh pair | Fine quadrature maximum angle (degrees) | Angle change across quadratures | Minimum individual MAC |', '| --- | ---: | ---: | ---: |']
    for run, result in mass['runs'].items():
        for pair, value in result['comparisons'].items():
            fine = value['levels'][-1]
            lines.append(f'| {run} / {pair} | {max(fine["principal_angles_degrees"]):.8g} | {value["quadrature_angle_change_degrees"]:.8g} | {min(fine["individual_mac"]):.8g} |')
    lines += ['', '## Matched HF budgets', '',
              'Identical nested geometry subsets; representation fitted within each counted budget. Two train-only validation partitions and two configurations per tuned family. The fixed original RBF is also retained. These are historical-test comparisons, not prospective model-selection evidence.', '',
              '| Family / HF budget | Completed trials | Geometry-median field RMS, median over repeats | Geometry-median frequency error %, median over repeats |', '| --- | ---: | ---: | ---: |']
    for key, entry in aggregates.items():
        field = entry.get('geometry_median_field_rms', {}).get('median_over_repeats')
        frequency = entry.get('geometry_median_frequency_error_pct', {}).get('median_over_repeats')
        lines.append(f'| {key} | {entry["complete"]}/{entry["attempted"]} | {field} | {frequency} |')
    lines += ['', '## Prospective accepted-mode accuracy', '',
              f'Method freeze: `{frozen["frozen_at_utc"]}`. Geometry generation: `{generation["generated_at_utc"]}`. LF-only predictions: `{blind["predicted_at_utc"]}`. No prospective HF input is supplied to predictors.', '',
              '| Group / model | Accepted geometries / rows | Geometry-median field RMS | Geometry-median frequency error % | Row field P95 |', '| --- | ---: | ---: | ---: | ---: |']
    for group, result in evaluation['groups'].items():
        for name, entry in result.items():
            if not isinstance(entry, dict) or 'point_metrics' not in entry:
                continue
            metric = entry['point_metrics']
            lines.append(f'| {group} / {name} | {metric["geometries"]} / {metric["rows"]} | {metric["geometry_median_field_rms"]["median"]:.8g} | {metric["geometry_median_frequency_error_pct"]["median"]:.8g} | {metric["field_rms"]["p95"]:.8g} |')
    lines += ['', '## Independent calibration and useful uncertainty', '',
              'Raw scales remain selected globally and regionally; local residual scales were not promoted. Both recipes are independently calibrated and compared diagnostically without changing the frozen selection. Frequency interval score is in Hz; relative width is descriptive, not a proper score.', '',
              '| IID selected-mean interval | Joint run coverage | Mean field width | Mean frequency width Hz | Field interval score | Frequency interval score Hz |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for level, intervals in selected_uq.items():
        for kind in ('marginal', 'joint'):
            if kind not in intervals:
                continue
            value = intervals[kind]
            covered = value['joint_covered']
            lines.append(f'| {level}% {kind} | {covered["successes"]}/{covered["total"]} | {value["mean_field_width"]:.8g} | {value["mean_frequency_width_hz"]:.8g} | {value["mean_field_interval_score"]:.8g} | {value["mean_frequency_interval_score_hz"]:.8g} |')
    lines += ['', 'Full Clopper–Pearson coverage intervals, Spearman discrimination, retained-risk/referral curves, RMS balls, mode rows, quarantines and geometry-paired bootstrap differences are in the JSON/CSV artifacts. A finite rank is not a sharpness or shift-coverage guarantee.', '',
              '## HF-verified engineering screening', '',
              f'Threshold: **{frozen["frequency_screen_threshold_hz"]:.9g} Hz**, fixed from historical training labels. Only stable reference label1 is screened. Decisions were recorded before evaluation HF; uncertain identities or unsupported intervals refer to HF.', '',
              '| Group / policy | Cases | HF referrals | False acceptances | False rejections | Verified correct saved calls | Unverified automatic decisions | Observed criterion |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for key, entry in engineering['results'].items():
        lines.append(f'| {key} | {entry["cases"]} | {entry["HF_referrals"]} | {entry["false_acceptances"]} | {entry["false_rejections"]} | {entry["verified_correct_saved_HF_calls"]} | {entry["unverified_automatic_decisions"]} | {entry["observed_operational_benefit_criterion"]} |')
    lines += ['', 'The validation campaign acquired every HF reference. Operational savings are counterfactual, not a refund of validation or setup cost. Per-policy training/calibration counts, fit/inference time, measured native reference time and finite-sample false-acceptance uncertainty are preserved. Historical training solve times are unavailable and are not fabricated.', '',
              f'LF solve/save reported-time sums (0.1-second log rounding retained): `{report["prospective_LF_reported_cost"]["groups"]}`. Full JSON identifies rounding limits and omitted process startup.', '',
              '## Reproduction and release', '',
              'The two actual ZIP parts are `audited_baseline.zip` and `p1_improvement_artifacts.zip`; `package_manifest.json` identifies their hashes. The baseline contains the original source, raw runs, models and saved native MPH references, not just hashes. The improvement part contains every study checkpoint, source revision, fitted weight, prospective raw bundle, blinded prediction, calibration, decision, log and report.', '',
              'Run `python p1_delivery.py replay --output <study-root>` for a no-refit LF-only model replay. Run `python p1_delivery.py verify --output <study-root>` for full archive SHA-256 and every improvement-member hash check. See the repository data README for restoration and fresh-run stage order. COMSOL requires an installed license; PyTorch weights are CPU-loaded. No upload or source commit was performed.', '',
              '## Retained limitations', '']
    lines += [f'- **Stage {claim["stage"]}:** {claim["limit"]}' for claim in claims]
    (output/'P1_IMPROVEMENT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'claim_report_written': True, 'claims': len(claims), 'native_failures': len(report['native_failures'])}), flush=True)


def replay(output):
    require_completed(output)
    source = output/'execution_sources/freeze-models'
    command = """import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from p1_prospective import LFInputs,check_freeze
from p1_improved_models import load_model
from p1_design import write_json
output=Path(sys.argv[2]); frozen=check_freeze(output)
inputs=LFInputs.load(output/'prospective/lf_inputs.npz')
assert not hasattr(inputs,'hf') and not hasattr(inputs,'f_hf')
# Preserve the blinded float32 neural GEMM batch layout, not just the weights.
rows=np.arange(inputs.n_rows)
results={}
for name in frozen['models']:
    prediction=load_model(output/'frozen_models'/name).predict_rows(inputs,rows)
    with np.load(output/'prospective/blinded_predictions'/f'{name}.npz',allow_pickle=False) as saved:
        errors={}
        for key in prediction:
            np.testing.assert_allclose(prediction[key],saved[key][rows],rtol=1.e-10,atol=1.e-12)
            errors[key]=float(np.max(np.abs(prediction[key]-saved[key][rows])))
    boundary_quantity=(inputs.lf[rows]+prediction['correction']
                       if frozen['models'][name]['configuration']['family']=='direct_hf'
                       else prediction['correction'])
    np.testing.assert_array_equal(boundary_quantity[:,~inputs.interior_mask],0.)
    results[name]=errors
write_json(output/'delivery_model_replay.json',{'replayed_without_HF_input':True,'original_blinded_batch_layout':True,'rows':rows.tolist(),'models':results},overwrite=True)
print(json.dumps(results))
"""
    environment = os.environ.copy()
    environment['P1_ORIGINAL_ROOT'] = str(ROOT)
    result = subprocess.run([sys.executable, '-u', '-c', command, str(source), str(output)],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300)
    print(result.stdout, end=''); print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise RuntimeError('Actual frozen-model replay failed')


def runtime_manifest():
    versions = {}
    for name in ('numpy', 'scipy', 'matplotlib', 'mph', 'torch', 'jpype1'):
        versions[name] = importlib.metadata.version(name)
    return {'recorded_at_utc': utc(), 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'packages': versions,
            'native_requirement': 'Installed licensed COMSOL Multiphysics6.4 with Structural Mechanics; binaries and license are not redistributed.',
            'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'seeds': {'study': SEED, 'model': 42, 'neural_deterministic_algorithms': True},
            'original_root_at_execution': str(ROOT)}


def pack(output):
    require_completed(output)
    if not (output/'delivery_model_replay.json').is_file():
        raise ValueError('Actual model replay is required before packaging')
    archive = output/'p1_improvement_artifacts.zip'
    package = output/'package_manifest.json'
    if archive.exists() or package.exists():
        raise FileExistsError('Do not replace an existing release')
    write_json(output/'runtime_environment.json', runtime_manifest(), overwrite=True)
    baseline = read_json(output/'baseline_manifest.json')
    excluded = {'audited_baseline.zip', archive.name, package.name, 'package_verification.json'}
    entries = [(path, 'study/'+path.relative_to(output).as_posix()) for path in sorted(output.rglob('*'))
               if path.is_file() and path.name not in excluded and '__pycache__' not in path.parts]
    entries += [(ROOT/name, 'source/plate_main/'+name) for name in SOURCE_FILES]
    size = sum(path.stat().st_size for path, _ in entries)
    if shutil.disk_usage(output).free < size+2**30:
        raise OSError('Insufficient space for an actual artifact archive plus1GiB margin')
    members = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as target:
        for path, name in entries:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream, target.open(name, 'w', force_zip64=True) as member:
                while chunk := stream.read(8*2**20):
                    digest.update(chunk); member.write(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Artifact changed during package creation: {path}')
            members.append({'path': name, 'bytes': before.st_size, 'sha256': digest.hexdigest()})
        target.writestr('archive_members.json', json.dumps({'created_at_utc': utc(), 'members': members}, indent=2)+'\n')
    write_json(package, {'created_at_utc': utc(), 'local_release_only': True,
        'parts': [{'path': baseline['archive'], 'bytes': baseline['archive_bytes'], 'sha256': baseline['archive_sha256']},
                  {'path': archive.name, 'bytes': archive.stat().st_size, 'sha256': file_sha256(archive)}],
        'improvement_members': len(members), 'improvement_source_bytes': size,
        'restore': ['Extract audited_baseline.zip to a new plate_main directory.',
                    'Extract the improvement archive to a separate release directory; its study/ is the study root.',
                    'Copy source/plate_main/ over the restored plate_main for the documented workflow and reports.',
                    'Set P1_ORIGINAL_ROOT to the restored plate_main; do not rewrite execution snapshots or frozen hashes.',
                    'The original source snapshot is retained inside the first archive; overlay only new workflow/docs/tests.',
                    'Run replay with the restored study root; scientific reruns use a fresh output root and the documented stage order.'],
        'limitations': ['No commit/push/upload', 'COMSOL binaries/license not included',
                       'Recorded outcomes and negative results do not become convergence or coverage certificates.']})
    print(json.dumps({'package_created': True, 'improvement_members': len(members),
                      'improvement_archive_bytes': archive.stat().st_size}), flush=True)


def verify(output):
    package = read_json(output/'package_manifest.json')
    for part in package['parts']:
        path = output/part['path']
        if path.stat().st_size != part['bytes'] or file_sha256(path) != part['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}')
        print(f'VERIFIED actual archive SHA256: {path.name}', flush=True)
    checked = 0
    with zipfile.ZipFile(output/'p1_improvement_artifacts.zip') as archive:
        members = json.loads(archive.read('archive_members.json'))['members']
        if set(archive.namelist()) != {member['path'] for member in members} | {'archive_members.json'}:
            raise ValueError('Improvement archive contains missing or unauthenticated members')
        for member in members:
            digest = hashlib.sha256(); size = 0
            with archive.open(member['path']) as stream:
                while chunk := stream.read(8*2**20):
                    digest.update(chunk); size += len(chunk)
            if size != member['bytes'] or digest.hexdigest() != member['sha256']:
                raise ValueError(f'Archive member changed: {member["path"]}')
            checked += 1
    write_json(output/'package_verification.json', {'verified_at_utc': utc(),
        'both_actual_archive_sha256_matched': True, 'improvement_members_stream_verified': checked,
        'package_manifest_sha256': file_sha256(output/'package_manifest.json')})
    print(json.dumps({'release_verified': True, 'improvement_members_stream_verified': checked}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('report', 'replay', 'pack', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'report': make_report, 'replay': replay, 'pack': pack, 'verify': verify}[args.stage](args.output.resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())
