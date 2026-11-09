"""Recompute continuous research attainment and separate qualification checks."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
LEDGER = ROOT / 'P4_PROGRESS.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(name, path, checks, recovered):
    if len(checks)!=6 or any(type(value) is not bool for value in checks.values()):
        raise ValueError('Expected all six frozen boolean estimator checks')
    if not 0<=recovered<=48:
        raise ValueError('Invalid pole-recovery count')
    passed = sum(checks.values())
    return dict(name=name,source=str(path.relative_to(ROOT)),source_sha256=sha(path),
        passed_checks=passed,total_checks=6,percentage=100*passed/6,checks=checks,
        poles_recovered=recovered,pole_denominator=48,pole_recovery_percentage=100*recovered/48)


def _attainment(value, target, direction):
    """Unknown is not a measured zero; lower-bound credit is handled separately."""
    if target <= 0 or not math.isfinite(target) or direction not in ('lower', 'higher'):
        raise ValueError('Expected a positive finite target and a scoring direction')
    if value is None:
        return None
    if not math.isfinite(value) or (direction == 'lower' and value < 0):
        raise ValueError('Invalid observed metric')
    ratio = (1 if value == 0 else target/value) if direction == 'lower' else value/target
    return 100*min(1, max(0, ratio))


def _criterion_metrics():
    """Use one frozen development predictor, never cherry-picked comparator metrics."""
    directory = SCRIPT/'fitter_margin_comparison_20261008'
    prediction_path = directory/'predictions.npz'
    result_path = directory/'results.json'
    verification = json.loads((directory/'verification.json').read_text())
    for key, path in (('predictions_sha256', prediction_path), ('results_sha256', result_path)):
        if verification[key] != sha(path):
            raise ValueError('Comparison evidence changed since verification')
    result = json.loads(result_path.read_text())
    acquisition_path = SCRIPT/'temporal_probe_20261007/protocol.json'
    acquisition = json.loads(acquisition_path.read_text())
    with np.load(prediction_path, allow_pickle=False) as archive:
        names = json.loads(str(archive['prediction_names']))
        prediction = archive['predictions'][names.index('framewise')]
        truth, observed, times = archive['truth'], archive['observed'], archive['times']
        records = json.loads(str(archive['records']))
    ids = np.array([row['design'] for row in records])
    held = np.array([row['split'] == 'held' for row in records])
    noise_names = np.array([row['noise'] for row in records])
    primary = ('clean', 'iid1', 'correlated1')
    metrics = {}

    def add(criterion, name, value, target, direction, **details):
        metrics.setdefault(criterion, []).append(dict(
            name=name, value=value, target=target, direction=direction, **details))

    for noise in primary:
        selected = held & (noise_names == noise)
        y, estimate, current = truth[selected], prediction[selected], observed[selected]
        designs = ids[selected]
        error = estimate-y
        finite = np.isfinite(estimate)
        boundary = (y > -.02) & (y < .05)
        for label, population, target in (
                ('all_frame_mae', np.ones_like(y, dtype=bool), .010 if noise == 'clean' else .015),
                ('boundary_mae', boundary, .005 if noise == 'clean' else .010)):
            per_design = {}
            for design in np.unique(designs):
                mask = population & (designs[:, None] == design)
                if mask.any():
                    per_design[design] = float(np.abs(error)[mask].mean()) if finite[mask].all() else None
            value = (float(np.mean(list(per_design.values())))
                     if per_design and all(v is not None for v in per_design.values()) else None)
            add('signed_headroom', noise+'/'+label, value, target, 'lower',
                population_frames=int(population.sum()), per_design=per_design)
        optimistic = (float((error[boundary] > .02).mean())
                      if boundary.any() and finite[boundary].all() else None)
        add('dangerous_point_errors', noise+'/boundary_optimism_rate', optimistic, .05, 'lower',
            population_frames=int(boundary.sum()))
        for side, side_mask in (('stable', y > 0), ('unstable', y < 0)):
            mask = current & side_mask & (np.abs(y) >= .01) & (np.abs(y) <= .05)
            wrong_sign = (float((np.sign(estimate[mask]) != np.sign(y[mask])).mean())
                          if mask.any() and finite[mask].all() else None)
            add('dangerous_point_errors', noise+'/'+side+'_wrong_sign_rate',
                wrong_sign, .10, 'lower', population_frames=int(mask.sum()),
                missing_reason=None if mask.any() else 'No eligible current points on this sign side')
        events = [row for row in result['temporal_cross_warning']
                  if row['record']['split'] == 'held' and row['record']['noise'] == noise
                  and row['readout'] == 'framewise' and row['budget'] == '0.05']
        leads = [row['flutter']['lead_seconds'] for row in events]
        timely = sum(lead is not None and lead >= 1 for lead in leads)
        add('timely_warning', noise+'/timely_crossing_fraction',
            timely/len(leads) if leads else None, .90, 'higher',
            crossing_count=len(leads), timely_count=timely, lead_seconds=leads,
            limitation='Exploratory frame-budget policy, not qualified episode-burden policy')
        add('timely_warning', noise+'/timely_miss_upper95', None, .10, 'lower',
            missing_reason='No independent design-run bound under the final warning contract')

    add('useful_clearance', 'independent_lower_bound_coverage', None, .95, 'higher',
        missing_reason='Current framewise predictor has no independently calibrated lower-bound output')
    add('useful_clearance', 'false_clearance_design_run_upper95', None, .05, 'lower',
        missing_reason='No independently evaluated clearance policy for this predictor')
    for noise in primary:
        for population in ('final2s_qualified_current_fraction', 'farther_stable_usable_bound_fraction'):
            add('nonvacuous_usefulness', noise+'/'+population, None, .80, 'higher',
                missing_reason='Accepted input or a finite point prediction is not a qualified advisory')
    add('nuisance_burden', 'independent_nuisance_episodes_per_hour_upper95', None, 1, 'lower',
        missing_reason='No independent episode-contract exposure/bound for this policy')
    add('comparative_impact', 'matched_nuisance_timely_miss_reduction', None, .25, 'higher',
        missing_reason='Existing frame budgets are not matched held episode/hour nuisance rates')
    add('comparative_impact', 'paired_design_improvement_evidence', None, 1, 'higher',
        missing_reason='No prespecified paired interval excluding no improvement')

    disclosures = {}
    for noise in ('clean', 'iid1', 'correlated1', 'iid5', 'correlated5'):
        mask = held & (noise_names == noise)
        events = [row for row in result['temporal_cross_warning']
                  if row['record']['split'] == 'held' and row['record']['noise'] == noise
                  and row['readout'] == 'framewise' and row['budget'] == '0.05']
        disclosures[noise] = dict(
            accepted_input_fraction=float(observed[mask].mean()) if mask.any() else None,
            lead_seconds=[row['flutter']['lead_seconds'] for row in events])
    for noise in ('iid5', 'correlated5'):
        add('stress_behavior', noise+'/false_clearance_design_run_upper95', None, .05, 'lower',
            missing_reason='Stress evaluation has no independently calibrated clearance-risk bound')
    disclosed = sum(disclosures[noise]['accepted_input_fraction'] is not None
                    and bool(disclosures[noise]['lead_seconds']) for noise in ('iid5', 'correlated5'))
    add('stress_behavior', 'stress_conditions_with_coverage_and_lead_disclosure',
        disclosed, 2, 'higher', condition_metrics=disclosures,
        limitation='Accepted-input availability is reported, not qualified-advisory availability')
    aperture = acquisition['samples']*float(np.median(np.diff(times)))/acquisition['stride_samples']
    add('online_feasibility', 'acquisition_seconds', aperture, .5, 'lower')
    timing_path = SCRIPT/'framewise_latency_20261008/results.json'
    timing_verification_path = timing_path.with_name('verification.json')
    timing = json.loads(timing_path.read_text())
    timing_verification = json.loads(timing_verification_path.read_text())
    if (timing_verification['results_sha256'] != sha(timing_path)
            or timing_verification['measurements_sha256'] != sha(timing_path.with_name('measurements.npz'))
            or timing_verification['verification_source_sha256'] != sha(SCRIPT/'verify_framewise_latency.py')):
        raise ValueError('Timing evidence does not match independent verification')
    if (timing['candidate'] != 'frozen_framewise_temporal_probe'
            or timing['source_sha256'] != sha(SCRIPT/'measure_framewise_latency.py')
            or timing['inputs'][str(prediction_path.relative_to(ROOT))] != sha(prediction_path)
            or timing['protocol_sha256'] != sha(timing_path.with_name('protocol.json'))):
        raise ValueError('Timing evidence is not for the current frozen candidate')
    if (timing_verification['scored_p95_ms'] != timing['scored_p95_ms']
            or timing_verification['replay_valid'] != timing['replay_valid']):
        raise ValueError('Independent timing statistics or replay verdict disagree')
    add('online_feasibility', 'full_path_p95_ms', timing['scored_p95_ms'], 10, 'lower',
        hardware=timing['hardware'], definition='Worst noise-condition accepted-path p95',
        replay_valid=timing['replay_valid'], replay=timing['replay'],
        source=str(timing_path.relative_to(ROOT)), accepted=timing['accepted'],
        limitation='Measured computation receives partial credit, not passed replay or qualified deployment; initialization and sensor transport separate')
    reported = sum(all(field in timing['per_noise'][noise]['accepted'] for noise in timing['per_noise'])
                   for field in ('p99_ms', 'max_ms', 'deadline_misses'))
    add('online_feasibility', 'p99_max_deadline_fields_reported', reported, 3, 'higher',
        condition_metrics=timing['per_noise'], all_frames=timing['all_frames'])
    add('independent_credibility', 'fresh_blind_designs_evaluated', 0, 120, 'higher',
        limitation='These two inspected held designs are development data, not fresh blind designs')
    add('independent_credibility', 'independently_validated_response_label_sources', 0, 1, 'higher',
        limitation='Same-equation replay is not an independent physical validation source')
    sources = (prediction_path, result_path, directory/'verification.json', acquisition_path,
               timing_path, timing_verification_path)
    return metrics, dict(
        candidate='frozen_framewise_temporal_probe', population='Two previously inspected held TRAIN designs',
        primary_conditions=list(primary), warning_frame_budget=.05,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sources},
        limitation='Development attainment; not blind validation, qualification or safety confidence')


def _update_attainment(endpoint):
    metrics, evidence = _criterion_metrics()
    if set(metrics) != {gate['id'] for gate in endpoint['gates']}:
        raise ValueError('Every research criterion must have its own metric components')
    total_weight = 0
    for gate in endpoint['gates']:
        weight = gate.setdefault('weight', .1)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Every criterion needs a positive finite weight')
        total_weight += weight
        components = metrics[gate['id']]
        for component in components:
            component['percentage'] = _attainment(
                component['value'], component['target'], component['direction'])
        known = [c['percentage'] for c in components if c['percentage'] is not None]
        gate.update(components=components, percentage=sum(known)/len(components),
            measured_component_fraction=len(known)/len(components),
            score_status='unmeasured' if not known else 'partial' if len(known) < len(components) else 'measured')
        if gate['id'] == 'online_feasibility':
            timing_component = components[1]
            replay = timing_component['replay']
            gate.update(status='measured_not_qualified',
                evidence_note=(
                    f"Warm serialized GTX1660SUPER/i5-9400F full-path worst-condition accepted p95="
                    f"{timing_component['value']:.6f}ms versus10ms target; "
                    f"{timing_component['accepted']['deadline_misses']}/"
                    f"{timing_component['accepted']['count']} accepted decisions miss10ms. "
                    f"{replay['prediction_mismatches']}/{replay['frames']} margin outputs exceed frozen replay tolerance; "
                    "availability, age and warning decisions match. Timing/reporting receives partial metric credit, "
                    "not passed replay or qualified deployment. Physical sampling, initialization and sensor transport separate."),
                evidence_sources=[timing_component['source'],
                    'p4_groups/operational_confirmation_20261003/framewise_latency_20261008/verification.json',
                    'p4_groups/operational_confirmation_20261003/framewise_latency_20261008/protocol.json'])
    endpoint['percentage'] = sum(g['weight']*g['percentage'] for g in endpoint['gates'])/total_weight
    endpoint['measured_weight_percentage'] = 100*sum(
        g['weight']*g['measured_component_fraction'] for g in endpoint['gates'])/total_weight
    endpoint['unknown_weight_percentage'] = 100-endpoint['measured_weight_percentage']
    endpoint['possible_upper_percentage'] = endpoint['percentage']+endpoint['unknown_weight_percentage']
    endpoint['qualification_percentage'] = 100*endpoint['passed_gates']/endpoint['total_gates']
    endpoint['scoring'] = dict(version=2, evidence=evidence,
        metric_rule='lower:100*min(1,target/value), zero error=100; higher:100*clip(value/target,0,1)',
        component_rule='Equal weight within each criterion; unknown receives no evidenced credit, not a performance value of zero',
        overall_rule='sum(criterion_weight*criterion_percentage)/sum(criterion_weight)',
        missing_rule='Headline is documented-attainment lower bound; disclose unknown weight and possible score upper bound')


def run():
    ledger=json.loads(LEDGER.read_text())
    baseline_path=SCRIPT/'time_varying_fit_20261007/results.json'
    curvature_path=SCRIPT/'modal_transport_20261008/curvature_analysis.json'
    selected_path=SCRIPT/'noise_aware_carrier_20261008/results.json'
    verified_path=selected_path.with_name('verification.json')
    baseline=json.loads(baseline_path.read_text())
    curvature=json.loads(curvature_path.read_text())
    selected=json.loads(selected_path.read_text())
    verification=json.loads(verified_path.read_text())
    if verification['result_sha256']!=sha(selected_path):
        raise ValueError('Selection result does not match independent verification')
    if verification['independent_gate_checks']!=selected['critical_pass_checks']['booleans']:
        raise ValueError('Verified selection gates disagree')
    history=[snapshot('linear_carrier',baseline_path,baseline['critical_pass_checks']['booleans'],
        baseline['critical_pass_checks']['checks']['all48_end_poles_match']['matched']),
        snapshot('free_curvature',curvature_path,curvature['physical_pass_checks'],
        sum(s['matches'] for s in curvature['summaries'] if s['arm']=='physical')),
        snapshot('noise_aware_selection',selected_path,selected['critical_pass_checks']['booleans'],
        selected['critical_pass_checks']['checks']['all48_end_poles_match']['matched'])]
    current=history[-1]
    if current['percentage']!=selected['progress']['percentage']:
        raise ValueError('Recorded stage percentage is inconsistent')
    endpoint=ledger['endpoint']
    if len(endpoint['gates'])!=endpoint['total_gates'] or endpoint['total_gates']!=10:
        raise ValueError('Changing the endpoint denominator requires an explicit criterion revision')
    if any(type(gate['passed']) is not bool for gate in endpoint['gates']):
        raise ValueError('Endpoint criterion status must be an evidenced boolean')
    # Qualification remains separate from continuous development attainment.
    endpoint['passed_gates']=sum(gate['passed'] for gate in endpoint['gates'])
    _update_attainment(endpoint)
    ledger['current_stage'].update(percentage=current['percentage'],passed_checks=current['passed_checks'],
        total_checks=6,current_candidate=current['name'],history=history,
        decision='Not promoted: noisy guardrail restored but recovery falls45->40/48; physical transport mismatch remains.',
        independent_verification=str(verified_path.relative_to(ROOT)),verification_sha256=sha(verified_path))
    ledger['updater_sha256']=sha(Path(__file__))
    LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
    print('P4_PROGRESS_UPDATE',json.dumps(dict(
        overall_progress=dict(percentage=endpoint['percentage'],
            measured_weight_percentage=endpoint['measured_weight_percentage'],
            unknown_weight_percentage=endpoint['unknown_weight_percentage'],
            possible_upper_percentage=endpoint['possible_upper_percentage'],
            qualification_passed=endpoint['passed_gates'],total=10,
            criteria=[dict(id=gate['id'],percentage=gate['percentage'],weight=gate['weight'],
                score_status=gate['score_status'],passed=gate['passed']) for gate in endpoint['gates']]),
        diagnostic=dict(stage_percentage=current['percentage'],stage_passed=current['passed_checks'],stage_total=6,
            pole_recovery=current['poles_recovered'],decision=ledger['current_stage']['decision']))),flush=True)


if __name__=='__main__':
    run()
