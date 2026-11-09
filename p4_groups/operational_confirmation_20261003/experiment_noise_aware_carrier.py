"""Apply fixed waveform-only selection to pinned fits; physical truth scores afterward."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import experiment_time_varying_fit as E
from noise_aware_carrier import select_carrier

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'noise_aware_carrier_20261008'
TRANSPORT = SCRIPT / 'modal_transport_20261008'


def run():
    started = time.perf_counter()
    target = OUT / 'results.json'
    if target.exists():
        raise RuntimeError('Completed experiment results are immutable')
    contract = json.loads((OUT/'protocol.json').read_text())
    for name, expected in contract['input_sha256'].items():
        if E.M.G.O.C.sha256(E.ROOT/name) != expected:
            raise RuntimeError(f'Frozen selection input mismatch: {name}')
    protocol, source, cached, analysis, provenance = E._verify_inputs()
    rows, _, physical, white = E._load_controls(source, cached, analysis)
    baseline_result = json.loads((E.OUT/'results.json').read_text())
    baseline = {E._key(r['condition']):r for r in baseline_result['critical_only_stage']['rows']}
    curved_result = json.loads((TRANSPORT/'curvature_results.json').read_text())
    curved = {E._key(r['condition']):r for r in curved_result['rows']}
    transport_result = json.loads((TRANSPORT/'results.json').read_text())
    transport = {E._key(r['condition']):r for r in transport_result['rows']}
    cp = curved_result['provenance']
    for name, path in (('implementation_sha256', SCRIPT/'evaluate_carrier_curvature.py'),
                       ('curvature_protocol_sha256', TRANSPORT/'curvature_protocol.json'),
                       ('transport_results_sha256', TRANSPORT/'results.json'),
                       ('transport_traces_sha256', TRANSPORT/'traces.npz')):
        if cp[name] != E.M.G.O.C.sha256(path):
            raise RuntimeError(f'Archived curvature provenance mismatch: {name}')
    chosen = [r for r in rows if r['samples']==128 and r['excitation']=='critical_only_oracle']
    if len(chosen)!=48 or set(baseline)!=set(curved) or set(curved)!=set(transport):
        raise RuntimeError('Expected all48 identical critical conditions')
    native_records, frozen_records = [], []
    with np.load(TRANSPORT/'traces.npz') as traces:
        for original in chosen:
            key = E._key(original)
            measured, measurement = E._prepare(original, physical, white)
            previous = baseline[key]
            if measurement['fit_signal_sha256'] != previous['measurement']['fit_signal_sha256']:
                raise RuntimeError('Pinned measured waveform changed')
            offset = np.asarray(measurement['calibration_offset'])[:, None]
            scale = np.asarray(measurement['calibration_scale'])[:, None]
            draw = white[original['noise_index'], :, :128]
            noise = np.zeros_like(draw) if original['noise']=='clean' else draw*(
                original['absolute_noise_rms_m']/np.sqrt(np.mean(draw**2)))
            frozen = np.clip((traces[f'w{original["wave_index"]}_frozen_carrier'].T+noise-offset)/scale,
                             -50, 50).astype(np.float32).astype(float)
            for arm, signal, linear, records in (
                ('physical', measured, previous['varying_fit'], native_records),
                ('frozen_carrier', frozen, transport[key]['counterfactuals']['frozen_carrier']['fit'], frozen_records)):
                selected = select_carrier(signal, source['dt_seconds'], linear, curved[key]['fits'][arm]['fit'])
                # No physical references are supplied to selection; truth is scored here.
                fit = selected['fit']
                score = E._score(fit.get('alpha_end'), fit.get('frequency_end'), original['references']['end'])
                if arm=='physical':
                    record = {**previous, 'varying_fit':fit, 'varying_end_score':score,
                              'carrier_selection':selected['selection'], 'selection_seconds':selected['elapsed_seconds']}
                else:
                    record = dict(condition=previous['condition'], fit=fit, score=score,
                                  carrier_selection=selected['selection'], references=original['references'])
                records.append(record)
    checks = E._pass_checks(native_records, protocol['pass_checks'])
    groups = []
    for noise in ('clean', 'iid1'):
        for rate in (0., .04, .15625):
            group = [r for r in native_records if r['condition']['noise']==noise
                     and r['condition']['ratio_rate_per_second']==rate]
            groups.append(dict(noise=noise, rate=rate, denominator=len(group),
                curved_selected=sum(r['carrier_selection']['chosen']=='curved' for r in group),
                recovered=sum(r['varying_end_score']['end_pole_matches_reference'] for r in group),
                alpha_mae_per_second=float(np.mean([r['varying_end_score']['alpha_absolute_error_per_second']
                    for r in group if r['varying_end_score']['available']]))))
    provenance.update(selection_protocol_sha256=E.M.G.O.C.sha256(OUT/'protocol.json'),
        inputs_sha256=contract['input_sha256'], implementation_sha256={name:E.M.G.O.C.sha256(SCRIPT/name)
            for name in ('noise_aware_carrier.py', 'experiment_noise_aware_carrier.py')})
    output = dict(protocol=contract, provenance=provenance, rows=native_records,
        frozen_carrier_rows=frozen_records, critical_pass_checks=checks, groups=groups,
        ordinary_stage=dict(status='not_run', reason='Critical-mode checkpoint only; no ordinary-mixture qualification'),
        progress=dict(passed_checks=sum(checks['booleans'].values()), total_checks=6,
                      percentage=100*sum(checks['booleans'].values())/6),
        elapsed_seconds=time.perf_counter()-started,
        execution='CPU waveform-only selection on pinned fits; no new physical trajectories or supervised fitting')
    E.M.G.O.C.save_json(target, E.M.clean_json(output))
    print('NOISE_AWARE_CARRIER_COMPLETE', json.dumps(dict(progress=output['progress'],
        checks=checks['booleans'], recovered=checks['checks']['all48_end_poles_match']['matched'],
        curved_selected=sum(r['carrier_selection']['chosen']=='curved' for r in native_records))), flush=True)
    print('SELECTION_GROUPS', json.dumps(groups), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
