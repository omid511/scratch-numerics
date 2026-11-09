"""Replay stored physical controls: distinguish pole selection from missing poles."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import short_window_timing as S
from cuda_backend import estimate_batch


def summarize(rows):
    usable = [r for r in rows if r['end_frequency_retained'] and r['saturation_fraction'] == 0]
    paired = [r for r in usable if r['paired_both_uncapped_frequency_retained']]
    median = lambda values: float(np.median(values)) if values else None
    return dict(windows=len(rows), identified=sum(r['end_quality']['identified'] for r in rows),
                uncapped_frequency_retained=len(usable),
                nearest_frequency_end_recovered=sum(r['end_quality']['recovered'] for r in rows),
                any_end_matching_pole=sum(r['any_end_matching_pole'] for r in rows),
                energy_ge1pct_end_matching_pole=sum(r['energy_ge1pct_end_matching_pole'] for r in rows),
                reference_frequency_continuous=sum(r['reference_frequency_continuous'] for r in rows),
                current128ms_gate_accepted=sum(r['current128ms_gate_accepted'] is True for r in rows),
                median_pole_count=median([r['end_quality']['modes'] for r in rows]),
                median_alpha_absolute_error={key: median([abs(r['reference_errors'][key]['signed_alpha_error']) for r in usable])
                                             for key in ('start', 'midpoint', 'end')},
                median_end_alpha_signed_error=median([r['reference_errors']['end']['signed_alpha_error'] for r in usable]),
                paired_median_ramp_minus_fixed_alpha=median([r['ramp_minus_fixed_alpha'] for r in paired]))


def run():
    result = json.loads((S.OUT / 'results.json').read_text())
    assert result['protocol'] == json.loads((S.OUT / 'protocol.json').read_text())
    for name, expected in result['source_sha256'].items():
        assert S.M.G.O.C.sha256(S.SCRIPT / name) == expected
    with np.load(S.OUT / 'raw_controls.npz') as z:
        raw = z['physical'].copy()
        white = z['noise_white'].copy()
        assert json.loads(str(z['records'])) == result['rows']
        assert json.loads(str(z['protocol'])) == result['protocol']
    rows = result['rows']
    for row in rows:
        samples = row['samples']
        physical = raw[row['wave_index'], :, :samples]
        assert np.isfinite(physical).all()
        draw = white[row['noise_index'], :, :samples]
        # Recover the archived absolute noise floor, not a new signal-dependent one.
        noise = np.zeros_like(physical) if row['noise'] == 'clean' else (
            row['absolute_noise_rms_m'] / float(np.sqrt(np.mean(draw ** 2))) * draw)
        signal, _, _ = S.M.normalize(physical, noise)
        mode = estimate_batch([signal], dt=result['protocol']['dt_seconds'])[0]
        assert len(mode['alpha']) == row['end_quality']['modes']
        if len(mode['alpha']):
            k = int(np.argmin(abs(mode['frequency'] - row['references']['end']['frequency'])))
            np.testing.assert_allclose(mode['alpha'][k], row['estimated_alpha'], rtol=0, atol=1e-4)
            np.testing.assert_allclose(mode['frequency'][k], row['estimated_frequency'], rtol=0, atol=1e-6)
            reference = row['references']['end']
            matching = ((abs(mode['frequency'] / reference['frequency'] - 1) <= .05) &
                        (abs(mode['alpha'] - reference['alpha']) <= max(1., .1 * abs(reference['alpha']))))
            row['any_end_matching_pole'] = bool(matching.any())
            row['energy_ge1pct_end_matching_pole'] = bool((matching & (mode['energy_fraction'] >= .01)).any())
        else:
            row['any_end_matching_pole'] = row['energy_ge1pct_end_matching_pole'] = False
        row['pole_bank'] = {key: mode[key].tolist() for key in ('alpha', 'frequency', 'energy_fraction')}
        row['current128ms_gate_accepted'] = None
        row['current128ms_forecast_skill'] = None
        if samples == 128:
            uncapped = bool(np.isfinite(signal).all() and not (abs(signal) >= 50).any())
            skill = None
            if uncapped:
                prefix = estimate_batch([signal[:, :64]])
                skill = float(S.M.P.forecast_skills([signal], prefix, 64, 64)[0])
            row['current128ms_forecast_skill'] = skill
            row['current128ms_gate_accepted'] = bool(uncapped and skill is not None and skill >= .5 and len(mode['alpha']))
    grouped = {}
    for row in rows:
        key = '/'.join(str(row[k]) for k in ('samples', 'end_ratio', 'excitation', 'noise', 'ratio_rate_per_second'))
        grouped.setdefault(key, []).append(row)
    analysis = dict(scope=result['protocol']['scope'], source_sha256=S.M.G.O.C.sha256(Path(__file__)),
                    result_sha256=S.M.G.O.C.sha256(S.OUT / 'results.json'),
                    raw_sha256=S.M.G.O.C.sha256(S.OUT / 'raw_controls.npz'),
                    groups={key: summarize(group) for key, group in grouped.items()}, rows=rows,
                    limitations=['Nearest-frequency pole and any matching pole are different diagnostics; neither is deployed identification truth',
                                 'Noise reconstructed from stored draw and absolute RMS; selected poles replay within1e-4/s and1e-6Hz',
                                 'Current128ms gate is prefix forecast>=0.5 plus uncapped finite full-window modes, not model availability or calibrated confidence',
                                 'Reference-frequency-continuous means<=5% span, not proof of mode identity',
                                 'Four repeatedly inspected TRAIN designs; no new model, warning policy or deployment qualification'])
    S.M.G.O.C.save_json(S.OUT / 'analysis.json', S.M.clean_json(analysis))
    for samples in (128, 256):
        for excitation in ('ordinary', 'critical_only_oracle'):
            for noise in ('clean', 'iid1'):
                for rate in (0., .04, .15625):
                    group = [r for r in rows if (r['samples'], r['excitation'], r['noise'], r['ratio_rate_per_second']) ==
                             (samples, excitation, noise, rate)]
                    print('TIMING_GROUP', samples, excitation, noise, rate, json.dumps(summarize(group)))
    print('TIMING_POLE_BANK_REPLAY_COMPLETE', len(rows))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
