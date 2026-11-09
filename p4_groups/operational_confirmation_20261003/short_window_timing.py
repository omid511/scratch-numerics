"""Matched-end stationary/ramp controls for the current causal128ms aperture."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import diagnostic_matrix as M
from cuda_backend import estimate_batch

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'short_window_timing_20261007'


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'results.json').exists(), 'Keep completed timing controls immutable'
    if not torch.cuda.is_available():
        raise RuntimeError('Physical propagation uses the existing CUDA backend; no silent CPU fallback')
    d, fingerprint = M.G.inputs()
    metadata_path = M.G.O.ROOT / 'p4_dataset_saturation/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert all(c['design'] in metadata['train_designs'] for c in protocol['cases'])
    rows = []
    waveforms = []
    noise_draws = []
    started = time.perf_counter()
    dt = protocol['dt_seconds']
    for case in protocol['cases']:
        _, plate = M.context(d, fingerprint, case)
        ordinary = plate.initial(case['replicate'])[0]
        projection = plate.reduced['projection']
        reference_scale = max(float(np.sqrt(np.mean((projection @ ordinary) ** 2))), 1e-30)
        white = np.random.default_rng(20261012 + int(case['design'][1:]) + case['replicate']).normal(
            size=(len(d['layout']), max(protocol['samples'])))
        noise_index = len(noise_draws)
        noise_draws.append(white)
        for samples in protocol['samples']:
            times = np.arange(samples) * dt
            for end_ratio in protocol['end_ratios']:
                for rate in protocol['ratio_rates_per_second']:
                    ratios = end_ratio - rate * (times[-1] - times)
                    reference = {name: M.truth(plate, float(ratio)) for name, ratio in
                                 (('start', ratios[0]), ('midpoint', (ratios[0] + ratios[-1]) / 2), ('end', ratios[-1]))}
                    _, vectors, critical = M.eigensystem(plate, ratios[0])
                    critical_initial = vectors[:, critical].real.cpu().numpy().copy()
                    critical_initial *= reference_scale / max(
                        float(np.sqrt(np.mean((projection @ critical_initial) ** 2))), 1e-30)
                    ratio_function = lambda t, end=end_ratio, speed=rate, finish=times[-1]: end - speed * (finish - t)
                    for excitation, initial in (('ordinary', ordinary), ('critical_only_oracle', critical_initial)):
                        raw, _ = M.propagate(plate, times, initial, ratio_function, np.zeros(samples))
                        wave_index = len(waveforms)
                        # Pad only stored artifacts; estimation uses exactly the declared samples.
                        padded = np.full((len(d['layout']), max(protocol['samples'])), np.nan)
                        padded[:, :samples] = raw
                        waveforms.append(padded)
                        for noise in protocol['noise']:
                            injected = np.zeros_like(raw) if noise == 'clean' else .01 * reference_scale * white[:, :samples]
                            signal, _, _ = M.normalize(raw, injected)
                            mode = estimate_batch([signal], dt=dt)[0]
                            row = dict(design=case['design'], replicate=case['replicate'], samples=samples,
                                       span_seconds=float(times[-1]), end_ratio=end_ratio, ratio_rate_per_second=rate,
                                       start_ratio=float(ratios[0]), excitation=excitation, noise=noise,
                                       absolute_noise_rms_m=float(np.sqrt(np.mean(injected ** 2))),
                                       saturation_fraction=float(np.mean(abs(signal) >= 50)),
                                       wave_index=wave_index, noise_index=noise_index, references=reference,
                                       reference_frequency_continuous=(max(v['frequency'] for v in reference.values()) /
                                                                       min(v['frequency'] for v in reference.values()) <= 1.05),
                                       end_quality=M.quality(mode, reference['end']), reference_errors={})
                            if len(mode['alpha']):
                                k = int(np.argmin(abs(mode['frequency'] - reference['end']['frequency'])))
                                alpha = float(mode['alpha'][k])
                                frequency = float(mode['frequency'][k])
                                row.update(estimated_alpha=alpha, estimated_frequency=frequency,
                                           end_frequency_retained=abs(frequency / reference['end']['frequency'] - 1) <= .05)
                                row['reference_errors'] = {name: dict(signed_alpha_error=alpha - ref['alpha'],
                                    frequency_relative_error=abs(frequency / ref['frequency'] - 1))
                                    for name, ref in reference.items()}
                            else:
                                row.update(estimated_alpha=None, estimated_frequency=None, end_frequency_retained=False)
                            rows.append(row)
        print('SHORT_TIMING_CASE_COMPLETE', case['design'], flush=True)
    assert len(rows) == (len(protocol['cases']) * len(protocol['samples']) * len(protocol['end_ratios']) *
                         len(protocol['ratio_rates_per_second']) * len(protocol['excitation']) * len(protocol['noise']))
    # Paired fixed counterparts share end ratio, samples, excitation and noise.
    fixed = {(r['design'], r['samples'], r['end_ratio'], r['excitation'], r['noise']): r
             for r in rows if r['ratio_rate_per_second'] == 0}
    for row in rows:
        match = fixed[(row['design'], row['samples'], row['end_ratio'], row['excitation'], row['noise'])]
        row['paired_fixed_alpha'] = match['estimated_alpha']
        row['paired_both_uncapped_frequency_retained'] = (row['end_frequency_retained'] and match['end_frequency_retained']
                                                         and row['saturation_fraction'] == match['saturation_fraction'] == 0)
        row['ramp_minus_fixed_alpha'] = None if row['estimated_alpha'] is None or match['estimated_alpha'] is None else (
            row['estimated_alpha'] - match['estimated_alpha'])
    paths = (Path(__file__), Path(M.__file__), Path(M.P.__file__), Path(M.G.__file__), SCRIPT / 'cuda_backend.py')
    result = dict(protocol=protocol, fingerprint=fingerprint,
                  source_sha256={p.name: M.G.O.C.sha256(p) for p in paths},
                  dataset_metadata_sha256=M.G.O.C.sha256(metadata_path), backend=M.G.backend_signature(),
                  rows=rows, elapsed_seconds=time.perf_counter() - started)
    M.G.O.C.save_npz(OUT / 'raw_controls.npz', physical=np.stack(waveforms), noise_white=np.stack(noise_draws),
                      records=json.dumps(M.clean_json(rows)), protocol=json.dumps(protocol))
    M.G.O.C.save_json(OUT / 'results.json', M.clean_json(result))
    print('SHORT_TIMING_COMPLETE', len(rows), result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
