"""Reproduce this fixed training-design audit's two fresh evaluation draws.

Run from repository root with the existing .venv. No reserved rows are read.
This is an experiment artifact, not a new production extractor.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_feature_row, physics_feature_groups
from mechanics.p4_margin_estimation.sidecar import extract
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize


def feature_row(signal, dt, ridge):
    modes = estimate_modes(signal, dt)
    groups = physics_feature_groups(modes)
    a, f, e = modes['alpha'], modes['frequency'], modes['energy_fraction']
    visible = np.flatnonzero(e >= .01)
    if len(visible) > 1:
        order = visible[np.lexsort((f[visible], -a[visible]))]
        lead, others = order[0], order[1:]
        neighbor = others[np.argmin(abs(f[others] - f[lead]))]
        natural = np.hypot(a, 2*np.pi*f)
        groups['spacing'][0] = 2*abs(f[lead]-f[neighbor])/(f[lead]+f[neighbor])
        groups['spacing'][2] = 2*np.hypot(a[lead]-a[neighbor], 2*np.pi*(f[lead]-f[neighbor]))/(natural[lead]+natural[neighbor])
    correlation = extract([SimpleNamespace(sensor_signals=signal, dt=dt)], ['xcorr'])[0]
    return np.concatenate([ridge._extract_features(signal, dt=dt), modal_feature_row(modes), *groups.values(), correlation])


def main():
    output = Path(__file__).resolve().parent
    with np.load(output / 'shared.npz') as z:
        ids, rows, reference = z['ids'], z['source_indices'], z['evaluation'][-2:]
    protocol = json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())
    layout = np.array(protocol['selected_sensor_indices'])
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        dts = z['dts']
    rowmap = {int(row): i for i, row in enumerate(rows)}
    fresh = np.empty_like(reference)
    ridge = PhysicsFeatureRidge()
    for di, did in enumerate(np.unique(ids)):
        with np.load(ROOT / 'p4_sensor_candidates' / (str(did)+'.npz')) as z:
            raw, source, scales = z['raw'], z['source_indices'], z['noise_scale']
        for j, index in enumerate(source):
            k = rowmap[int(index)]
            white = np.random.default_rng(np.random.SeedSequence([932, int(index)])).normal(size=raw[j].shape)
            common = np.random.default_rng(np.random.SeedSequence([933, int(index)])).normal(size=(1, raw[j].shape[1]))
            for ni, noise in enumerate((white, np.sqrt(.2)*white + np.sqrt(.8)*common)):
                measured = raw[j] + .01*scales[j]*noise
                normalized = causal_calibration_normalize(measured, calibration_samples=51, normalize_mode='per_channel')
                signal = np.clip(normalized, -50, 50).astype(np.float32)[layout]
                fresh[ni, k] = feature_row(signal, float(dts[index]), ridge)
        if (di+1) % 20 == 0:
            print('Replayed designs', di+1, flush=True)
    np.testing.assert_allclose(fresh, reference, atol=1e-10, rtol=1e-10)
    print('Fresh evaluation feature replay passed:', fresh.shape, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
