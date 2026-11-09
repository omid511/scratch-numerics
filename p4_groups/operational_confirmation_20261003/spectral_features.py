"""Three measured-response descriptors; no modal or reliability interpretation."""
import numpy as np

SAMPLE_COUNT = 128
SAMPLE_RATE = 1000.0
BINS = np.arange(1, 64)
FREQUENCIES = BINS * SAMPLE_RATE / SAMPLE_COUNT
# Periodic Hann, not the symmetric finite-interval window.
HANN = .5 - .5 * np.cos(2 * np.pi * np.arange(SAMPLE_COUNT) / SAMPLE_COUNT)
NAMES = ('summed_power_entropy', 'fitted_frequency_neighborhood_mass',
         'broadband_spatial_concentration')


def descriptors(signals, modal_frequencies):
    """Return values, validity and chosen bins; undefined values remain NaN.

    Frequencies must already be the existing top-three fitted-strength modes.
    A nearest-bin tie selects the lower bin. No input gate is changed here.
    """
    x = np.asarray(signals, dtype=np.float64)
    f = np.asarray(modal_frequencies, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != SAMPLE_COUNT or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite real channels by128 measured samples')
    if f.ndim != 1 or len(f) > 3:
        raise ValueError('Expected at most three existing fitted frequencies')
    spectrum = np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * HANN, axis=1)[:, 1:64]
    power = np.sum(abs(spectrum) ** 2, axis=0)
    total = float(power.sum())
    values = np.full(3, np.nan)
    valid = np.zeros(3, dtype=bool)
    selected_bins = np.full(3, -1, dtype=np.int16)
    if total <= 0:
        return values, valid, selected_bins
    if not np.isfinite(total):
        raise FloatingPointError('Nonfinite measured spectral power')
    p = power / total
    positive = p > 0
    values[0] = -np.sum(p[positive] * np.log(p[positive])) / np.log(len(p))
    valid[0] = True
    union = np.zeros(len(BINS), dtype=bool)
    for i, frequency in enumerate(f):
        if not np.isfinite(frequency) or not 0 < frequency < SAMPLE_RATE / 2:
            continue
        center = int(np.argmin(abs(FREQUENCIES - frequency)))
        selected_bins[i] = BINS[center]
        union[max(0, center - 1):min(len(BINS), center + 2)] = True
    if union.any():
        values[1] = p[union].sum()
        valid[1] = True
    spatial = spectrum @ spectrum.conj().T
    values[2] = np.sum(abs(spatial) ** 2) / total ** 2
    valid[2] = True
    if not np.isfinite(values[valid]).all() or np.any(values[valid] < -1e-12) or np.any(values[valid] > 1 + 1e-12):
        raise FloatingPointError('Spectral descriptors outside mathematical bounds')
    return values, valid, selected_bins


def smoke():
    t = np.arange(SAMPLE_COUNT) / SAMPLE_RATE
    tone = np.cos(2 * np.pi * FREQUENCIES[9] * t)
    channels = np.stack((tone, -tone))
    a, valid, bins = descriptors(channels, [FREQUENCIES[9]])
    b, same, _ = descriptors(7.3 * channels, [FREQUENCIES[9]])
    assert np.array_equal(valid, same) and np.allclose(a, b, atol=1e-14)
    assert abs(np.fft.rfft(channels.mean(axis=0))[10]) < 1e-12
    assert np.isclose(a[1], 1) and np.isclose(a[2], 1)
    missing, mask, _ = descriptors(channels, [])
    assert np.array_equal(mask, [True, False, True]) and np.isnan(missing[1])
    empty, mask, _ = descriptors(np.zeros_like(channels), [FREQUENCIES[9]])
    assert not mask.any() and np.isnan(empty).all()
    _, _, tied = descriptors(channels, [(FREQUENCIES[9] + FREQUENCIES[10]) / 2])
    assert tied[0] == 10
    _, _, edges = descriptors(channels, [1., 499.])
    assert np.array_equal(edges[:2], [1, 63])
    independent = np.stack((tone, np.cos(2 * np.pi * FREQUENCIES[25] * t)))
    value, _, _ = descriptors(independent, [FREQUENCIES[9], FREQUENCIES[25]])
    assert np.isclose(value[2], .5)
    return dict(antiphase_retained=True, amplitude_invariance=True,
                undefined_values_masked=True, bin_ties_and_edges=True,
                orthogonal_spectral_directions=True)


if __name__ == '__main__':
    import json
    print('SPECTRAL_FEATURE_SMOKE', json.dumps(smoke()), flush=True)
