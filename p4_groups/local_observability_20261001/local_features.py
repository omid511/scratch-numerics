"""Measured-band observability probe; no labels, solver modes or target speed."""
import numpy as np
from scipy.signal import fftconvolve

GLOBAL_NAMES = ['usable_fraction', 'usable_cycles', 'available', 'valid_center_fraction',
                'integrated_band_share', 'band_log_power_slope_over_4pi_f']
LOCAL_NAMES = ['band_share_p10', 'band_share_p90', 'late_minus_early_band_share',
               'fraction_above_half_mean_band_share', 'band_power_effective_time_fraction',
               'band_log_power_line_rmse']
FREQUENCY_COLUMNS = [28, 31]


def band_features(signal, dt, frequency):
    """Full-support complex Gabor statistics before the first selected-channel cap.

    Candidate frequency must be measured, not an oracle mode. Unavailable bands
    have an explicit flag; aperture quantities remain meaningful. Nothing beyond
    the observed prefix is padded or extrapolated. Returned local quantities
    describe observation quality, not calibrated confidence or mechanical energy.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('expected finite channels x observed samples')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(frequency) or frequency < 0:
        raise ValueError('expected positive physical dt and nonnegative measured frequency')
    hits = np.flatnonzero(np.any(np.abs(x) >= 50., axis=0))
    n = int(hits[0]) if len(hits) else x.shape[1]
    control, local = np.zeros(6), np.zeros(6)
    control[:2] = n / x.shape[1], n * dt * frequency
    if not frequency or frequency >= .5 / dt:
        return control, local
    half = int(np.ceil(6 / (frequency * dt)))
    centers = n - 2 * half
    if centers < 8:
        return control, local
    x = x[:, :n]
    amplitude = np.max(np.abs(x))
    if amplitude == 0:
        return control, local
    x = x / amplitude
    time = np.arange(-half, half + 1) * dt
    sigma = 2 / frequency
    envelope = np.exp(-.5 * (time / sigma) ** 2)
    envelope /= envelope.sum()
    kernel = envelope * np.exp(2j * np.pi * frequency * time)
    coefficients = fftconvolve(x, kernel[None, :], mode='valid', axes=1)
    band = (coefficients.real ** 2 + coefficients.imag ** 2).sum(axis=0)
    total = fftconvolve((x * x).sum(axis=0), envelope, mode='valid')
    if total.max() <= 0 or band.max() <= 0:
        return control, local
    total = np.maximum(total, np.finfo(float).eps * total.max())
    share = band / total
    if not np.isfinite(share).all() or share.max() > 1 + 1e-7:
        raise FloatingPointError('invalid local band-to-total power ratio')
    share = np.clip(share, 0, 1)
    center_time = np.arange(centers) * dt
    center_time -= center_time.mean()
    log_power = np.log(np.maximum(band, 1e-12 * band.max()))
    slope = center_time @ (log_power - log_power.mean()) / (center_time @ center_time)
    residual = log_power - log_power.mean() - slope * center_time
    middle = centers // 2
    control[2:] = 1., centers / n, band.sum() / total.sum(), slope / (4 * np.pi * frequency)
    local[:2] = np.quantile(share, [.1, .9])
    local[2:] = (share[middle:].mean() - share[:middle].mean(),
                 np.mean(share >= .5 * share.mean()),
                 band.sum() ** 2 / (centers * (band @ band)),
                 np.sqrt(np.mean(residual ** 2)))
    if not np.isfinite(control).all() or not np.isfinite(local).all():
        raise FloatingPointError('nonfinite band descriptors')
    return control, local


def feature_row(signal, dt, frequencies):
    if len(frequencies) != 2:
        raise ValueError('expected two measured candidate-band frequencies')
    parts = [band_features(signal, dt, float(f)) for f in frequencies]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def numerical_smoke():
    dt, f = .001, 70.
    t = np.arange(512) * dt
    tone = np.stack([np.sin(2 * np.pi * f * t), .7 * np.cos(2 * np.pi * f * t)])
    control, steady = band_features(tone, dt, f)
    assert control[2] == 1 and abs(control[4] - .5) < .01
    assert steady[5] < .01 and steady[4] > .99
    for args in [(10 * tone, dt, f), (tone, 2 * dt, f / 2)]:
        c, q = band_features(*args)
        np.testing.assert_allclose(c, control, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(q, steady, rtol=1e-9, atol=1e-9)
    burst = tone * np.exp(-.5 * ((t - .26) / .035) ** 2)
    _, intermittent = band_features(burst, dt, f)
    assert intermittent[4] < steady[4] - .3 and intermittent[5] > steady[5] + .1
    capped = tone.copy()
    capped[:, 300:] = 50
    changed_tail = capped.copy()
    changed_tail[:, 301:] = -20
    for a, b in zip(band_features(capped, dt, f), band_features(changed_tail, dt, f)):
        np.testing.assert_array_equal(a, b)
    for x, frequency in [(tone[:, :96], f), (np.zeros_like(tone), f), (tone, 0.)]:
        c, q = band_features(x, dt, frequency)
        assert c[2] == 0 and np.array_equal(q, np.zeros(6))
    print('BAND_SMOKE_PASSED: steady/burst observability, amplitude/time-unit invariance, cap boundary and unavailable bands', flush=True)
