"""Sim-to-real domain randomization for transient signals.

Perturbations applied per sample:
  - Per-channel gain, bias, and linear gain drift
  - White noise with SNR control
  - Pink (1/f) noise on 25% of noisy clips
  - Common-mode noise across channels
  - Channel dropout and burst dropout
  - Timing skew (±1 sample per channel)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SensorPerturbationConfig:
    snr_db: float | None = None  # None = clean, no noise
    per_channel_gain: bool = True
    per_channel_bias: bool = True
    gain_drift: bool = True
    colored_noise_prob: float = 0.25
    common_mode_fraction: float = 0.3
    channel_drop概率: tuple[int, float] = ((0, 0.6), (1, 0.25), (2, 0.15))
    burst_dropout_prob: float = 0.25
    burst_duration_ms: tuple[float, float] = (10.0, 50.0)
    timing_skew: bool = True


class SensorPerturber:
    """Applies realistic sensor perturbations to multi-channel signals."""

    def __init__(self, config: SensorPerturbationConfig | None = None):
        self.config = config or SensorPerturbationConfig()

    def perturb(
        self, signals: np.ndarray, rng: np.random.Generator, fs: float = 1024.0
    ) -> np.ndarray:
        """Apply realistic sensor perturbations.

        signals: (n_sensors, n_timesteps)
        fs: sampling frequency in Hz
        Returns: perturbed copy (n_sensors, n_timesteps)
        """
        cfg = self.config
        n_sensors, n_t = signals.shape
        out = signals.copy().astype(np.float64)
        rms = np.sqrt(np.mean(out**2)) or 1.0

        # 1. Per-channel gain
        if cfg.per_channel_gain:
            gains = rng.uniform(0.9, 1.1, size=(n_sensors, 1))
            out *= gains

        # 2. Per-channel bias
        if cfg.per_channel_bias:
            biases = rng.uniform(-0.02, 0.02, size=(n_sensors, 1)) * rms
            out += biases

        # 3. Gain drift (linear ramp across clip)
        if cfg.gain_drift:
            drift_factors = rng.uniform(-0.02, 0.02, size=(n_sensors, 1))
            t_norm = np.linspace(0.0, 1.0, n_t).reshape(1, -1)
            out *= 1.0 + drift_factors * t_norm

        # 4. White noise with SNR control
        noise_power = np.zeros_like(out)
        if cfg.snr_db is not None:
            sigma = rms * 10.0 ** (-cfg.snr_db / 20.0)
            noise_power = rng.normal(0, sigma, size=out.shape)

        # 5. Colored (pink / 1/f) noise
        if cfg.snr_db is not None and rng.random() < cfg.colored_noise_prob:
            pink = self._pink_noise(n_t, n_sensors, rng)
            sigma = rms * 10.0 ** (-cfg.snr_db / 20.0)
            pink_rms = np.sqrt(np.mean(pink**2))
            if pink_rms > 0:
                pink *= sigma / pink_rms
            noise_power += pink

        # 6. Common-mode noise
        if cfg.snr_db is not None:
            sigma = rms * 10.0 ** (-cfg.snr_db / 20.0)
            common_pattern = rng.normal(0, sigma, size=(1, n_t))
            weight = rng.uniform(0.0, cfg.common_mode_fraction)
            noise_power += common_pattern * weight

        out += noise_power

        # 7. Channel dropout
        drop_counts = [c for c, _ in cfg.channel_drop概率]
        drop_probs = [p for _, p in cfg.channel_drop概率]
        n_drop = rng.choice(drop_counts, p=drop_probs)
        if n_drop > 0:
            drop_idx = rng.choice(n_sensors, size=min(n_drop, n_sensors), replace=False)
            out[drop_idx] = 0.0

        # 8. Burst dropout (all channels zero for a segment)
        if rng.random() < cfg.burst_dropout_prob:
            burst_len_ms = rng.uniform(*cfg.burst_duration_ms)
            burst_samples = int(burst_len_ms * fs / 1000.0)
            burst_samples = min(burst_samples, n_t)
            if burst_samples > 0:
                start = rng.integers(0, n_t - burst_samples + 1)
                out[:, start : start + burst_samples] = 0.0

        # 9. Timing skew (±1 sample per channel)
        if cfg.timing_skew:
            shifts = rng.integers(-1, 2, size=n_sensors)  # uniform over {-1, 0, 1}
            for s in range(n_sensors):
                if shifts[s] != 0:
                    out[s] = np.roll(out[s], int(shifts[s]))

        return out

    @staticmethod
    def _pink_noise(n_t: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
        """Generate pink (1/f) noise via frequency-domain shaping."""
        freqs = np.fft.rfftfreq(n_t, d=1.0)
        freqs[0] = 1.0  # avoid division by zero
        # Random phases per channel
        phases = rng.uniform(0, 2 * np.pi, size=(n_channels, len(freqs)))
        magnitudes = 1.0 / np.sqrt(freqs)  # 1/f shaping
        spectrum = magnitudes[None, :] * np.exp(1j * phases)
        noise = np.fft.irfft(spectrum, n=n_t)
        return noise


def augment_clip(
    signals: np.ndarray,
    n_augmented: int,
    config: SensorPerturbationConfig,
    base_seed: int,
    fs: float = 1024.0,
) -> list[np.ndarray]:
    """Generate n_augmented perturbed versions of one clip."""
    perturber = SensorPerturber(config)
    return [
        perturber.perturb(signals, np.random.default_rng(base_seed + i), fs)
        for i in range(n_augmented)
    ]
