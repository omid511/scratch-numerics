"""Waveform-only multichannel LS-ESPRIT features for the P4 ridge probe.

Shared poles are estimated from a temporal Hankel subspace pooled across
physical sensors, not from a spatially averaged signal. Use the contiguous
prefix before the first sensor-cap hit: clipped oscillations are not linear
modal responses. Model order (at most eight oscillatory pairs plus DC), lag
length and numerical threshold are fixed, not tuned against margin labels.

This is an identification probe, not an oracle: growth clamping, noise, weak
excitation and close modes can bias or hide poles. A missing estimate has an
explicit availability flag and mode count, not an invented physical estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

FEATURE_NAMES = tuple(
    f"mode{i}_{name}" for i in range(1, 4)
    for name in ("alpha_per_s", "frequency_hz", "energy_fraction")
) + ("max_visible_alpha_per_s", "energy_weighted_alpha_per_s",
     "mode_count", "relative_fit_error", "usable_fraction", "available")
QUALITY_INDICES = (11, 12, 13, 14)


def estimate_modes(signals, dt, *, cap=50.0, max_modes=8, lag=64, include_shapes=False):
    """Return poles and fitted modal energy from raw physical channels.

    Pole order is descending fitted component energy. Energies are not an
    orthogonal decomposition: interference may make their sum exceed signal
    energy. They rank observability here, not mechanical modal energy.
    Invalid acquisition inputs raise; unidentifiable signals return no modes.
    With include_shapes, also return per-mode complex sensor coefficients
    (cosine minus i*sine), in the same energy order. Their global complex
    scale is arbitrary; they describe the supplied, possibly calibrated,
    channels rather than recovering erased physical amplitude ratios.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError("expected finite physical channels with at least eight samples")
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("modal estimation requires a positive physical dt")
    if not np.isfinite(cap) or cap <= 0 or max_modes < 1 or lag < 4:
        raise ValueError("invalid modal estimator settings")
    n_total = x.shape[1]
    hits = np.flatnonzero(np.any(np.abs(x) >= cap, axis=0))
    n = int(hits[0]) if hits.size else n_total
    empty = {"alpha": np.empty(0), "frequency": np.empty(0),
             "energy_fraction": np.empty(0), "relative_fit_error": 1.0,
             "usable_fraction": n / n_total}
    if include_shapes:
        empty["mode_shapes"] = np.empty((0, x.shape[0]), dtype=np.complex128)
    # Need temporal aperture beyond the requested maximum order.
    if n < max(32, 2 * max_modes + 4):
        return empty
    x = x[:, :n]
    scale = np.max(np.abs(x))
    if scale == 0:
        return empty
    x = x / scale
    width = min(lag, n // 2)
    windows = np.lib.stride_tricks.sliding_window_view(x, width, axis=1)
    h = windows.reshape(-1, width)
    covariance = h.T @ h
    eigenvalues, u = eigh(covariance, check_finite=False)
    if eigenvalues[-1] <= 0:
        return empty
    # 1e-6 relative singular-value threshold, squared for the covariance.
    rank = min(2 * max_modes + 1, width - 1,
               int(np.count_nonzero(eigenvalues > eigenvalues[-1] * 1e-12)))
    if rank < 2:
        return empty
    subspace = u[:, -rank:]
    shift = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=None)[0]
    poles = np.linalg.eigvals(shift)
    # One representative of each oscillatory conjugate pair; omit DC/real roots.
    poles = poles[(poles.imag > 1e-8) & (np.abs(poles) > 1e-12)]
    if not len(poles):
        return empty
    alpha = np.log(np.abs(poles)) / dt
    frequency = np.angle(poles) / (2 * np.pi * dt)
    t = np.arange(n) * dt
    growth = alpha[:, None] * t
    # Column scaling avoids overflow without changing fitted modal trajectories.
    envelope = np.exp(growth - growth.max(axis=1, keepdims=True))
    phase = 2 * np.pi * frequency[:, None] * t
    basis = np.column_stack([np.ones(n),
                             (envelope * np.cos(phase)).T,
                             (envelope * np.sin(phase)).T])
    coefficients = np.linalg.lstsq(basis, x.T, rcond=None)[0]
    reconstructed = basis @ coefficients
    centered_energy = np.sum((x - x.mean(axis=1, keepdims=True)) ** 2)
    if centered_energy <= np.finfo(float).eps * np.sum(x ** 2):
        return empty
    relative_error = float(np.sum((x.T - reconstructed) ** 2) / centered_energy)
    count = len(poles)
    energies = np.array([
        np.sum((basis[:, 1 + i, None] * coefficients[1 + i]
                + basis[:, 1 + count + i, None] * coefficients[1 + count + i]) ** 2)
        for i in range(count)
    ])
    if energies.sum() <= 0 or not np.isfinite(energies).all():
        return empty
    order = np.argsort(-energies, kind="stable")
    result = {"alpha": alpha[order], "frequency": frequency[order],
              "energy_fraction": energies[order] / energies.sum(),
              "relative_fit_error": relative_error, "usable_fraction": n / n_total}
    if include_shapes:
        shapes = np.empty((count, x.shape[0]), dtype=np.complex128)
        np.take(coefficients[1:1 + count], order, axis=0, out=shapes.real)
        np.take(coefficients[1 + count:], order, axis=0, out=shapes.imag)
        shapes.imag *= -1
        result["mode_shapes"] = shapes
    return result


def extract_modal_features(signals, dt):
    return modal_feature_row(estimate_modes(signals, dt))


def modal_feature_row(modes):
    """Encode an existing estimate without repeating modal identification."""
    row = np.zeros(len(FEATURE_NAMES))
    row[12] = modes["relative_fit_error"]
    row[13] = modes["usable_fraction"]
    count = len(modes["alpha"])
    if count:
        for i in range(min(3, count)):
            row[3 * i:3 * i + 3] = (modes["alpha"][i], modes["frequency"][i],
                                     modes["energy_fraction"][i])
        visible = modes["energy_fraction"] >= 0.01
        row[9] = np.max(modes["alpha"][visible])
        row[10] = modes["alpha"] @ modes["energy_fraction"]
        row[11] = count
        row[14] = 1.0
    if not np.isfinite(row).all():
        raise FloatingPointError("nonfinite modal features")
    return row


PHYSICS_FEATURE_NAMES = {
    "damping": tuple(f"energy_mode{i}_damping_ratio" for i in range(1, 4))
    + ("minimum_visible_damping_ratio", "energy_weighted_damping_ratio"),
    "stability": tuple(
        f"least_stable_mode{i}_{name}" for i in range(1, 4)
        for name in ("alpha_per_s", "frequency_hz", "energy_fraction")
    ) + ("visible_mode_count",),
    "spacing": ("least_stable_neighbor_relative_frequency_gap",
                "minimum_visible_relative_frequency_gap",
                "least_stable_neighbor_relative_pole_distance", "neighbor_present"),
}


def physics_feature_groups(modes):
    """Analytical features from estimated poles, never from solver truth.

    Damping ratio is -alpha/hypot(alpha, omega), negative for growth.
    'Least stable' means largest estimated growth rate among modes with at
    least 1% fitted component energy, not oracle identification of the
    flutter-driving mode. Frequency gaps are instantaneous; they do not
    measure the derivative of eigenvalues with respect to velocity.
    """
    groups = {name: np.zeros(len(names)) for name, names in PHYSICS_FEATURE_NAMES.items()}
    alpha = np.asarray(modes["alpha"])
    frequency = np.asarray(modes["frequency"])
    energy = np.asarray(modes["energy_fraction"])
    if not len(alpha):
        return groups
    omega = 2 * np.pi * frequency
    natural = np.hypot(alpha, omega)
    damping = -alpha / natural
    n = min(3, len(alpha))
    groups["damping"][:n] = damping[:n]
    groups["damping"][4] = damping @ energy
    visible = np.flatnonzero(energy >= .01)
    if not len(visible):
        return groups
    groups["damping"][3] = np.min(damping[visible])
    order = visible[np.lexsort((frequency[visible], -alpha[visible]))]
    groups["stability"][-1] = len(order)
    for i, index in enumerate(order[:3]):
        groups["stability"][3*i:3*i+3] = (alpha[index], frequency[index], energy[index])
    if len(order) > 1:
        lead = order[0]
        others = order[1:]
        neighbor = others[np.argmin(np.abs(frequency[others] - frequency[lead]))]
        f = frequency[visible]
        i, j = np.triu_indices(len(f), k=1)
        pole_distance = np.hypot(alpha[lead] - alpha[neighbor], omega[lead] - omega[neighbor])
        groups["spacing"][:] = (
            abs(frequency[lead] - frequency[neighbor]) / frequency[lead],
            np.min(2 * np.abs(f[i] - f[j]) / (f[i] + f[j])),
            pole_distance / natural[lead], 1.0,
        )
    if any(not np.isfinite(row).all() for row in groups.values()):
        raise FloatingPointError("nonfinite analytical modal features")
    return groups


def _spatial_inputs(modes, sensor_pairs):
    """Validate channel alignment and order visible modes by stability."""
    if "mode_shapes" not in modes:
        raise ValueError("spatial features require shape-enabled modal extraction")
    shapes = np.asarray(modes["mode_shapes"])
    pairs = np.asarray(sensor_pairs)
    if shapes.ndim != 2 or not shapes.shape[1] or shapes.shape[0] != len(modes["alpha"]):
        raise ValueError("mode shapes must align with estimated modes and sensor channels")
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs >= shapes.shape[1])):
        raise ValueError("sensor pairs must contain valid integer channel indices")
    visible = np.flatnonzero(np.asarray(modes["energy_fraction"]) >= .01)
    alpha, frequency = np.asarray(modes["alpha"]), np.asarray(modes["frequency"])
    order = visible[np.lexsort((frequency[visible], -alpha[visible]))]
    return shapes, pairs, order


def _fill_spatial_pattern(row, shape, pairs):
    """Fill one pattern and return its scaled shape/power for pair similarity."""
    scale = np.max(np.abs(shape))
    if not np.isfinite(scale):
        raise FloatingPointError("nonfinite modal sensor shape")
    if scale == 0:
        return None
    shape = shape / scale
    power = shape.real ** 2 + shape.imag ** 2
    total = power.sum()
    cross = shape[pairs[:, 0]] * shape[pairs[:, 1]].conj() / total
    channels = len(shape)
    row[:channels] = power / total
    row[channels:channels + len(pairs)] = cross.real
    row[channels + len(pairs):-1] = cross.imag
    row[-1] = 1.
    if not np.isfinite(row).all():
        raise FloatingPointError("nonfinite spatial modal features")
    return shape, total


def modal_spatial_row(modes, sensor_pairs):
    """Unit-trace complex Gramian of the least-stable >=1%-energy mode.

    Fixed channel powers and real/imaginary pair entries preserve spatial
    phase without global excitation scale/phase or angle-wrap artifacts.
    Missing estimates return zeros with availability zero.
    """
    shapes, pairs, order = _spatial_inputs(modes, sensor_pairs)
    row = np.zeros(shapes.shape[1] + 2 * len(pairs) + 1)
    if len(order):
        _fill_spatial_pattern(row, shapes[order[0]], pairs)
    return row


def modal_spatial_addition_row(modes, sensor_pairs):
    """Second least-stable visible pattern plus gauge-invariant mode similarity.

    Append to modal_spatial_row: another pattern/availability, squared normalized
    complex inner product, and pair availability. Similarity is spatial
    collinearity of calibrated channels, not a physical coupling measurement.
    Each mode may have independent global excitation scale/phase.
    """
    shapes, pairs, order = _spatial_inputs(modes, sensor_pairs)
    size = shapes.shape[1] + 2 * len(pairs) + 1
    row = np.zeros(size + 2)
    if len(order) < 2:
        return row
    second = _fill_spatial_pattern(row[:size], shapes[order[1]], pairs)
    if second is None:
        return row
    lead = shapes[order[0]]
    scale = np.max(np.abs(lead))
    if not np.isfinite(scale):
        raise FloatingPointError("nonfinite modal sensor shape")
    if scale > 0:
        lead = lead / scale
        shape, total = second
        row[-2] = abs(np.vdot(lead, shape)) ** 2 / (np.vdot(lead, lead).real * total)
        row[-1] = 1.
    if not np.isfinite(row).all():
        raise FloatingPointError("nonfinite spatial modal similarity")
    return row
