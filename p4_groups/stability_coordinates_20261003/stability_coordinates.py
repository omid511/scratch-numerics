"""Dimensionless modal-pair coordinates; no labels or solver inputs."""
import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))


def pair_coordinates(alpha, omega, available, kind):
    alpha, omega = np.asarray(alpha), np.asarray(omega)
    available = np.asarray(available, dtype=bool)
    if alpha.shape != omega.shape or alpha.shape[-1] != 2 or available.shape != alpha.shape[:-1]:
        raise ValueError('expected aligned pairs and availability')
    if not np.isfinite(alpha).all() or not np.isfinite(omega).all():
        raise ValueError('nonfinite measured poles')
    if kind not in ('hurwitz', 'products'):
        raise ValueError('unknown coordinate representation')
    valid = available & np.all(omega > 0, axis=-1)
    # A common positive scale preserves roots' half-plane and swap symmetry.
    scale = np.sqrt(np.mean(alpha * alpha + omega * omega, axis=-1))
    scale = np.where(valid, scale, 1.)
    a, w = alpha / scale[..., None], omega / scale[..., None]
    b, d = -2 * a[..., 0], -2 * a[..., 1]
    c, e = a[..., 0] ** 2 + w[..., 0] ** 2, a[..., 1] ** 2 + w[..., 1] ** 2
    if kind == 'hurwitz':
        c1 = b + d
        c2 = c + e + b * d
        c3 = b * e + d * c
        c4 = c * e
        delta2 = c1 * c2 - c3
        delta3 = delta2 * c3 - c1 * c1 * c4
        values = (c1, c2, c3, c4, delta2, delta3)
    else:
        beta1, beta2 = -a[..., 0], -a[..., 1]
        values = (beta1 + beta2, beta1 * beta2, c + e, c * e,
                  (beta1 - beta2) ** 2, (c - e) ** 2)
    result = np.stack((*values, valid.astype(float)), axis=-1)
    result = np.where(valid[..., None], result, 0.)
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite stability coordinates')
    return result


def additions(features, kind):
    q = np.asarray(features)
    if q.shape[-1] != 174 or not np.isfinite(q).all():
        raise ValueError('expected finite measured174-feature rows')
    # Existing stability triplets: alpha, frequency in Hz, fitted energy.
    poles = q[..., 27:36].reshape(*q.shape[:-1], 3, 3)
    rows = []
    for first, second in PAIRS:
        pair = poles[..., [first, second], :]
        valid = (q[..., 36] > second) & np.all(pair[..., 2] >= .01, axis=-1)
        rows.append(pair_coordinates(pair[..., 0], 2 * np.pi * pair[..., 1], valid, kind))
    return np.concatenate(rows, axis=-1)


def augmented(features, kind):
    return np.concatenate((features, additions(features, kind)), axis=-1)


def numerical_check():
    alpha = np.array([[-.1, -.2], [0., -.2], [.1, -.2], [-.1, .1], [-.02, -.03]])
    omega = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 1.], [1., 1.001]])
    result = pair_coordinates(alpha, omega, np.ones(5, bool), 'hurwitz')
    assert result[0, 5] > 0 and abs(result[1, 5]) < 1e-14 and result[2, 5] < 0
    # Opposite growth can yield Delta3=0 despite instability: never classify by it alone.
    assert result[3, 0] == 0 and result[3, 5] == 0 and alpha[3].max() > 0
    for kind in ('hurwitz', 'products'):
        row = pair_coordinates(alpha, omega, np.ones(5, bool), kind)
        np.testing.assert_allclose(pair_coordinates(alpha * 137, omega * 137, np.ones(5, bool), kind), row, atol=1e-14)
        np.testing.assert_allclose(pair_coordinates(alpha[:, ::-1], omega[:, ::-1], np.ones(5, bool), kind), row, atol=1e-14)
        np.testing.assert_array_equal(pair_coordinates(alpha, omega, np.zeros(5, bool), kind), np.zeros((5, 7)))
    scales = np.sqrt(np.mean(alpha ** 2 + omega ** 2, axis=1))
    max_polynomial_error = 0.
    for k in range(len(alpha)):
        poles = ((alpha[k, :, None] + 1j * omega[k, :, None] * np.array([1., -1.])) / scales[k]).reshape(-1)
        expected = np.poly(poles).real[1:]
        max_polynomial_error = max(max_polynomial_error, float(abs(result[k, :4] - expected).max()))
        np.testing.assert_allclose(result[k, :4], expected, atol=1e-14)
        if abs(alpha[k].sum()) > 1e-12:
            beta = -alpha[k] / scales[k]
            w = omega[k] / scales[k]
            b1, b2 = beta
            w1, w2 = w
            # Chajjed et al. Eq1, independently evaluated away from its singularity.
            flutter = ((w2*w2-w1*w1+b2*b2-b1*b1)/2) ** 2
            flutter += 4*b1*b2*((w2*w2+w1*w1)/2 + (b1+b2)**2/2)
            flutter -= ((b2-b1)/(b2+b1)*(w2*w2-w1*w1)/2 + (b1+b2)**2/2) ** 2
            np.testing.assert_allclose(result[k, 5] / result[k, 0] ** 2, flutter, atol=1e-14)
    return dict(stable_neutral_unstable=True, swap_and_scale_invariance=True,
                zero_damping_sum_finite=True, missing_pairs_explicit=True,
                published_flutter_formula=True, max_polynomial_error=max_polynomial_error)
