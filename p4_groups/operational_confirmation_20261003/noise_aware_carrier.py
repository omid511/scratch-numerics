"""Experimental waveform-only curvature selection; not calibrated confidence."""
import time
import numpy as np
from evaluate_carrier_curvature import fit_curved
from time_varying_modal_fit import fit_single_mode, _failure

NAMES = ('alpha_mid', 'alpha_slope', 'alpha_curvature',
         'frequency_mid', 'frequency_slope', 'frequency_curvature')


def _interior(fit):
    return (fit.get('success') is True and not fit.get('bound_hits')
            and np.isfinite((fit.get('alpha_end', np.nan),
                             fit.get('frequency_end', np.nan))).all())


def _residual(x, u, theta):
    am, ad, ac, fm, fd, fc = theta
    envelope = np.exp(am*u + .5*ad*u**2 + ac*u**3/6)
    phase = 2*np.pi*(fm*u + .5*fd*u**2 + fc*u**3/6)
    basis = np.column_stack((np.ones_like(u), envelope*np.cos(phase), envelope*np.sin(phase)))
    if not np.isfinite(basis).all():
        raise ValueError('Nonfinite carrier basis')
    coefficients, _, rank, _ = np.linalg.lstsq(basis, x.T, rcond=None)
    if rank != 3:
        raise ValueError('Rank-deficient channel-coefficient basis')
    residual = (basis @ coefficients - x.T).T
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite profiled waveform residual')
    return residual


def select_carrier(signal, dt, linear, curved):
    """Choose only between fits of this same waveform; no labels or metadata."""
    started = time.perf_counter()
    raw = np.asarray(signal)
    if np.iscomplexobj(raw):
        raise ValueError('Real calibrated measurements required')
    x = np.asarray(raw, dtype=float)
    dt = float(dt)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 4:
        raise ValueError('Channels-by-samples measurements required')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Finite positive physical dt required')
    selection = dict(chosen=None, status=None, statistic=None, threshold=2*np.log(x.size))

    def finish(fit, chosen, status):
        selection.update(chosen=chosen, status=status)
        return dict(fit=fit, selection=selection, elapsed_seconds=time.perf_counter()-started)

    if not np.isfinite(x).all() or np.any(abs(x) >= 50):
        failed = _failure('Nonfinite or cap50 measurements', 'invalid_measurements', started)
        return finish(failed, None, 'invalid_measurements')
    if not np.isfinite(np.sum((x-x.mean(axis=1, keepdims=True))**2)) or not np.any(x-x.mean(axis=1, keepdims=True)):
        failed = _failure('Nonzero finite centered energy required', 'invalid_energy', started)
        return finish(failed, None, 'invalid_energy')
    if not _interior(linear):
        return finish(linear, None, 'linear_unavailable')
    if not _interior(curved):
        return finish(linear, 'linear', 'curved_unavailable')

    duration = (x.shape[1]-1)*dt
    powers = np.array([duration, duration**2, duration**3]*2)
    theta = np.array([curved['parameters'][name] for name in NAMES])*powers
    u = (np.arange(x.shape[1])-.5*(x.shape[1]-1))*dt/duration
    try:
        residual = _residual(x, u, theta)
        steps = np.cbrt(np.finfo(float).eps)*np.maximum(1., abs(theta))
        jac = np.empty((x.size, 6))
        for j, step in enumerate(steps):
            delta = np.zeros(6)
            delta[j] = step
            jac[:, j] = ((_residual(x, u, theta+delta)-_residual(x, u, theta-delta))/(2*step)).ravel()
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
        rank = int(np.sum(singular > singular[0]*1e-10))
        selection['jacobian_rank'] = rank
        if rank != 6:
            return finish(linear, 'linear', 'curvature_jacobian_rank_deficient')
        dof = x.shape[1]-3-6/x.shape[0]
        if dof <= 0:
            return finish(linear, 'linear', 'insufficient_residual_degrees_of_freedom')
        variance = np.maximum(np.sum(residual**2, axis=1)/dof,
                              np.finfo(float).eps**2*np.mean(x**2, axis=1))
        inverse = (vt.T/singular**2) @ vt
        meat = jac.T @ (np.repeat(variance, x.shape[1])[:, None]*jac)
        covariance = inverse @ meat @ inverse
        indices = [2, 5]
        curve_cov = covariance[np.ix_(indices, indices)]
        curve_cov = .5*(curve_cov+curve_cov.T)
        eigenvalues = np.linalg.eigvalsh(curve_cov)
        if (not np.isfinite(curve_cov).all() or eigenvalues[0] <= 0
                or eigenvalues[0] <= eigenvalues[-1]*1e-12):
            return finish(linear, 'linear', 'curvature_covariance_unavailable')
        curvature = theta[indices]
        statistic = float(curvature @ np.linalg.solve(curve_cov, curvature))
        if not np.isfinite(statistic) or statistic < 0:
            return finish(linear, 'linear', 'curvature_statistic_unavailable')
        selection.update(statistic=statistic, dimensionless_curvature=curvature.tolist(),
                         curvature_covariance=curve_cov.tolist(), channel_variances=variance.tolist(),
                         residual_dof_per_channel=dof)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
        selection['diagnostic_error'] = str(error)
        return finish(linear, 'linear', 'curvature_diagnostic_failed')
    if statistic > selection['threshold']:
        return finish(curved, 'curved', 'curvature_supported')
    return finish(linear, 'linear', 'curvature_unsupported')


def fit_noise_aware(signal, dt, *, settings):
    """Complete measured-only path; preserve both candidate diagnostics."""
    started = time.perf_counter()
    stationary = fit_single_mode(signal, dt, varying=False, settings=settings)
    linear = fit_single_mode(signal, dt, varying=True, settings=settings, stationary=stationary)
    curved = fit_curved(signal, dt, linear, settings)
    result = select_carrier(signal, dt, linear, curved)
    result.update(stationary_fit=stationary, linear_fit=linear, curved_fit=curved,
                  elapsed_seconds=time.perf_counter()-started)
    return result
