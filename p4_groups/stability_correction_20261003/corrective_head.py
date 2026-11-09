"""Convex affine correction with explicit TRAIN-validation proxy constraints."""
import time
import numpy as np
from scipy.optimize import minimize
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling

UNIT = .01
REGULARIZATION = .1


def transform(head, x):
    z = (x - head['mean']) / head['scale']
    z[:, ~head['active']] = 0.
    return np.column_stack((z, np.ones(len(z))))


def correction(head, x):
    return UNIT * (transform(head, x) @ head['coef'])


def functions(a, base, truth, validation_a, validation_base, validation_truth):
    error0 = (base - truth) / UNIT
    near = abs(truth) < .05
    if not near.any():
        raise ValueError('empty boundary objective')
    n = len(validation_truth) // 4
    np.testing.assert_array_equal(validation_truth.reshape(4, n), np.tile(validation_truth[:n], (4, 1)))
    far = validation_truth[:n] >= .15
    if not far.any():
        raise ValueError('empty far-stable constraint cohort')
    v = validation_a.reshape(4, n, -1)
    ve0 = ((validation_base - validation_truth) / UNIT).reshape(4, n)

    def objective(coef):
        e = error0 + a @ coef
        positive = np.maximum(e[near], 0.)
        loss = .5*np.mean(e*e) + .5*np.mean(positive*positive)
        loss += .5*REGULARIZATION*np.dot(coef[:-1], coef[:-1])
        grad = a.T @ e / len(e) + a[near].T @ positive / near.sum()
        grad[:-1] += REGULARIZATION*coef[:-1]
        return float(loss), grad

    def metrics(coef):
        e = ve0 + np.einsum('vnd,d->vn', v, coef)
        mse = np.mean(e*e, axis=1)
        mse_grad = 2*np.einsum('vnd,vn->vd', v, e) / n
        negative = np.maximum(-e[:, far] - .05/UNIT, 0.)
        under = np.mean(negative*negative, axis=1)
        under_grad = -2*np.einsum('vnd,vn->vd', v[:, far], negative) / far.sum()
        smooth = np.sqrt(e[0]*e[0] + (1e-6/UNIT)**2)
        clean = float(smooth.mean())
        clean_grad = v[0].T @ (e[0]/smooth) / n
        return np.r_[mse, under, clean], np.vstack((mse_grad, under_grad, clean_grad))

    reference, _ = metrics(np.zeros(a.shape[1]))
    normalization = np.maximum(reference, 1e-4)

    def constraint(coef):
        values, gradients = metrics(coef)
        return (reference - values)/normalization, -gradients/normalization[:, None]

    return objective, constraint, reference


def fit_head(x, base, truth, vx, vbase, vtruth, constrained):
    mean, scale, active = fit_ridge_scaling(x)
    head = dict(mean=mean, scale=scale, active=active)
    a, va = transform(head, x), transform(head, vx)
    objective, constraint, reference = functions(a, base, truth, va, vbase, vtruth)
    initial = np.zeros(a.shape[1])
    np.testing.assert_array_equal(constraint(initial)[0], np.zeros(9))
    start = time.perf_counter()
    if constrained:
        result = minimize(objective, initial, jac=True, method='SLSQP',
            constraints=[dict(type='ineq', fun=lambda w: constraint(w)[0], jac=lambda w: constraint(w)[1])],
            options=dict(maxiter=2000, ftol=1e-10))
    else:
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          options=dict(maxiter=2000, ftol=1e-14, gtol=1e-9))
    head['coef'] = result.x
    slacks = constraint(result.x)[0]
    metadata = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit),
        loss=float(result.fun), seconds=time.perf_counter()-start, constrained=bool(constrained),
        constraint_reference_normalized_metrics=reference.tolist(), constraint_slacks=slacks.tolist(),
        active_columns=int(active.sum()), correction_coefficient_norm=float(np.linalg.norm(result.x)))
    if not result.success or not np.isfinite(result.x).all() or (constrained and slacks.min() < -1e-7):
        raise RuntimeError(f'corrective optimizer/feasibility failed: {metadata}')
    return head, metadata


def gradient_check(x, base, truth, vx, vbase, vtruth):
    mean, scale, active = fit_ridge_scaling(x)
    head = dict(mean=mean, scale=scale, active=active)
    a, va = transform(head, x), transform(head, vx)
    objective, constraint, _ = functions(a, base, truth, va, vbase, vtruth)
    w = np.random.default_rng(17).normal(size=a.shape[1]) * .002
    _, analytic = objective(w)
    _, cj = constraint(w)
    numeric, nj = np.empty_like(w), np.empty_like(cj)
    for j in range(len(w)):
        shift = np.zeros_like(w); shift[j] = 1e-6
        numeric[j] = (objective(w+shift)[0] - objective(w-shift)[0])/(2e-6)
        nj[:, j] = (constraint(w+shift)[0] - constraint(w-shift)[0])/(2e-6)
    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(cj, nj, atol=2e-5, rtol=2e-5)
    return dict(objective_gradient_max_error=float(abs(analytic-numeric).max()),
                constraint_gradient_max_error=float(abs(cj-nj).max()), zero_correction_feasible=True)
