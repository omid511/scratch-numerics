#!/usr/bin/env python3
"""Generate a NEW design table; existing samples are never overwritten.
Run fsdt_mode_shapes.py separately for LF frequencies and shapes.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import qmc
PARAM_RANGES = {'alpha':[.1,.9], 'beta':[.1,1.], 'theta_c':[0.,75.], 'eta1':[.5,3.], 'eta2':[.02,.15]}

def generate_lhs(n_samples, n_params, seed):
    """
    Generate a reproducible scrambled Latin Hypercube sample in [0, 1]^d.

    The SciPy design uses its default ``scramble=True`` and is not centered;
    this matches the frozen ``lhs_samples_v2.csv`` table.
    """
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    return sampler.random(n=n_samples)


def apply_sensitivity_transform(lhs_unit):
    """
    Apply the frozen legacy sensitivity transform to unit LHS samples.

    The transform is intentionally retained for reproducibility.  Its
    piecewise eta mappings have empirical density peaks near eta1=1.75 and
    eta2=0.085 in the configured physical ranges; they are not centered at
    the paper's illustrative values.
    """
    lhs_unit = np.asarray(lhs_unit, dtype=float)
    if lhs_unit.ndim != 2 or lhs_unit.shape[1] != len(PARAM_RANGES):
        raise ValueError('Expected a two-dimensional five-column LHS')
    if not np.isfinite(lhs_unit).all() or np.any((lhs_unit < 0) | (lhs_unit > 1)):
        raise ValueError('Unit LHS values must lie in [0, 1]')
    transformed = np.zeros_like(lhs_unit)

    # alpha: exponent < 1 stretches the upper end.
    transformed[:, 0] = lhs_unit[:, 0] ** 0.6
    # beta: exponent > 1 compresses samples toward the lower end.
    transformed[:, 1] = lhs_unit[:, 1] ** 1.8
    # theta_c: exponent < 1 stretches the upper end.
    transformed[:, 2] = lhs_unit[:, 2] ** 0.5

    # Legacy piecewise maps; the density mode is near the join x=0.5.
    x = lhs_unit[:, 3]
    transformed[:, 3] = np.where(
        x < 0.5, 0.5 * (2 * x) ** 0.5,
        0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 2.0)
    x = lhs_unit[:, 4]
    transformed[:, 4] = np.where(
        x < 0.5, 0.5 * (2 * x) ** 0.7,
        0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 1.5)
    return transformed


def scale_to_physical(lhs_unit):
    """
    Scale LHS samples from [0, 1]^d to physical parameter ranges.
    Applies sensitivity-aware transformation first.
    
    Parameters
    ----------
    lhs_unit : np.ndarray of shape (n_samples, 5)
        Unit LHS samples.
    
    Returns
    -------
    np.ndarray of shape (n_samples, 5)
        Physical parameter values.
    """
    # Apply non-linear transform to concentrate in high-sensitivity regions
    lhs_transformed = apply_sensitivity_transform(lhs_unit)
    
    param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
    physical = np.zeros_like(lhs_transformed)
    for i, name in enumerate(param_names):
        lo, hi = PARAM_RANGES[name]
        physical[:, i] = lo + lhs_transformed[:, i] * (hi - lo)
    return physical



def write_design_table(path, values):
    """Write a new immutable design table with explicit one-based run IDs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PARAM_RANGES):
        raise ValueError('Expected an (n, 5) physical design matrix')
    path = Path(path)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', *PARAM_RANGES])
        writer.writerows((index, *row) for index, row in enumerate(values, 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'lhs_samples_v3.csv')
    parser.add_argument('--distribution', choices=['legacy', 'uniform'], default='legacy',
                        help='legacy preserves the frozen nonlinear transforms')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count must be positive')
    unit = generate_lhs(args.count, len(PARAM_RANGES), args.seed)
    if args.distribution == 'legacy':
        values = scale_to_physical(unit)
    else:
        bounds = np.array(list(PARAM_RANGES.values()))
        values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    write_design_table(args.output, values)
    print(f'Wrote {args.count} designs to {args.output}; use the same table for LF and HF.')


if __name__ == '__main__':
    main()

