"""Scale/phase invariant modal comparison; no COMSOL dependency.

Near-repeated eigenvalues are quarantined rather than arbitrarily rotating
eigenvectors into physically misleading individual-mode training labels.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PARAMETERS = ('alpha', 'beta', 'theta_c', 'eta1', 'eta2')


def boundary_mask(x, y):
    """Return the normalized CCCC correction mask on a ``(y, x)`` grid.

    The mask is zero on every domain edge and positive in the interior.  The
    coordinates are normalized from their supplied endpoints so this helper
    works for dimensional or unit-square grids.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for axis, name in ((x, 'x'), (y, 'y')):
        if (axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)):
            raise ValueError(f'Invalid {name} coordinate axis')
    x_unit = (x - x[0]) / (x[-1] - x[0])
    y_unit = (y - y[0]) / (y[-1] - y[0])
    return ((y_unit * (1 - y_unit))[:, None]
            * (x_unit * (1 - x_unit))[None, :])


def transverse_peak_ratios(shapes):
    """Return per-mode raw transverse peaks relative to the run maximum."""
    a = np.asarray(shapes)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Mode shapes must be finite (modes, y, x) arrays')
    peaks = np.max(np.abs(a), axis=(1, 2))
    if np.any(peaks <= 0):
        raise ValueError('Exactly zero transverse mode; cannot score strength')
    return peaks / np.max(peaks)


def normalized(shapes):
    """Unit spatial L2 norm, using max scaling first (safe for tiny eigenvectors)."""
    a = np.asarray(shapes, dtype=complex)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Mode shapes must be finite (modes, y, x) arrays')
    peak = np.max(np.abs(a), axis=(1, 2))
    if np.any(peak == 0):
        raise ValueError('Exactly zero transverse mode; cannot normalize')
    a = a / peak[:, None, None]
    return a / np.linalg.norm(a.reshape(len(a), -1), axis=1)[:, None, None]


def mac_matrix(reference, candidates):
    a = normalized(reference).reshape(len(reference), -1)
    b = normalized(candidates).reshape(len(candidates), -1)
    return np.clip(np.abs(a.conj() @ b.T)**2, 0, 1)


def assign(reference, candidates):
    scores = mac_matrix(reference, candidates)
    if scores.shape[0] > scores.shape[1]:
        raise ValueError('Insufficient candidate modes')
    rows, cols = linear_sum_assignment(-scores)
    mapping = np.empty(len(reference), dtype=int)
    mapping[rows] = cols
    return mapping, scores


def align_real(shape, reference=None, imaginary_tolerance=1e-6):
    a = normalized(np.asarray(shape)[None])[0]
    if reference is None:
        # Establish reference gauge once; subsequent runs align to this reference.
        overlap = a.ravel()[np.argmax(np.abs(a))]
    else:
        b = normalized(np.asarray(reference)[None])[0]
        overlap = np.vdot(b, a)
    if abs(overlap) == 0:
        raise ValueError('No phase reference: orthogonal modes')
    a *= np.exp(-1j * np.angle(overlap))
    if np.linalg.norm(a.imag) > imaginary_tolerance * np.linalg.norm(a.real):
        raise ValueError('Genuinely complex shape unsupported by real P1 field target')
    a = a.real
    return a / np.max(np.abs(a))


def repeated(frequencies, index, relative_gap):
    f = np.asarray(frequencies)
    gaps = np.abs(f - f[index]) / np.maximum(f, f[index])
    gaps[index] = np.inf
    return bool(np.any(gaps <= relative_gap))


def assignment_margin(scores, row, col):
    """Margin over best row OR column competitor, exposing ambiguous assignments."""
    row_other = np.delete(scores[row], col)
    col_other = np.delete(scores[:, col], row)
    runner = max(np.max(row_other, initial=0), np.max(col_other, initial=0))
    return float(scores[row, col] - runner)


def load_run(path, fidelity):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata'].item()))
        x, y = np.asarray(data['x']), np.asarray(data['y'])
        w, f = np.asarray(data['w']), np.asarray(data['frequencies'])
        stored_peak = (np.asarray(data['w_peak_abs'])
                       if 'w_peak_abs' in data.files else None)
        mode_indices = (np.asarray(data['eigen_mode_indices'])
                        if 'eigen_mode_indices' in data.files else None)
        transverse_fraction = (
            np.asarray(data['transverse_fraction'])
            if 'transverse_fraction' in data.files else None)
    if (meta.get('schema_version') != 2 or meta.get('model_version') != 'p1-v2'
            or meta.get('fidelity') != fidelity):
        raise ValueError(f'{path}: incompatible dataset provenance')
    if not isinstance(meta.get('run_id'), int):
        raise ValueError(f'{path}: missing integer run_id')
    if set(PARAMETERS) - set(meta.get('parameters', {})):
        raise ValueError(f'{path}: missing physical parameters')
    if any(not np.isfinite(float(meta['parameters'][p])) for p in PARAMETERS):
        raise ValueError(f'{path}: nonfinite parameters')
    for axis in (x, y):
        if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f'{path}: invalid grid')
        if not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-8, atol=1e-14):
            raise ValueError(f'{path}: only uniform grids supported')
    if w.shape != (len(f), len(y), len(x)) or f.ndim != 1 or not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError(f'{path}: inconsistent shapes/frequencies')
    normalized(w)  # Reject invalid fields, but do not impose an amplitude floor.
    actual_peak = np.max(np.abs(w), axis=(1, 2))
    peak = actual_peak
    if stored_peak is not None:
        if (stored_peak.shape != peak.shape or not np.isfinite(stored_peak).all()
                or np.any(stored_peak <= 0)
                or not np.allclose(stored_peak, actual_peak, rtol=1e-5, atol=0)):
            raise ValueError(f'{path}: invalid raw transverse peak provenance')
        peak = stored_peak
    peak_ratio = peak / np.max(peak)
    if fidelity == 'lf':
        if (mode_indices is None or transverse_fraction is None
                or mode_indices.shape != f.shape
                or transverse_fraction.shape != f.shape
                or not np.isfinite(mode_indices).all()
                or not np.all(mode_indices == np.round(mode_indices))
                or np.any(mode_indices < 1)
                or not np.isfinite(transverse_fraction).all()
                or np.any(transverse_fraction < 0)
                or np.any(transverse_fraction > 1)):
            raise ValueError(f'{path}: LF mode-selection provenance is incomplete')
    elif transverse_fraction is not None:
        if (transverse_fraction.shape != f.shape
                or not np.isfinite(transverse_fraction).all()
                or np.any(transverse_fraction < 0)
                or np.any(transverse_fraction > 1)):
            raise ValueError(f'{path}: invalid HF transverse-energy provenance')
    expected_surface, expected_extraction = ('top', 'comsol-interp') if fidelity == 'hf' else ('midplane', 'ritz')
    if meta.get('surface') != expected_surface or meta.get('extraction') != expected_extraction:
        raise ValueError(f'{path}: unexpected observation surface/extraction')
    # Every bundle must carry provenance sufficient to reject mixed revisions.
    required = ('config_hash', 'source_hash') if fidelity == 'hf' else ('input_hash', 'code_hash')
    if any(not isinstance(meta.get(k), str) or not meta[k] for k in required):
        raise ValueError(f'{path}: incomplete provenance; regenerate with the v2 driver')
    return dict(x=x, y=y, w=w, f=f, w_peak_abs=peak,
                w_peak_ratio=peak_ratio, eigen_mode_indices=mode_indices,
                transverse_fraction=transverse_fraction, meta=meta,
                path=str(Path(path).resolve()))


def check_pair(lf, hf):
    if lf['meta']['run_id'] != hf['meta']['run_id']:
        raise ValueError('Run ID mismatch')
    if any(not np.isclose(lf['meta']['parameters'][p], hf['meta']['parameters'][p], rtol=0, atol=1e-12) for p in PARAMETERS):
        raise ValueError('LF/HF physical parameter mismatch')
    for axis in ('x', 'y'):
        if lf[axis].shape != hf[axis].shape or not np.allclose(lf[axis], hf[axis], rtol=0, atol=1e-12):
            raise ValueError('LF/HF grid mismatch')
