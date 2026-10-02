"""Two-hidden-layer ReLU objective with an optional linear input/output skip."""
import numpy as np

def objective_setup(x, y, hidden, seed, alpha, skip, *,
                    consistency_pairs=None, consistency_weight=0.,
                    consistency_pair_weights=None):
    """Squared loss/L2 plus weight/2 times the weighted mean paired gap²."""
    if not np.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    pairs = None
    if consistency_pairs is not None:
        pairs = np.asarray(consistency_pairs)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.any(pairs < 0) or np.any(pairs >= len(y))):
            raise ValueError("consistency_pairs must be nonempty integer row pairs within training data")
    if consistency_weight and pairs is None:
        raise ValueError("positive consistency_weight requires paired rows")
    pair_weights = None
    if consistency_pair_weights is not None:
        pair_weights = np.asarray(consistency_pair_weights, dtype=float)
        if (pairs is None or pair_weights.shape != (len(pairs),)
                or not np.isfinite(pair_weights).all() or np.any(pair_weights < 0)
                or not np.isfinite(pair_weights.sum()) or pair_weights.sum() <= 0):
            raise ValueError("consistency_pair_weights require finite nonnegative pair weights with positive sum")
        pair_weights = pair_weights / pair_weights.sum()
    rng = np.random.RandomState(seed)
    units = [x.shape[1], *hidden, 1]
    weights, biases = [], []
    for a, b in zip(units[:-1], units[1:]):
        bound = np.sqrt(6 / (a + b))
        weights.append(rng.uniform(-bound, bound, (a, b)))
        biases.append(rng.uniform(-bound, bound, b))
    shapes = [w.shape for w in weights] + [b.shape for b in biases]
    if skip:
        shapes.append((x.shape[1], 1))
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    initial = np.concatenate([v.ravel() for v in weights + biases] + ([np.zeros(x.shape[1])] if skip else []))
    def unpack(theta):
        return [theta[a:b].reshape(s) for a, b, s in zip(offsets[:-1], offsets[1:], shapes)]
    def forward(theta, q):
        p = unpack(theta)
        h1 = np.maximum(q @ p[0] + p[3], 0)
        h2 = np.maximum(h1 @ p[1] + p[4], 0)
        pred = h2 @ p[2] + p[5]
        if skip:
            pred = pred + q @ p[6]
        return pred[:, 0]
    def loss_grad(theta):
        p = unpack(theta)
        h1 = np.maximum(x @ p[0] + p[3], 0)
        h2 = np.maximum(h1 @ p[1] + p[4], 0)
        pred = h2 @ p[2] + p[5]
        if skip:
            pred += x @ p[6]
        residual = pred - y[:, None]
        regularized = p[:3] + ([p[6]] if skip else [])
        loss = (np.sum(residual ** 2) + alpha * sum(np.sum(v ** 2) for v in regularized)) / (2 * len(y))
        if consistency_weight:
            left, right = pairs[:, 0], pairs[:, 1]
            difference = pred[left] - pred[right]
            if pair_weights is None:
                loss += consistency_weight * np.sum(difference ** 2) / (2 * len(pairs))
                contribution = difference * (consistency_weight * len(y) / len(pairs))
            else:
                loss += consistency_weight * np.sum(pair_weights[:, None] * difference ** 2) / 2
                contribution = difference * (consistency_weight * len(y) * pair_weights[:, None])
            # Remaining backprop divides by n_rows; derivatives above rescale pairs.
            np.add.at(residual, left, contribution)
            np.add.at(residual, right, -contribution)
        d2 = (residual @ p[2].T) * (h2 > 0)
        d1 = (d2 @ p[1].T) * (h1 > 0)
        grads = [x.T @ d1 + alpha * p[0], h1.T @ d2 + alpha * p[1],
                 h2.T @ residual + alpha * p[2], d1.sum(0), d2.sum(0), residual.sum(0)]
        if skip:
            grads.append(x.T @ residual + alpha * p[6])
        return loss, np.concatenate([v.ravel() for v in grads]) / len(y)
    return initial, loss_grad, forward
