"""Gaussian process surrogate with RBF kernel (torch backend).

Implements GP regression with optional gradient enhancement via
finite-difference gradients. Uses torch for kernel matrix operations
(Cholesky, solve) for potential GPU acceleration.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_torch(x: np.ndarray, dtype=torch.float64) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).to(dtype)


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray,
                length_scales: np.ndarray, signal_var: float) -> np.ndarray:
    """RBF (squared-exponential) kernel.

    Args:
        X1: (n1, d) input points.
        X2: (n2, d) input points.
        length_scales: (d,) per-dimension length scales.
        signal_var: signal variance (amplitude^2).

    Returns:
        (n1, n2) kernel matrix.
    """
    X1t = _to_torch(X1) / _to_torch(length_scales)
    X2t = _to_torch(X2) / _to_torch(length_scales)
    sq_dist = (X1t ** 2).sum(1, keepdim=True) + (X2t ** 2).sum(1) - 2 * X1t @ X2t.T
    return (signal_var * torch.exp(-0.5 * sq_dist)).numpy()


def _rbf_kernel_grad(X1: np.ndarray, X2: np.ndarray,
                     length_scales: np.ndarray, signal_var: float) -> np.ndarray:
    """Gradient of RBF kernel w.r.t. X2: dK/dX2_j = K * (X1_i - X2_j) / ls_j^2.

    Returns:
        (n1 * d, n2) matrix. Row index = point * d + dim, col index = point.
    """
    n1, d = X1.shape
    n2 = X2.shape[0]
    K = _rbf_kernel(X1, X2, length_scales, signal_var)
    diff = (X1[:, None, :] - X2[None, :, :]) / (length_scales ** 2)
    K_grad = K[:, :, None] * diff  # (n1, n2, d)
    # Reorder to (n1, d, n2) then reshape to (n1*d, n2)
    return K_grad.transpose(0, 2, 1).reshape(n1 * d, n2)


def _augmented_kernel(X: np.ndarray, length_scales: np.ndarray,
                      signal_var: float, noise_var: float,
                      include_grad: bool = False) -> np.ndarray:
    """Build augmented kernel matrix for function + gradient observations.

    If include_grad=False, returns standard K + noise * I.
    If include_grad=True, returns block matrix [K, K_d; K_d^T, K_dd] + noise * I.
    """
    n, d = X.shape
    K = _rbf_kernel(X, X, length_scales, signal_var)

    if not include_grad:
        return K + noise_var * np.eye(n)

    K_d = _rbf_kernel_grad(X, X, length_scales, signal_var)  # (n*d, n)
    # K_dd: (n*d, n*d) — second derivatives
    # d²K/(dx_i dx'_j) = K * [-(x_i-x'_i)(x_j-x'_j)/(l_i² l_j²) + δ_{ij}/l_i²]
    K_dd = np.zeros((n * d, n * d))
    for a in range(n):
        for b in range(n):
            for i in range(d):
                for j in range(d):
                    val = -K[a, b] * (X[a, i] - X[b, i]) / (length_scales[i] ** 2) \
                              * (X[a, j] - X[b, j]) / (length_scales[j] ** 2)
                    if i == j:
                        val += K[a, b] / (length_scales[i] ** 2)
                    K_dd[a * d + i, b * d + j] = val

    # Build full augmented matrix.
    # K_d[i*d+k, j] = ∂K(x_i, x_j)/∂x_j_k = Cov(f(x_i), ∂f_k(x_j)).
    # Cov(∂f_k(x_i), f(x_j)) = -K_d[i*d+k, j] (antisymmetry of RBF derivatives).
    # Negate both off-diagonal blocks for correct sign and symmetry.
    size = n + n * d
    K_aug = np.zeros((size, size))
    K_aug[:n, :n] = K
    K_aug[:n, n:] = -K_d.T
    K_aug[n:, :n] = -K_d
    K_aug[n:, n:] = K_dd
    K_aug += noise_var * np.eye(size)
    return K_aug


class GPSurrogate:
    """RBF-kernel GP surrogate with optional gradient enhancement.

    Targets are standardized internally (zero mean, unit variance) so the
    default ``signal_var=1.0`` prior stays scale-correct on real-unit
    outputs (e.g. flutter lambda O(1e2)); ``predict`` and
    ``get_training_data`` report original units.

    Args:
        length_scales: (d,) initial length scales. If None, derived
            per-dimension from the training spread (std, floored by the
            input range).
        signal_var: Signal variance.
        noise_var: Noise variance (observation noise).
        use_gradients: Whether to augment training data with finite-diff gradients.
        grad_perturbation: Relative perturbation for finite differences.
    """

    def __init__(
        self,
        length_scales: np.ndarray | None = None,
        signal_var: float = 1.0,
        noise_var: float = 1e-6,
        use_gradients: bool = False,
        grad_perturbation: float = 1e-4,
    ):
        self.length_scales = length_scales
        # True only when scales came from the constructor: refits must keep
        # explicit user scales but recompute the data-driven heuristic.
        # (Setting length_scales back to None also re-arms the heuristic.)
        self._ls_explicit = length_scales is not None
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.use_gradients = use_gradients
        self.grad_perturbation = grad_perturbation
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_raw: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_scale: float = 1.0
        self._K_inv: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._n_train = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
            objective_fn=None,
            optimize_hyperparams: bool = False) -> None:
        """Fit GP to training data.

        Args:
            X: (n, d) training inputs.
            y: (n,) training outputs.
            objective_fn: Optional callable(X) -> y for computing gradients.
                If use_gradients=True and objective_fn is provided, gradients
                are computed automatically via finite differences.
            optimize_hyperparams: When True, maximize the log marginal
                likelihood over log length-scales and log signal variance
                (L-BFGS-B) before the final fit; on any optimizer failure
                the fixed heuristic hyperparameters are kept. Default False
                preserves exact legacy behavior.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, d), got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one training point")
        self._X_train = X.copy()
        n, d = X.shape

        if self.length_scales is None or not self._ls_explicit:
            # Floor the heuristic by the input range: a single training
            # point or a constant dimension gives std ~ 0, and ls ~ 1e-6
            # would decorrelate the RBF kernel instantly (posterior mean
            # collapsing to the zero prior just off the data).
            x_range = np.ptp(X, axis=0)
            ls_floor = np.maximum(0.1 * np.maximum(x_range, 1.0), 1e-6)
            self.length_scales = np.maximum(np.std(X, axis=0), ls_floor) + 1e-6
        else:
            ls = np.asarray(self.length_scales, dtype=float).ravel()
            if ls.shape != (d,):
                raise ValueError(
                    f"length_scales has shape {ls.shape} but X has {d} "
                    "dims; pass matching length_scales or reset to None"
                )
            self.length_scales = ls

        # Standardize targets so the signal_var=1.0 prior is scale-correct
        # on real-unit outputs. Gradients scale by 1/scale accordingly.
        self._y_raw = y.copy()
        self._y_mean = float(np.mean(y))
        _y_std = float(np.std(y))
        self._y_scale = _y_std if _y_std > 1e-12 else 1.0
        z = (y - self._y_mean) / self._y_scale

        # Compute gradients via finite differences if requested
        if self.use_gradients:
            if objective_fn is None:
                raise ValueError(
                    "use_gradients=True requires an objective_fn for "
                    "finite-difference gradients"
                )
            grad_data = self._compute_gradients(X, objective_fn) / self._y_scale
            y_aug = np.concatenate([z, grad_data])
        else:
            y_aug = z.copy()

        self._y_train = y_aug
        self._n_train = n

        if optimize_hyperparams:
            self._optimize_hyperparams()

        # Build and factor kernel matrix (torch for Cholesky/solve)
        K = _augmented_kernel(
            X, self.length_scales, self.signal_var, self.noise_var,
            include_grad=self.use_gradients,
        )

        # Cholesky factorization with progressive jitter
        K_t = _to_torch(K)
        y_t = _to_torch(y_aug)
        for jitter in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
            try:
                L = torch.linalg.cholesky(K_t + jitter * torch.eye(K_t.shape[0], dtype=K_t.dtype))
                alpha = torch.cholesky_solve(y_t.unsqueeze(1), L).squeeze(1)
                if not torch.all(torch.isfinite(alpha)):
                    continue
                self._L = L
                self._alpha = alpha.numpy()
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise np.linalg.LinAlgError("GP kernel matrix not positive definite")

    def _lml_for(self, length_scales: np.ndarray, signal_var: float) -> float:
        """Log marginal likelihood for given hyperparameters.

        Returns -inf (instead of raising) when the kernel matrix is not
        positive definite, so hyperparameter optimization can move on.
        """
        y = self._y_train
        K = _augmented_kernel(
            self._X_train, length_scales, signal_var, self.noise_var,
            include_grad=self.use_gradients,
        )
        n = K.shape[0]
        K_t = _to_torch(K)
        try:
            L = torch.linalg.cholesky(K_t)
        except torch.linalg.LinAlgError:
            return -np.inf
        sign, logdet = torch.linalg.slogdet(K_t)
        if sign <= 0:
            return -np.inf
        alpha = torch.cholesky_solve(
            _to_torch(y).unsqueeze(1), L
        ).squeeze(1)
        return float(-0.5 * (_to_torch(y) @ alpha + logdet + n * np.log(2 * np.pi)))

    def _optimize_hyperparams(self) -> None:
        """Maximize log marginal likelihood over log length-scales and
        log signal variance via L-BFGS-B; keep heuristics on failure."""
        from scipy.optimize import minimize

        ls0 = np.asarray(self.length_scales, dtype=float)
        theta0 = np.concatenate([np.log(ls0), [np.log(self.signal_var)]])

        def neg_lml(theta: np.ndarray) -> float:
            ls = np.exp(theta[:-1])
            sv = float(np.exp(theta[-1]))
            if not np.all(np.isfinite(ls)) or not np.isfinite(sv):
                return np.inf
            val = self._lml_for(ls, sv)
            return -val if np.isfinite(val) else np.inf

        try:
            res = minimize(neg_lml, theta0, method="L-BFGS-B",
                           options={"maxiter": 100})
        except Exception:
            return
        if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
            return
        ls_new = np.exp(res.x[:-1])
        sv_new = float(np.exp(res.x[-1]))
        # Accept only if it actually improves on the heuristic start.
        if -res.fun > self._lml_for(ls0, self.signal_var):
            self.length_scales = ls_new
            self.signal_var = sv_new

    def _compute_gradients(self, X: np.ndarray, objective_fn) -> np.ndarray:
        """Compute finite-difference gradients at training points."""
        n, d = X.shape
        grads = np.zeros((n * d,))
        for i in range(n):
            for j in range(d):
                x_plus = X[i].copy()
                x_minus = X[i].copy()
                h = self.grad_perturbation * max(abs(X[i, j]), 1.0)
                x_plus[j] += h
                x_minus[j] -= h
                grads[i * d + j] = (objective_fn(x_plus) - objective_fn(x_minus)) / (2 * h)
        return grads

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_train, y_train) copies in original units."""
        if self._X_train is None:
            raise RuntimeError("GP not fitted. Call fit() first.")
        if self._y_raw is None:  # pragma: no cover - defensive
            return self._X_train.copy(), self._y_train[:self._n_train].copy()
        return self._X_train.copy(), self._y_raw.copy()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict at new points.

        Returns:
            mean: (n_test,) predicted mean.
            std: (n_test,) predicted standard deviation.
        """
        if self._alpha is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} dims but GP was fitted with "
                f"{self._X_train.shape[1]}"
            )
        n_test = X.shape[0]
        n_train = self._X_train.shape[0]
        d = self._X_train.shape[1]

        # K_star: (n_test, n_train)
        K_star = _rbf_kernel(X, self._X_train, self.length_scales, self.signal_var)

        if self.use_gradients:
            # K_star_full: (n_test, n_train + n_train*d)
            K_star_f = _rbf_kernel(X, self._X_train, self.length_scales, self.signal_var)
            # K_star_d_full: (n_test, n_train*d) — point-major: [df/dx_0(x0), df/dx_1(x0), ...]
            K_star_d_full = np.zeros((n_test, self._n_train * d))
            for j in range(self._n_train):
                for k in range(d):
                    diff = (X[:, k] - self._X_train[j, k]) / (self.length_scales[k] ** 2)
                    K_star_d_full[:, j * d + k] = K_star_f[:, j] * diff
            K_star_full = np.hstack([K_star_f, K_star_d_full])
            mean = K_star_full @ self._alpha

            K_star_full_t = _to_torch(K_star_full)
            v = torch.cholesky_solve(K_star_full_t.T, self._L).T.numpy()
            K_ii = _rbf_kernel(X, X, self.length_scales, self.signal_var)
            var = np.diag(K_ii) - np.sum(K_star_full * v, axis=1)
        else:
            mean = K_star @ self._alpha
            K_star_t = _to_torch(K_star)
            v = torch.cholesky_solve(K_star_t.T, self._L).T.numpy()
            K_ii = _rbf_kernel(X, X, self.length_scales, self.signal_var)
            var = np.diag(K_ii) - np.sum(K_star * v, axis=1)

        var = np.maximum(var, 0.0)
        return mean * self._y_scale + self._y_mean, np.sqrt(var) * self._y_scale

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood for hyperparameter optimization."""
        if self._alpha is None:
            return -np.inf
        y = self._y_train
        K = _augmented_kernel(
            self._X_train, self.length_scales, self.signal_var, self.noise_var,
            include_grad=self.use_gradients,
        )
        n = K.shape[0]
        K_t = _to_torch(K)
        sign, logdet = torch.linalg.slogdet(K_t)
        if sign <= 0:
            return -np.inf
        alpha_t = _to_torch(self._alpha)
        y_t = _to_torch(y)
        return float(-0.5 * (y_t @ alpha_t + logdet + n * np.log(2 * np.pi)))
