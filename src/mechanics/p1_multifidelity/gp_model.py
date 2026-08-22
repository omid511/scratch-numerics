"""GP over latent space: design params θ → posterior over z."""
from __future__ import annotations
import numpy as np
import torch


class RBFKernel:
    """Radial basis function (squared exponential) kernel."""

    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0):
        self.length_scale = length_scale
        self.signal_variance = signal_variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1t = torch.as_tensor(X1, dtype=torch.float64)
        X2t = torch.as_tensor(X2, dtype=torch.float64)
        sq = torch.cdist(X1t, X2t).pow(2)
        K = self.signal_variance * torch.exp(-0.5 * sq / self.length_scale**2)
        return K.numpy()


class LatentGP:
    """GP mapping design parameters θ → latent codes z.

    Independent GP per latent dimension (simpler, interpretable).
    Torch-backed: uses torch.linalg.cholesky and torch.linalg.solve.
    """

    def __init__(self, d_z: int = 16, kernel: RBFKernel | None = None, noise: float = 1e-4):
        self.d_z = d_z
        self.kernel = kernel or RBFKernel()
        self.noise = noise
        self._theta_train: np.ndarray | None = None
        self._z_train: np.ndarray | None = None
        self._K_inv: list[np.ndarray] = []
        self._L_inv: list[np.ndarray] = []

    def fit(self, theta: np.ndarray, z: np.ndarray) -> None:
        self._theta_train = theta.copy()
        self._z_train = z.copy()
        n = theta.shape[0]
        I_n = torch.eye(n, dtype=torch.float64)

        K = torch.as_tensor(self.kernel(theta, theta), dtype=torch.float64) + self.noise * I_n
        self._K_inv = []
        self._L_inv = []
        for i in range(z.shape[1]):
            K_use = K
            try:
                L = torch.linalg.cholesky(K_use)
            except RuntimeError:
                K_use = K + 1e-6 * I_n
                L = torch.linalg.cholesky(K_use)
            L_inv = torch.linalg.solve_triangular(L, I_n, upper=False)
            self._L_inv.append(L_inv.numpy())
            self._K_inv.append((L_inv.T @ L_inv).numpy())

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K_s = torch.as_tensor(
            self.kernel(self._theta_train, theta), dtype=torch.float64
        )
        K_ss = torch.as_tensor(
            self.kernel(theta, theta), dtype=torch.float64
        )
        z_train_t = torch.as_tensor(self._z_train, dtype=torch.float64)

        z_mean = np.zeros((theta.shape[0], self.d_z))
        z_var = np.zeros((theta.shape[0], self.d_z))

        for i in range(self.d_z):
            K_inv_t = torch.as_tensor(self._K_inv[i], dtype=torch.float64)
            L_inv_t = torch.as_tensor(self._L_inv[i], dtype=torch.float64)

            alpha = K_inv_t @ z_train_t[:, i]
            mu = K_s.T @ alpha

            v = L_inv_t @ K_s
            var = torch.diag(K_ss) - v.pow(2).sum(dim=0)
            var = var.clamp(min=0.0)

            z_mean[:, i] = mu.numpy()
            z_var[:, i] = var.numpy()

        return z_mean, z_var

    def sample(self, theta: np.ndarray, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        mean, var = self.predict(theta)
        n_test = theta.shape[0]
        samples = np.zeros((n_samples, n_test, self.d_z))

        for i in range(self.d_z):
            std = np.sqrt(var[:, i])
            eps = rng.standard_normal((n_samples, n_test))
            samples[:, :, i] = mean[:, i] + eps * std[np.newaxis, :]

        return samples

    def negative_log_marginal_likelihood(self, theta: np.ndarray, z_col: np.ndarray) -> float:
        n = theta.shape[0]
        K = torch.as_tensor(self.kernel(theta, theta), dtype=torch.float64) + self.noise * torch.eye(n, dtype=torch.float64)
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K = K + 1e-6 * torch.eye(n, dtype=torch.float64)
            L = torch.linalg.cholesky(K)

        z_t = torch.as_tensor(
            np.asarray(z_col, dtype=np.float64).reshape(n, -1)
        )
        alpha = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, z_t, upper=False),
            upper=True,
        )
        nlml = 0.5 * z_t.ravel() @ alpha.ravel() + torch.log(torch.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)
        return float(nlml)


class ARDRBFKernel:
    """RBF kernel with Automatic Relevance Determination (per-dimension
    length scales) and learned signal variance.

    Parameters are torch tensors with ``requires_grad_=True`` so they can be
    optimized by marginal-likelihood maximization (see fit_hyperparameters).
    ``__call__`` mirrors RBFKernel (numpy in, numpy out) so it drops into
    LatentGP unchanged.

    Args:
        d_in: number of input dimensions.
        log_length_scales: optional initial per-dim log length scales.
        log_signal_variance: initial log signal variance.
    """

    def __init__(
        self,
        d_in: int,
        log_length_scales: np.ndarray | None = None,
        log_signal_variance: float = 0.0,
    ):
        if log_length_scales is None:
            log_length_scales = np.zeros(d_in)
        log_length_scales = np.asarray(log_length_scales, dtype=np.float64)
        if log_length_scales.shape != (d_in,):
            raise ValueError(f"log_length_scales must have shape ({d_in},)")
        self.log_length_scales = torch.tensor(
            log_length_scales, dtype=torch.float64, requires_grad=True
        )
        self.log_signal_variance = torch.tensor(
            float(log_signal_variance), dtype=torch.float64, requires_grad=True
        )

    @property
    def length_scales(self) -> np.ndarray:
        return np.exp(self.log_length_scales.detach().numpy())

    @property
    def signal_variance(self) -> float:
        return float(np.exp(self.log_signal_variance.item()))

    def _cov(self, X1t: "torch.Tensor", X2t: "torch.Tensor") -> "torch.Tensor":
        inv_ls = torch.exp(-2.0 * self.log_length_scales)
        # Pairwise squared distance with per-dim scaling.
        x1 = X1t * inv_ls
        x2 = X2t * inv_ls
        sq = (
            x1.pow(2).sum(dim=1)[:, None]
            + x2.pow(2).sum(dim=1)[None, :]
            - 2.0 * x1 @ x2.T
        ).clamp(min=0.0)
        return torch.exp(self.log_signal_variance) * torch.exp(-0.5 * sq)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1t = torch.as_tensor(np.asarray(X1, dtype=np.float64))
        X2t = torch.as_tensor(np.asarray(X2, dtype=np.float64))
        return self._cov(X1t, X2t).detach().numpy()


def fit_hyperparameters(
    theta: np.ndarray,
    z_col: np.ndarray,
    n_steps: int = 300,
    lr: float = 0.05,
    noise: float = 1e-4,
) -> ARDRBFKernel:
    """Learn ARD length scales and signal variance by minimizing the negative
    log marginal likelihood of a single-output GP (torch autograd, Adam).

    Args:
        theta: (n, d_in) inputs.
        z_col: (n,) targets for ONE output dimension.
        n_steps / lr: Adam optimization budget.
        noise: fixed observation noise variance.

    Returns the fitted ARDRBFKernel (parameters updated in place and returned).
    """
    theta_t = torch.as_tensor(np.asarray(theta, dtype=np.float64))
    y = torch.as_tensor(
        np.asarray(z_col, dtype=np.float64).ravel().reshape(-1, 1)
    )
    n, d_in = theta_t.shape
    kernel = ARDRBFKernel(d_in=d_in)
    params = [kernel.log_length_scales, kernel.log_signal_variance]
    optimizer = torch.optim.Adam(params, lr=lr)
    jitter = torch.eye(n, dtype=torch.float64)


    def nlml() -> "torch.Tensor":
        K = kernel._cov(theta_t, theta_t) + noise * jitter
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            L = torch.linalg.cholesky(K + 1e-6 * jitter)
        alpha = torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, y, upper=False), upper=True
        )
        return (
            0.5 * y.ravel() @ alpha.ravel()
            + torch.log(torch.diag(L)).sum()
            + 0.5 * n * np.log(2 * np.pi)
        )
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = nlml()
        loss.backward()
        optimizer.step()
    return kernel
