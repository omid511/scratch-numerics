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

        z_t = torch.as_tensor(z_col, dtype=torch.float64)
        alpha = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, z_t, upper=False),
            upper=True,
        )
        nlml = 0.5 * z_t @ alpha + torch.log(torch.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)
        return float(nlml)
