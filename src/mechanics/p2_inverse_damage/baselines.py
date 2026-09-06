"""Field-prediction baselines for the P2 comparison protocol.

Three simple baselines mapping measurements -> damage FIELD predictions,
against which the cINN posterior is compared (roadmap §Baselines):

1. DirectRegressionBaseline   — MLP, sigmoid field output, MSE (point estimate)
2. BayesianRegressionBaseline — closed-form ridge regression with per-pixel
                                posterior variance approximation
3. PixelClassifierBaseline    — joint logistic regression over grid cells

Style matches decoder/posterior: plain classes, numpy forward API,
torch internals where training needs gradients.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ─── 1. Direct regression: measurements → damage field ──────────────


class DirectRegressionBaseline:
    """MLP regressor mapping measurements to a flattened damage field.

    Deterministic point estimate; sigmoid output keeps the field in [0, 1].
    Loss: MSE between predicted and true flattened fields.
    """

    def __init__(
        self,
        d_in: int,
        grid_shape: tuple[int, int],
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        self.d_in = d_in
        self.grid_shape = grid_shape
        self._out_dim = grid_shape[0] * grid_shape[1]
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: list[nn.Module] = []
        dims = [d_in] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], self._out_dim))
        self.mlp = nn.Sequential(*layers)

        gen = torch.Generator().manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, generator=gen)
                nn.init.zeros_(m.bias)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-2,
        batch_size: int | None = None,
        verbose: bool = False,
        seed: int = 0,
    ) -> list[float]:
        """Train on measurements X (n, d_in) and fields Y (n, gy*gx) with MSE."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32).reshape(-1, self._out_dim)
        opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        n = len(X_t)
        if batch_size is None:
            batch_size = n
        history: list[float] = []
        tgen = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=tgen)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                pred = torch.sigmoid(self.mlp(X_t[idx]))
                loss = nn.functional.mse_loss(pred, Y_t[idx])
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach()) * len(idx)
            history.append(epoch_loss / n)
            if verbose and (len(history) % 50 == 0 or len(history) == 1):
                print(f"epoch {len(history):4d}  mse {history[-1]:.6f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict damage fields. X: (n, d_in) → (n, gy, gx) in [0, 1]."""
        with torch.no_grad():
            raw = self.mlp(torch.as_tensor(X, dtype=torch.float32))
            return torch.sigmoid(raw).reshape(-1, *self.grid_shape).numpy()


# ─── 2. Bayesian ridge regression with analytic Gaussian posterior ──


class BayesianRegressionBaseline:
    """Linear-Gaussian baseline: closed-form ridge posterior per pixel.

    Model y_p = w_p · x + b_p + eps, eps ~ N(0, sigma2_p) independently per
    pixel p. The ridge solution gives the Gaussian weight posterior mean;
    the predictive variance at input x combines weight uncertainty
    (leverage x^T A^{-1} x, A = X^T X + lam I) with the residual noise
    floor sigma2_p. Closed form, no iterations.
    """

    def __init__(self, grid_shape: tuple[int, int], lam: float = 1e-2):
        self.grid_shape = grid_shape
        self._out_dim = grid_shape[0] * grid_shape[1]
        self.lam = lam
        self.W_: np.ndarray | None = None  # (d_in, out_dim)
        self.b_: np.ndarray | None = None  # (out_dim,)
        self.A_inv_: np.ndarray | None = None  # (d_in, d_in)
        self.sigma2_: np.ndarray | None = None  # (out_dim,) residual noise

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BayesianRegressionBaseline":
        """Closed-form ridge fit. X: (n, d_in), Y: (n, gy*gx)."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(len(X), self._out_dim)
        # Augment with a constant column so the intercept shares the ridge prior.
        Xa = np.hstack([X, np.ones((len(X), 1))])
        A = Xa.T @ Xa + self.lam * np.eye(Xa.shape[1])
        self.A_inv_ = np.linalg.inv(A)
        Wb = self.A_inv_ @ Xa.T @ Y
        self.W_, self.b_ = Wb[:-1], Wb[-1]
        resid = Y - X @ self.W_ - self.b_
        self.sigma2_ = np.maximum(resid.var(axis=0), 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mean damage fields. X: (n, d_in) → (n, gy, gx)."""
        mean = np.asarray(X, dtype=np.float64) @ self.W_ + self.b_
        return mean.reshape(-1, *self.grid_shape)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std fields.

        X: (n, d_in) → mean (n, gy, gx), std (n, gy, gx).
        Std grows with leverage x^T A^{-1} x, so it is largest far from
        the training-data centroid.
        """
        X = np.asarray(X, dtype=np.float64)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        mean = X @ self.W_ + self.b_  # (n, out)
        leverage = np.einsum("ni,ij,nj->n", Xa, self.A_inv_, Xa)  # (n,)
        var = self.sigma2_[None, :] * (1.0 + leverage[:, None])  # (n, out)
        std = np.sqrt(var)
        return (
            mean.reshape(-1, *self.grid_shape),
            std.reshape(-1, *self.grid_shape),
        )


# ─── 3. Pixel classifier: logistic regression over all cells jointly ──


class PixelClassifierBaseline:
    """Per-grid-cell logistic regression, vectorized as one linear layer.

    Logits = X @ W + b with shape (n, gy*gx); sigmoid gives independent
    P(cell damaged) per cell. Thresholded predict at 0.5.
    """

    def __init__(
        self,
        d_in: int,
        grid_shape: tuple[int, int],
        seed: int = 42,
    ):
        self.d_in = d_in
        self.grid_shape = grid_shape
        self._out_dim = grid_shape[0] * grid_shape[1]
        self.linear = nn.Linear(d_in, self._out_dim)
        # NOTE(seed): kept for API compatibility; zero-init is deterministic
        # and must not reseed the global torch RNG as a side effect.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-1,
        verbose: bool = False,
    ) -> list[float]:
        """Train on binary targets Y (n, gy*gx) in {0, 1} with BCE."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32).reshape(-1, self._out_dim)
        opt = torch.optim.Adam(self.linear.parameters(), lr=lr)
        history: list[float] = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.linear(X_t), Y_t
            )
            loss.backward()
            opt.step()
            history.append(float(loss))
            if verbose and (len(history) % 50 == 0 or len(history) == 1):
                print(f"epoch {len(history):4d}  bce {history[-1]:.6f}")
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Damage probabilities per cell. X: (n, d_in) → (n, gy, gx) in [0, 1]."""
        with torch.no_grad():
            logits = self.linear(torch.as_tensor(X, dtype=torch.float32))
            return (
                torch.sigmoid(logits).reshape(-1, *self.grid_shape).numpy()
            )

    def predict(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Thresholded binary damage map. X: (n, d_in) → (n, gy, gx) in {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)
