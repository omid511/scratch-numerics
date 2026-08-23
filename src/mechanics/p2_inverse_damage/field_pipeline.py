"""Field-level CVAE pipeline for P2 inverse damage identification.

Loads a spatial damage-field dataset produced by ``run_p2.py generate-fields``
(NPZ schema: ``fields_values`` (n, gy, gx) retention grids, ``meas_frequencies``
(n, n_modes), ``meas_severity` (n,) and ``split_train/val/test`` index arrays),
trains a conditional VAE that reconstructs damage *fields* conditioned on
frequency measurements, and evaluates the proposal-2 simulation-based gates
(SP3 interval coverage, SBC rank uniformity, posterior-mean MSE vs the direct
regression baseline).

Conditioning reality (v1 dataset): mode shapes are NOT persisted by
``run_p2.py generate-fields`` — supervision is frequencies-only. The generic
:class:`~mechanics.p2_inverse_damage.encoder.MeasurementEncoder` assumes
``n_modes + n_modes * gy * gx`` inputs (flattened mode shapes), so this module
wraps an equivalent frequencies-only encoder (:class:`FreqOnlyEncoder`) with
the same MLP pattern, and adapts the two-phase training recipe from
``train.py`` (autoencoder pre-training, then ELBO posterior training with
free-bits KL) in-file rather than modifying the shared signatures, which
require mode-shape arrays.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as _normal

from .baselines import DirectRegressionBaseline
from .decoder import DamageDecoder
from .eval import (
    posterior_mean_mse,
    sbc_ranks,
    sbc_rank_uniformity_pvalue,
)
from .posterior import ConditionalPosterior

# Gate thresholds from the P2 charter (SP-gate evaluation).
SP3_COVERAGE_GATE = 0.80   # SP3: 90% per-pixel interval coverage must exceed 0.80
SBC_ERROR_GATE = 0.10      # SBC: mean |bin proportion − uniform| must stay below 0.10

_REQUIRED_NPZ_KEYS = (
    "fields_values",
    "meas_frequencies",
    "meas_severity",
    "split_train",
    "split_val",
    "split_test",
)


# ─── Dataset loading ─────────────────────────────────────────────────


def load_field_dataset(path: str) -> dict:
    """Load a field dataset NPZ written by ``run_p2.py generate-fields``.

    Returns a dict with keys:

        fields:    (n, gy, gx) retention values in (0, 1]
        freqs:     (n, n_modes) raw eigenfrequencies [Hz]
        log_freqs: (n, n_modes) natural-log frequencies (conditioning input;
                   log compresses the dynamic range, mirroring MeasurementEncoder)
        severity:  (n,) = 1 - mean(retention), in (0, 1)
        summaries: optional (n, n_modes*2) per-mode [RMS(w), max|w|] mode-
                   shape summary features, passed through when present
        splits:    {"train": [...], "val": [...], "test": [...]} int index lists

    Raises ValueError on missing keys or inconsistent shapes.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in _REQUIRED_NPZ_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"field dataset NPZ missing keys: {missing}")
        fields = np.asarray(data["fields_values"], dtype=np.float64)
        freqs = np.asarray(data["meas_frequencies"], dtype=np.float64)
        severity = np.asarray(data["meas_severity"], dtype=np.float64)
        # save_dataset_npz prefixes measurement keys with "meas_";
        # accept both that and a bare "summaries" key.
        if "meas_summaries" in data.files:
            summaries = np.asarray(data["meas_summaries"], dtype=np.float64)
        elif "summaries" in data.files:
            summaries = np.asarray(data["summaries"], dtype=np.float64)
        else:
            summaries = None
        splits = {
            "train": np.asarray(data["split_train"], dtype=int).tolist(),
            "val": np.asarray(data["split_val"], dtype=int).tolist(),
            "test": np.asarray(data["split_test"], dtype=int).tolist(),
        }
        config_json = str(data["config"]) if "config" in data.files else None

    n = fields.shape[0]
    if fields.ndim != 3:
        raise ValueError(f"fields_values must be (n, gy, gx); got {fields.shape}")
    if freqs.shape[0] != n or severity.shape[0] != n:
        raise ValueError(
            f"sample-count mismatch: fields {n}, freqs {freqs.shape}, "
            f"severity {severity.shape}"
        )
    overlap = (set(splits["train"]) & set(splits["val"]))
    overlap |= set(splits["train"]) & set(splits["test"])
    overlap |= set(splits["val"]) & set(splits["test"])
    if overlap:
        raise ValueError(f"splits overlap at indices {sorted(overlap)}")

    log_freqs = np.log(np.maximum(freqs, 1e-12))

    out = {
        "fields": fields,
        "freqs": freqs,
        "log_freqs": log_freqs,
        "severity": severity,
        "splits": splits,
    }
    if summaries is not None:
        if summaries.shape[0] != n:
            raise ValueError(
                f"sample-count mismatch: fields {n}, summaries {summaries.shape}"
            )
        out["summaries"] = summaries
    if config_json is not None:
        try:
            out["config"] = json.loads(config_json)
        except json.JSONDecodeError:
            pass
    return out


# ─── Frequencies-only conditioning encoder ───────────────────────────


class FreqOnlyEncoder(nn.Module):
    """Encode log-frequencies (n_modes,) → conditioning vector c ∈ R^{d_c}.

    Drop-in replacement for the frequency half of
    :class:`~mechanics.p2_inverse_damage.encoder.MeasurementEncoder`: the v1
    field dataset stores no mode shapes, so the flattened-mode-shape block of
    the input is simply absent. Same hidden-layer pattern and init scheme.
    """

    def __init__(
        self,
        n_modes: int = 6,
        d_c: int = 64,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.d_c = d_c
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers: list[nn.Module] = []
        dims = [n_modes] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], d_c))
        self.mlp = nn.Sequential(*layers)

        torch.manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_tensor(self, log_freqs: torch.Tensor) -> torch.Tensor:
        """log_freqs: (B, n_modes) → c: (B, d_c), grad-preserving."""
        return self.mlp(log_freqs)

    def forward(self, log_freqs: np.ndarray) -> np.ndarray:
        """NumPy inference path. log_freqs: (batch, n_modes) → (batch, d_c)."""
        x = torch.as_tensor(log_freqs, dtype=torch.float32)
        with torch.no_grad():
            return self.mlp(x).numpy()


def _run_epochs(
    n_epochs: int,
    x: np.ndarray,
    y: np.ndarray,
    params: list[torch.nn.Parameter],
    step_fn,
    *,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
    progress: bool,
    tag: str,
) -> list[float]:
    """Shared Adam mini-batch loop for both field pipelines.

    ``step_fn(bx, by)`` returns a scalar loss tensor; per-epoch mean loss is
    recorded. Mirrors the loop previously nested in ``train_field_cvae``.
    """
    optim = torch.optim.Adam(params, lr=lr)
    history = []
    for epoch in range(n_epochs):
        perm = rng.permutation(len(x))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            bx = torch.as_tensor(x[idx], dtype=torch.float32)
            by = torch.as_tensor(y[idx], dtype=torch.float32)
            optim.zero_grad()
            loss = step_fn(bx, by)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        history.append(epoch_loss / n_batches)
        if progress and (
            (epoch + 1) % max(1, n_epochs // 5) == 0 or epoch + 1 == n_epochs
        ):
            print(
                f"[{tag}] epoch {epoch + 1}/{n_epochs} "
                f"loss {history[-1]:.6f}",
                flush=True,
            )
    return history



class FreqSummaryEncoder(nn.Module):
    """Encode concat(log_freqs, summaries) → conditioning vector c ∈ R^{d_c}.

    Input layout (documented dims):
      * ``log_freqs``:  (B, n_modes)
      * ``summaries``:  (B, n_modes * d_summary_block), per-mode block of
        ``d_summary_block`` mode-shape summary statistics
        ([RMS(w), max|w|] → d_summary_block = 2 in the dataset writer).

    Same MLP trunk / hidden dims / init scheme as
    :class:`FreqOnlyEncoder`; only the input width grows by
    ``n_modes * d_summary_block``.
    """

    def __init__(
        self,
        n_modes: int = 6,
        d_c: int = 64,
        d_summary_block: int = 2,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.d_c = d_c
        self.d_summary_block = d_summary_block
        self.d_in = n_modes + n_modes * d_summary_block
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers: list[nn.Module] = []
        dims = [self.d_in] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], d_c))
        self.mlp = nn.Sequential(*layers)

        torch.manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_modes + n_modes*d_summary_block) → c: (B, d_c)."""
        return self.mlp(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """NumPy inference path. (batch, d_in) → (batch, d_c)."""
        x_t = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.mlp(x_t).numpy()


# ─── Training ────────────────────────────────────────────────────────


def _epoch_means(
    step_pairs: list[list[float]], n_rows: int, batch_size: int
) -> list[list[float]]:
    """Per-step [a, b] pairs → per-epoch means [[a, b], ...]."""
    if not step_pairs:
        return []
    n_batches = max(1, (n_rows + batch_size - 1) // batch_size)
    arr = np.asarray(step_pairs, dtype=np.float64)
    return arr.reshape(-1, n_batches, 2).mean(axis=1).tolist()


def train_field_cvae(
    dataset: dict,
    *,
    d_c: int = 64,
    d_z: int = 16,
    epochs_ae: int = 30,
    epochs_post: int = 50,
    batch_size: int = 16,
    seed: int = 0,
    lr: float = 1e-3,
    progress: bool = True,
    cond_decoder: bool = True,
    kl_anneal_epochs: int = 0,
    kl_weight_final: float = 1.0,
    free_bits: bool = True,
    use_summaries: bool = False,
    posterior_init_std: float = 0.01,
) -> dict:
    """Train the field CVAE on the train split of a loaded field dataset.
    Phase 1 (autoencoder pre-training): c = encoder(log_freqs), z ~ N(0, I),
    decoder(z, c) reconstructed against the true retention field with MSE;
    encoder + decoder receive gradients.

    Phase 2 (ELBO): c → posterior(c) → z (reparameterization) → decoder;
    reconstruction MSE + free-bits-clamped KL against N(0, I); encoder,
    decoder and posterior all receive gradients. Mirrors ``train.py``
    conventions adapted to frequencies-only conditioning.

    Anti-collapse options (posterior collapse observed with defaults —
    ELBO pinned at the free-bits floor, SP3 coverage 0.69):
      * ``cond_decoder=False`` routes measurement information ONLY through
        z (decoder receives a zero conditioning block), forcing the latent
        to carry it — the standard CVAE de-conditioning remedy.
      * ``free_bits=False`` drops the per-dim ``clamp(min=0.5)`` (plain
        KL) and holds the KL weight at ``kl_weight_final`` for all of
        Phase 2 (``kl_anneal_epochs`` is ignored on this path). With a
        small final weight the posterior can carry information without
        the floor's dead-gradient regime — but also without its collapse
        protection.
      * ``kl_anneal_epochs>0`` ramps the KL weight linearly from zero over
        that many ELBO epochs before full weight.
      * ``posterior_init_std`` scales the posterior head re-init. The
        0.01 default puts mu/logvar outputs inside the free-bits clamp
        (kl_per_dim < 0.5 everywhere) from step zero, so ``torch.clamp``
        passes zero gradient and the posterior can never escape collapse;
        larger values (e.g. 0.5-1.0) start the KL above the floor with a
        live gradient. (The pre-default failure was the opposite extreme:
        std=default gave mu~O(10), KL~2.7e4 nats at step zero.)

    Only ``dataset["splits"]["train"]`` indices are touched.

    ``use_summaries=True`` switches conditioning to
    :class:`FreqSummaryEncoder` and augments every model input row to
    ``concat(log_freqs, summaries)`` (width n_modes + n_modes*2); the
    augmented rows must also be fed at evaluation time (see
    :func:`evaluate_sp_gates`, which reads the echoed option).
    Returns dict with keys: ``encoder``, ``decoder``, ``posterior`` (modules),
    ``history`` = {"ae": [...], "posterior": [...]}, ``grid_shape`` = (gy, gx),
    ``n_modes``, ``splits`` (echo), ``options`` echo.
    """
    fields = np.asarray(dataset["fields"], dtype=np.float32)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float32)
    splits = dataset["splits"]
    train_idx = np.asarray(splits["train"], dtype=int)

    n, gy, gx = fields.shape
    n_modes = log_freqs.shape[1]
    if use_summaries:
        if "summaries" not in dataset:
            raise ValueError("use_summaries=True requires dataset['summaries']")
        summaries = np.asarray(dataset["summaries"], dtype=np.float32)
        x_train = np.concatenate(
            [log_freqs[train_idx], summaries[train_idx]], axis=1
        )
        encoder = FreqSummaryEncoder(
            n_modes=n_modes,
            d_c=d_c,
            d_summary_block=summaries.shape[1] // n_modes,
            seed=seed,
        )
    else:
        x_train = log_freqs[train_idx]
        encoder = FreqOnlyEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
    y_train = fields[train_idx]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    decoder = DamageDecoder(d_z=d_z,
                            d_c=(d_c if cond_decoder else 0),
                            grid_size=(gy, gx), seed=seed)
    posterior = ConditionalPosterior(d_z=d_z, d_c=d_c, seed=seed)
    # Small-variance re-init of the posterior heads: their default init
    # produced mu~O(10) against the N(0,I) prior -> KL ~2.7e4 nats at step
    # zero, drowning the reconstruction term (empirically measured).
    for head in (posterior.trunk, posterior.mu_head, posterior.logvar_head):
        if hasattr(head, "parameters"):
            for prm in head.parameters():
                if prm.requires_grad and prm.ndim > 1:
                    torch.nn.init.normal_(prm, mean=0.0, std=posterior_init_std)


    # Phase 1: autoencoder pre-training (z ~ N(0, I)).
    def _ae_step(bx, by):
        c = encoder.forward_tensor(bx)
        z = torch.randn(bx.shape[0], decoder.d_z)
        cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
        pred = decoder.forward_tensor(z, cc)
        return torch.mean((pred - by) ** 2)

    comps: list[list[float]] = []
    if not free_bits:
        kl_weight = [kl_weight_final]
    else:
        kl_weight = [kl_weight_final if kl_anneal_epochs == 0 else 0.0]
    step_counter = [0]

    history_ae = _run_epochs(
        epochs_ae, x_train, y_train,
        list(encoder.mlp.parameters()) + list(decoder.mlp.parameters()),
        _ae_step,
        batch_size=batch_size, lr=lr, rng=rng, progress=progress,
        tag="train_field_cvae ae",
    )

    # Phase 2: ELBO with free-bits KL (matches train.train_posterior).
    def _elbo_step(bx, by):
        if free_bits and kl_anneal_epochs > 0:
            kl_weight[0] = min(
                kl_weight_final,
                kl_weight_final * (step_counter[0] + 1) / kl_anneal_epochs,
            )
        c = encoder.forward_tensor(bx)
        cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
        mu, log_var = posterior.forward_tensor(c)
        # Bounded log-variance: an unbounded branch lets KL explode through
        # exp(log_var) early in training and stall the ELBO (observed ~11k
        # nats flat). +-10 nats is far beyond any calibrated posterior here.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        pred = decoder.forward_tensor(z, cc)
        recon = torch.mean((pred - by) ** 2)
        kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
        if free_bits:
            kl_per_dim = torch.clamp(kl_per_dim, min=0.5)
        kl = torch.mean(torch.sum(kl_per_dim, dim=-1))
        comps.append([float(recon.detach()), float(kl.detach())])
        step_counter[0] += 1
        return recon + kl_weight[0] * kl

    history_post = _run_epochs(
        epochs_post, x_train, y_train,
        (
            list(encoder.mlp.parameters())
            + list(decoder.mlp.parameters())
            + list(posterior.trunk.parameters())
            + list(posterior.mu_head.parameters())
            + list(posterior.logvar_head.parameters())
        ),
        _elbo_step,
        batch_size=batch_size, lr=lr, rng=rng, progress=progress,
        tag="train_field_cvae elbo",
    )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "posterior": posterior,
        "history": {"ae": history_ae, "posterior": history_post,
                    # Per-epoch [recon, raw-KL] means (KL before weighting).
                    "posterior_components": _epoch_means(
                        comps, len(x_train), batch_size)},
        "grid_shape": (gy, gx),
        "n_modes": n_modes,
        "splits": splits,
        "options": {"cond_decoder": cond_decoder,
                    "kl_anneal_epochs": kl_anneal_epochs,
                    "use_summaries": use_summaries,
                    "posterior_init_std": posterior_init_std,
                    "kl_weight_final": kl_weight_final,
                    "free_bits": free_bits},
     }


# ─── Heteroscedastic Gaussian head ───────────────────────────────────


class HeteroscedasticFieldDecoder(nn.Module):
    """Decode latent z (+ conditioning c) → per-pixel (mu, log_sigma) grids.

    Mirrors :class:`DamageDecoder`'s MLP trunk but replaces the single
    sigmoid output with TWO parallel linear heads over the shared trunk
    feature map: a mean head passed through a sigmoid (retention ∈ (0, 1])
    and a log-sigma head squashed through a sigmoid onto
    ``[log(sigma_min), log(sigma_max)]`` (bounded by construction).
    """

    def __init__(
        self,
        d_z: int,
        d_c: int,
        grid_size: tuple[int, int],
        sigma_range: tuple[float, float] = (0.005, 0.5),
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.d_z = d_z
        self.d_c = d_c
        self.grid_size = grid_size
        self._out_dim = grid_size[0] * grid_size[1]
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        layers: list[nn.Module] = []
        dims = [d_z + d_c] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        # Unit-scale trunk features for BOTH heads: wide-posterior latents
        # (std ~1, free-bits KL gives no pressure to shrink) can spike the
        # kaiming trunk output by an order of magnitude; through the mu head
        # that saturates the output sigmoid (logits pinned past +15, dead
        # gradient, predictions frozen at 1.0 — observed). The norm keeps
        # logits in the live-gradient region for any latent scale.
        self.trunk_norm = nn.LayerNorm(hidden_dims[-1])
        self.mu_head = nn.Linear(hidden_dims[-1], self._out_dim)
        # Sigma branch: narrow projection -> per-pixel head (trunk features
        # are already unit-scale via trunk_norm). The narrow bottleneck
        # matters: with a direct Linear(hidden, out) head, Adam's
        # per-parameter normalized steps compound across fan_in*features
        # into multi-unit jumps of the raw log-sigma pre-activation PER
        # STEP, violently overshooting the squashing sigmoid into
        # saturation (where its gradient dies). Through a 16-wide
        # bottleneck one step moves it by O(1e-3).
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, self._out_dim),
        )
        # Bounded sigma by construction: raw head output squashed through a
        # sigmoid onto [log(sigma_min), log(sigma_max)]. A hard clamp would
        # zero the gradient whenever the raw head sits outside the range,
        # freezing the sigma head entirely.
        self._log_sigma_min = float(np.log(sigma_range[0]))
        self._log_sigma_span = float(
            np.log(sigma_range[1]) - np.log(sigma_range[0])
        )

        torch.manual_seed(seed)
        for module in (self.trunk, self.mu_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.sigma_head[0].weight)
        nn.init.zeros_(self.sigma_head[0].bias)
        # Zero-init the last sigma layer: raw starts at 0, i.e. sigma at
        # the geometric middle of sigma_range under the squashing sigmoid.
        nn.init.zeros_(self.sigma_head[2].weight)
        nn.init.zeros_(self.sigma_head[2].bias)

    def forward_tensor(
        self, z: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grad-preserving batch forward. Returns (mu, log_sigma), (B, gy, gx)."""
        x = z if c is None else torch.cat([z, c], dim=-1)
        h = self.trunk_norm(self.trunk(x))
        mu = torch.sigmoid(self.mu_head(h))
        sigma, log_sigma = self.sigma_tensor(h)
        return (mu.reshape(-1, *self.grid_size),
                log_sigma.reshape(-1, *self.grid_size))

    def sigma_tensor(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Grad-preserving sigma/log_sigma from trunk features (B, gy, gx)."""
        log_sigma = self._log_sigma_min + self._log_sigma_span * torch.sigmoid(
            self.sigma_head(h)
        )
        log_sigma = log_sigma.reshape(-1, *self.grid_size)
        return torch.exp(log_sigma), log_sigma

    def forward(
        self, z: np.ndarray, c: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy inference path. Returns (mu, log_sigma), (B, gy, gx) each."""
        z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        c_t = None if c is None else torch.as_tensor(c, dtype=torch.float32)
        with torch.no_grad():
            mu, log_sigma = self.forward_tensor(z_t, c_t)
        return mu.numpy(), log_sigma.numpy()


def train_field_cvae_heteroscedastic(
    dataset: dict,
    *,
    d_c: int = 64,
    d_z: int = 16,
    epochs_ae: int = 30,
    epochs_post: int = 50,
    batch_size: int = 16,
    seed: int = 0,
    lr: float = 1e-3,
    progress: bool = True,
    cond_decoder: bool = True,
    kl_anneal_epochs: int = 0,
    sigma_range: tuple[float, float] = (0.005, 0.5),
    use_summaries: bool = False,
    posterior_init_std: float = 0.01,
) -> dict:
    """Train a heteroscedastic Gaussian CVAE head on a loaded field dataset.

    Same encoder / conditional posterior / two-phase structure as
    :func:`train_field_cvae`; the decoder output layer and the Phase-2
    reconstruction term differ:

      * The decoder trunk feeds TWO parallel heads producing per-pixel
        ``mu`` (sigmoid-bounded retention mean) and ``log_sigma`` grids.
      * Phase-2 reconstruction is the Gaussian NLL
        ``mean(log_sigma + (y - mu)^2 / (2 * sigma^2))`` where
        ``log_sigma = log(sigma_min) + (log(sigma_max) - log(sigma_min)) *
        sigmoid(raw)`` — sigma bounded in ``sigma_range`` by construction
        (a hard clamp would zero its gradient at the endpoints). The KL term
        (free-bits floor 0.5 nats/dim, ±10-nat log-variance clamp, optional
        linear anneal) is identical to :func:`train_field_cvae`.
      * Reconstruction is evaluated at the posterior-MEAN latent: the
        returned ``interval_predictor`` reports mean-latent fields, so the
        learned sigma must calibrate that same prediction's residual, not
        the spread induced by sampling z.
      * Phase 2 runs in two sub-phases: the sigma head stays frozen at the
        range's geometric midpoint for the first half of ``epochs_post``
        while mu converges, then it is released at 0.1x lr. Training them
        jointly from step zero runaway-couples (sigma above residual scale
        collapses the NLL mu-gradient, mu degrades, residuals grow, sigma
        chases them to the ceiling).

    Only ``dataset["splits"]["train"]`` indices are touched.
    ``use_summaries=True`` mirrors :func:`train_field_cvae`: conditioning
    switches to :class:`FreqSummaryEncoder` on concat(log_freqs, summaries)
    rows; the echoed option tells :func:`evaluate_sp_gates` to feed the
    same augmented rows back through the interval predictor.

    Returns the same models-dict contract as :func:`train_field_cvae`
    (``encoder``/``decoder``/``posterior``, ``history``, ``grid_shape``,
    ``n_modes``, ``splits``, ``options`` — plus ``sigma_range`` echoed in
    ``options``) PLUS ``interval_predictor``: a closure mapping an
    ``(n, n_modes)`` log-frequency matrix to per-pixel ``(mu, sigma)``
    grids of shape ``(n, gy, gx)``, evaluated at the posterior-mean latent
    (:func:`evaluate_sp_gates` consumes it for analytic SP3 intervals).
    """
    fields = np.asarray(dataset["fields"], dtype=np.float32)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float32)
    splits = dataset["splits"]
    train_idx = np.asarray(splits["train"], dtype=int)

    n, gy, gx = fields.shape
    n_modes = log_freqs.shape[1]
    if use_summaries:
        if "summaries" not in dataset:
            raise ValueError("use_summaries=True requires dataset['summaries']")
        summaries = np.asarray(dataset["summaries"], dtype=np.float32)
        x_train = np.concatenate(
            [log_freqs[train_idx], summaries[train_idx]], axis=1
        )
    else:
        x_train = log_freqs[train_idx]
    y_train = fields[train_idx]
    log_sigma_min = float(np.log(sigma_range[0]))
    log_sigma_max = float(np.log(sigma_range[1]))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    encoder = (
        FreqSummaryEncoder(
            n_modes=n_modes,
            d_c=d_c,
            d_summary_block=summaries.shape[1] // n_modes,
            seed=seed,
        )
        if use_summaries
        else FreqOnlyEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
    )
    decoder = HeteroscedasticFieldDecoder(d_z=d_z,
                                          d_c=(d_c if cond_decoder else 0),
                                          grid_size=(gy, gx),
                                          sigma_range=sigma_range, seed=seed)
    posterior = ConditionalPosterior(d_z=d_z, d_c=d_c, seed=seed)
    # Same small-variance posterior re-init as train_field_cvae: default
    # heads put KL ~O(10^4) nats at step zero and drown the NLL term.
    for head in (posterior.trunk, posterior.mu_head, posterior.logvar_head):
        if hasattr(head, "parameters"):
            for prm in head.parameters():
                if prm.requires_grad and prm.ndim > 1:
                    torch.nn.init.normal_(prm, mean=0.0, std=posterior_init_std)

    # Phase 1: autoencoder pre-training on the mean head (z ~ N(0, I)).
    def _ae_step(bx, by):
        c = encoder.forward_tensor(bx)
        cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
        z = torch.randn(bx.shape[0], decoder.d_z)
        pred_mu, _ = decoder.forward_tensor(z, cc)
        return torch.mean((pred_mu - by) ** 2)

    kl_weight = [1.0 if kl_anneal_epochs == 0 else 0.0]
    step_counter = [0]

    history_ae = _run_epochs(
        epochs_ae, x_train, y_train,
        (list(encoder.mlp.parameters())
         + list(decoder.trunk.parameters())
         + list(decoder.mu_head.parameters())),
        _ae_step,
        batch_size=batch_size, lr=lr, rng=rng, progress=progress,
        tag="train_field_cvae_heteroscedastic ae",
    )

    # Phase 2a (mu warmup): ELBO with MSE reconstruction at the posterior-
    # MEAN latent + the identical free-bits KL. Two deliberate deviations
    # from naive joint NLL training, each fixing an observed failure:
    #
    # * Posterior-mean latent, not a reparameterized sample: the interval
    #   predictor reports mean-latent fields, so the sigma head must
    #   calibrate THAT prediction's residual. Sampling z folds the latent
    #   spread into the residuals (observed: sigma -> 0.30 vs true
    #   aleatoric 0.05).
    # * MSE, not NLL, for mu: with sigma frozen near the small end of
    #   sigma_range the NLL scales mu-gradients by 1/sigma^2 (~400x),
    #   which destabilizes mu into a systematically biased compromise
    #   (observed: residual std 0.05 but RMS 0.30).
    def _elbo_mse_step(bx, by):
        if kl_anneal_epochs > 0:
            kl_weight[0] = min(1.0, (step_counter[0] + 1) / kl_anneal_epochs)
        c = encoder.forward_tensor(bx)
        cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
        mu, log_var = posterior.forward_tensor(c)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        pred_mu, _ = decoder.forward_tensor(mu, cc)
        recon = torch.mean((pred_mu - by) ** 2)
        kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
        kl = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.5), dim=-1))
        comps.append([float(recon.detach()), float(kl.detach())])
        step_counter[0] += 1
        return recon + kl_weight[0] * kl

    comps: list[list[float]] = []

    _main_params = (
        list(encoder.mlp.parameters())
        + list(decoder.trunk.parameters())
        + list(decoder.mu_head.parameters())
        + list(posterior.trunk.parameters())
        + list(posterior.mu_head.parameters())
        + list(posterior.logvar_head.parameters())
    )
    epochs_mu = max(1, epochs_post // 2) if epochs_post > 1 else epochs_post
    history_post = _run_epochs(
        epochs_mu, x_train, y_train, _main_params,
        _elbo_mse_step,
        batch_size=batch_size, lr=lr, rng=rng, progress=progress,
        tag="train_field_cvae_heteroscedastic elbo(mu)",
    )

    # Phase 2b (sigma calibration): Gaussian NLL on the FROZEN mu path.
    # Only the sigma head trains, so it settles onto the true residual RMS
    # without runaway coupling.
    def _nll_sigma_step(bx, by):
        with torch.no_grad():
            c = encoder.forward_tensor(bx)
            cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
            mu, _ = posterior.forward_tensor(c)
            pred_mu, _ = decoder.forward_tensor(mu, cc)
            x = torch.cat([mu, cc], dim=-1) if cc.shape[1] else mu
            h = decoder.trunk(x)
        sigma, log_sigma = decoder.sigma_tensor(h.detach())
        return torch.mean(
            log_sigma + (by - pred_mu) ** 2 / (2.0 * sigma**2)
        )

    history_post += _run_epochs(
        max(0, epochs_post - epochs_mu), x_train, y_train,
        list(decoder.sigma_head.parameters()),
        _nll_sigma_step,
        batch_size=batch_size, lr=lr, rng=rng, progress=progress,
        tag="train_field_cvae_heteroscedastic elbo(sigma)",
    )

    def interval_predictor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, n_modes) log-freqs → ((n, gy, gx) mu, (n, gy, gx) sigma).

        Deterministic predictive center: z is the posterior MEAN (no
        sampling), so all reported uncertainty comes from the learned
        sigma head, clamped into ``sigma_range``.
        """
        lf = np.asarray(X, dtype=np.float32)
        c_np = np.asarray(encoder.forward(lf))
        cc = c_np if cond_decoder else np.zeros((len(lf), 0))
        mu_z, log_var = posterior(c_np)
        mu_grid, log_sigma_grid = decoder.forward(mu_z, cc)
        sigma_grid = np.exp(log_sigma_grid)   # already bounded by construction
        return mu_grid, sigma_grid

    return {
        "encoder": encoder,
        "decoder": decoder,
        "posterior": posterior,
        "history": {"ae": history_ae, "posterior": history_post,
                    # Per-epoch [recon, raw-KL] means over the mu-warmup
                    # sub-phase (KL before weighting).
                    "posterior_components": _epoch_means(
                        comps, len(x_train), batch_size)},
        "grid_shape": (gy, gx),
        "n_modes": n_modes,
        "splits": splits,
        "options": {"cond_decoder": cond_decoder,
                    "kl_anneal_epochs": kl_anneal_epochs,
                    "sigma_range": sigma_range,
                    "use_summaries": use_summaries,
                    "posterior_init_std": posterior_init_std},
        "interval_predictor": interval_predictor,
    }


# ─── SP-gate evaluation ──────────────────────────────────────────────

def _field_ensemble(
    encoder,
    decoder,
    posterior,
    log_freq_row: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    cond_decoder: bool = True,
) -> np.ndarray:
    """Draw n_samples posterior field samples for one observation.

    x_row: (n_modes,) log-freqs, or (n_modes + n_modes*2,) augmented row
    when the encoder is a FreqSummaryEncoder → (n_samples, gy, gx) fields.
    """
    c = np.asarray(encoder.forward(log_freq_row[np.newaxis, :]))       # (1, d_c)
    mu, log_var = posterior(c)                                          # (1, d_z) each
    eps = rng.standard_normal((n_samples, mu.shape[-1]))
    z = mu + np.exp(0.5 * log_var) * eps                                # (S, d_z)
    if cond_decoder:
        dec_in_cond = np.repeat(c, n_samples, axis=0)
    else:
        dec_in_cond = np.zeros((n_samples, 0))
    return np.asarray(decoder.forward(z, dec_in_cond))                  # (S, gy, gx)


def _severity(field_stack: np.ndarray) -> np.ndarray:
    """Per-sample mean-severity of stacked fields (S, gy, gx) → (S,)."""
    return 1.0 - field_stack.mean(axis=(-2, -1))


def evaluate_sp_gates(
    models: dict,
    dataset: dict,
    *,
    n_samples: int = 50,
    alpha: float = 0.10,
    seed: int | None = None,
    baseline_epochs: int = 200,
) -> dict:
    """Evaluate the P2 SP-gates on the **validation split only**.

    Two model paths, selected by the presence of ``interval_predictor`` in
    ``models``:

    * ENSEMBLE PATH (default; e.g. :func:`train_field_cvae` output): models
      expose ``encoder``/``decoder``/``posterior`` with the standard call
      conventions.
    * ANALYTIC PATH (e.g. :func:`train_field_cvae_heteroscedastic` output):
      ``models["interval_predictor"]`` is a callable mapping an
      ``(n, n_modes)`` log-frequency matrix to per-pixel ``(mu, sigma)``
      grids. SP3 then uses CLOSED-FORM normal intervals
      ``mu ± z_{alpha/2} * sigma`` (z_{alpha/2} = 1.645 at alpha = 0.10)
      instead of quantiled ensembles — no posterior sampling for coverage.

    Gates:

    SP3 (interval coverage): per-pixel central ``(1 - alpha)`` intervals —
    quantiled ensembles on the ensemble path, analytic normal bounds on the
    analytic path — and average the fraction of pixels whose true retention
    lies inside. Gate: coverage > 0.80.

    SBC (sample-based, BOTH paths): rank of the true mean severity within
    each val observation's severity ensemble of ``n_samples`` draws —
    decoded posterior fields q(z|c)·decoder on the ensemble path,
    ``mu + sigma * eps`` per pixel on the analytic path (ties count as "not
    below", standard SBC). Uniformity assessed two ways over the exact
    discrete support {0..n_samples}: a chi-square goodness-of-fit p-value
    (large p ⇒ calibrated) and the calibration error mean(|bin proportion -
    uniform proportion|). Gate: error < 0.10.

    Point accuracy: mean-field MSE against the truth (ensemble mean or mu),
    reported next to a :class:`DirectRegressionBaseline` (MLP log-freqs →
    field) fitted on the train split.

    Returns dict with keys: ``coverage``, ``coverage_gate_pass``, ``ranks``,
    ``sbc_pvalue``, ``sbc_error``, ``sbc_gate_pass``, ``cvae_mse``,
    ``baseline_mse``, ``n_val``, ``alpha``, ``n_samples``.
    """
    predictor = models.get("interval_predictor")

    fields = np.asarray(dataset["fields"], dtype=np.float64)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float64)
    severity = np.asarray(dataset["severity"], dtype=np.float64)
    val_idx = np.asarray(dataset["splits"]["val"], dtype=int)
    if len(val_idx) == 0:
        raise ValueError("validation split is empty")

    rng = np.random.default_rng(seed)
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    z_crit = float(_normal.ppf(1.0 - alpha / 2.0))     # 1.645 at alpha=0.10

    options = models.get("options", {}) if isinstance(models, dict) else {}
    cond_decoder = bool(options.get("cond_decoder", True))
    use_summaries = bool(options.get("use_summaries", False))
    # Feed the model the SAME rows it was trained on: with use_summaries
    # the conditioning input is concat(log_freqs, summaries). The
    # DirectRegressionBaseline below stays on log_freqs only (it is the
    # no-summaries reference).
    X = log_freqs
    if use_summaries:
        if "summaries" not in dataset:
            raise ValueError("use_summaries=True requires dataset['summaries']")
        X = np.concatenate(
            [log_freqs,
             np.asarray(dataset["summaries"], dtype=np.float64)],
            axis=1,
        )

    covered_pixels = []
    severity_ensembles = []
    mean_field_preds = []
    if predictor is not None:
        # Analytic path: closed-form normal SP3 intervals; SBC severities
        # still sampled as mu + sigma * eps per pixel.
        mu_val, sigma_val = predictor(X[val_idx])
        lower = mu_val - z_crit * sigma_val
        upper = mu_val + z_crit * sigma_val
        truth = fields[val_idx]
        covered_pixels = np.mean(
            (truth >= lower) & (truth <= upper), axis=(1, 2)
        ).tolist()
        for j in range(len(val_idx)):
            ens = mu_val[j][np.newaxis] + sigma_val[j][np.newaxis] * (
                rng.standard_normal((n_samples, *mu_val[j].shape))
            )
            severity_ensembles.append(_severity(ens))
        mean_field_preds = list(mu_val)

    else:
        encoder = models["encoder"]
        decoder = models["decoder"]
        posterior_m = models["posterior"]
        for i in val_idx:
            ens = _field_ensemble(
                encoder, decoder, posterior_m, X[i], n_samples, rng,
                cond_decoder=cond_decoder,
            )
            lower = np.quantile(ens, lower_q, axis=0)
            upper = np.quantile(ens, upper_q, axis=0)
            truth = fields[i]
            covered_pixels.append(
                float(np.mean((truth >= lower) & (truth <= upper)))
            )
            severity_ensembles.append(_severity(ens))
            mean_field_preds.append(ens.mean(axis=0))

    coverage = float(np.mean(covered_pixels))
    ranks = sbc_ranks(severity_ensembles, severity[val_idx])
    sbc_pvalue = sbc_rank_uniformity_pvalue(ranks, n_bins=n_samples + 1)

    # Calibration error over the exact discrete support {0..n_samples}.
    counts = np.bincount(ranks, minlength=n_samples + 1).astype(float)
    props = counts / counts.sum()
    sbc_error = float(np.mean(np.abs(props - 1.0 / (n_samples + 1))))

    cvae_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(mean_field_preds, val_idx)]
    ))

    baseline = DirectRegressionBaseline(
        d_in=log_freqs.shape[1],
        grid_shape=fields.shape[1:],
    )
    train_idx = np.asarray(dataset["splits"]["train"], dtype=int)
    baseline.fit(
        log_freqs[train_idx],
        fields[train_idx].reshape(len(train_idx), -1),
        epochs=baseline_epochs,
    )
    baseline_preds = baseline.predict(log_freqs[val_idx])
    baseline_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(baseline_preds, val_idx)]
    ))

    return {
        "coverage": coverage,
        "coverage_gate_pass": bool(coverage > SP3_COVERAGE_GATE),
        "ranks": ranks,
        "sbc_pvalue": sbc_pvalue,
        "sbc_error": sbc_error,
        # Both statistics required: mean |bin prop - uniform| alone can sit
        # under the gate while a systematically skewed histogram rejects
        # uniformity (observed: error 0.025 at chi-square p = 0.0).
        "sbc_gate_pass": bool(sbc_error < SBC_ERROR_GATE and sbc_pvalue > 0.05),
        "cvae_mse": cvae_mse,
        "baseline_mse": baseline_mse,
        "n_val": int(len(val_idx)),
        "alpha": alpha,
        "n_samples": n_samples,
    }


def _bma_combine(
    mu_stack: np.ndarray, sigma_stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian model averaging of K member predictions.

    mu_stack / sigma_stack: (K, n, gy, gx). Returns (mu_bar, total_sigma)
    with per-pixel total variance by the law of total variance:
    ``Var = mean_k(sigma_k^2 + mu_k^2) - mu_bar^2`` (aleatoric + between-
    member epistemic spread).
    """
    mu_bar = mu_stack.mean(axis=0)
    total_var = np.mean(sigma_stack**2 + mu_stack**2, axis=0) - mu_bar**2
    return mu_bar, np.sqrt(np.maximum(total_var, 0.0))


def train_heteroscedastic_ensemble(
    dataset: dict,
    *,
    n_models: int = 5,
    seeds: list[int] | None = None,
    bootstrap_resample: bool = False,
    **kwargs,
) -> dict:
    """Train ``n_models`` heteroscedastic members with distinct seeds.

    Each member is a full :func:`train_field_cvae_heteroscedastic` run
    (staging untouched); remaining keyword arguments pass through to it.
    Sequential CPU training.

    With ``bootstrap_resample=True`` (bagging) each member trains on its
    own bootstrap resample — same size as the train split, drawn WITH
    replacement from TRAIN rows only, fields/log_freqs kept row-aligned,
    rng seeded with the member's seed. Val/test rows are never touched.
    This injects genuine data diversity so between-member spread carries
    real epistemic variance; seed-only ensembles converge to near-identical
    members. The drawn indices are stored per member as
    ``bootstrap_indices``.

    Returns ``{"members": [model dicts], "combined": {...}}`` where
    ``combined["interval_predictor"](X) -> (mu_bar, total_sigma)`` is the
    BMA-combined predictor (consumable directly by
    :func:`evaluate_sp_gates`'s analytic path).
    """
    if seeds is None:
        seeds = list(range(n_models))
    if len(seeds) != n_models:
        raise ValueError(
            f"seeds has {len(seeds)} entries, expected n_models={n_models}"
        )
    members = []
    for seed in seeds:
        fit_dataset = dataset
        if bootstrap_resample:
            train_idx = np.asarray(dataset["splits"]["train"], dtype=int)
            take = np.random.default_rng(seed).choice(
                train_idx, size=len(train_idx), replace=True
            )
            # Member-local dataset: ONLY bootstrap train rows; the trainer
            # touches nothing but splits["train"], so val/test are absent
            # here and remain untouched in the caller's dataset.
            fit_dataset = {
                "fields": np.asarray(dataset["fields"])[take],
                "log_freqs": np.asarray(dataset["log_freqs"])[take],
                "splits": {"train": list(range(len(take))),
                           "val": [], "test": []},
            }
            model = train_field_cvae_heteroscedastic(
                fit_dataset, seed=seed, **kwargs
            )
            model["bootstrap_indices"] = take
        else:
            model = train_field_cvae_heteroscedastic(
                dataset, seed=seed, **kwargs
            )
        members.append(model)
    member_predictors = [m["interval_predictor"] for m in members]

    def interval_predictor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = [pred(X) for pred in member_predictors]
        mu_stack = np.stack([p[0] for p in preds], axis=0)
        sigma_stack = np.stack([p[1] for p in preds], axis=0)
        return _bma_combine(mu_stack, sigma_stack)

    combined = {
        "interval_predictor": interval_predictor,
        "n_models": int(n_models),
        "seeds": [int(s) for s in seeds],
        "bootstrap_resample": bool(bootstrap_resample),
        "member_predictors": member_predictors,
    }
    return {"members": members, "combined": combined}


def evaluate_ensemble_sp_gates(
    ensemble: dict,
    dataset: dict,
    *,
    alpha: float = 0.10,
    seed: int | None = None,
    baseline_epochs: int = 200,
) -> dict:
    """SP-gates for a heteroscedastic ensemble on the **validation split**.

    BMA combination per val observation:
      * predictive mean ``mu_bar = mean_k(mu_k)``;
      * per-pixel total variance by the law of total variance
        ``mean_k(sigma_k^2 + mu_k^2) - mu_bar^2``;
      * SP3 analytic interval ``mu_bar ± z_{alpha/2} * sqrt(total_var)``
        (z_{alpha/2} = 1.645 at alpha = 0.10).

    SBC severity ensembles draw ONE field sample per member
    (``mu_k + sigma_k * eps``), giving a K-sample severity ensemble per
    observation; ranks and gates as in :func:`evaluate_sp_gates`
    (coverage > 0.80; SBC error < 0.10 AND chi-square p > 0.05).

    Point accuracy: BMA mean-field MSE next to a
    :class:`DirectRegressionBaseline` fitted on the train split.

    Returns the evaluate_sp_gates key set plus ``per_member_coverage``
    (each member's own analytic-interval coverage, transparency).
    """
    fields = np.asarray(dataset["fields"], dtype=np.float64)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float64)
    severity = np.asarray(dataset["severity"], dtype=np.float64)
    val_idx = np.asarray(dataset["splits"]["val"], dtype=int)
    if len(val_idx) == 0:
        raise ValueError("validation split is empty")

    rng = np.random.default_rng(seed)
    z_crit = float(_normal.ppf(1.0 - alpha / 2.0))

    if "combined" in ensemble:
        predictors = ensemble["combined"]["member_predictors"]
    else:
        # Duck-typed ensembles: members carrying interval_predictor only.
        predictors = [m["interval_predictor"] for m in ensemble["members"]]
    pred_stack = [predictor(log_freqs[val_idx]) for predictor in predictors]
    mu_stack = np.stack([p[0] for p in pred_stack], axis=0)      # (K,n,gy,gx)
    sigma_stack = np.stack([p[1] for p in pred_stack], axis=0)
    k_members = mu_stack.shape[0]

    mu_bar, total_sigma = _bma_combine(mu_stack, sigma_stack)
    lower, upper = mu_bar - z_crit * total_sigma, mu_bar + z_crit * total_sigma
    truth = fields[val_idx]
    coverage = float(np.mean((truth >= lower) & (truth <= upper)))

    # Transparency: each member judged alone on its own intervals.
    per_member_coverage = [
        float(np.mean(
            (truth >= mu_k - z_crit * sig_k) & (truth <= mu_k + z_crit * sig_k)
        ))
        for mu_k, sig_k in zip(mu_stack, sigma_stack)
    ]

    # SBC: one sample per member -> K-sample severity ensembles.
    severity_ensembles = []
    eps = rng.standard_normal(sigma_stack.shape)
    samples = mu_stack + sigma_stack * eps                     # (K,n,gy,gx)
    for j in range(len(val_idx)):
        severity_ensembles.append(_severity(samples[:, j]))
    ranks = sbc_ranks(severity_ensembles, severity[val_idx])
    sbc_pvalue = sbc_rank_uniformity_pvalue(ranks, n_bins=k_members + 1)
    counts = np.bincount(ranks, minlength=k_members + 1).astype(float)
    props = counts / counts.sum()
    sbc_error = float(np.mean(np.abs(props - 1.0 / (k_members + 1))))

    cvae_mse = float(np.mean(
        [posterior_mean_mse(mu_bar[j], fields[i])
         for j, i in enumerate(val_idx)]
    ))

    baseline = DirectRegressionBaseline(
        d_in=log_freqs.shape[1],
        grid_shape=fields.shape[1:],
    )
    train_idx = np.asarray(dataset["splits"]["train"], dtype=int)
    baseline.fit(
        log_freqs[train_idx],
        fields[train_idx].reshape(len(train_idx), -1),
        epochs=baseline_epochs,
    )
    baseline_preds = baseline.predict(log_freqs[val_idx])
    baseline_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(baseline_preds, val_idx)]
    ))

    return {
        "coverage": coverage,
        "coverage_gate_pass": bool(coverage > SP3_COVERAGE_GATE),
        "per_member_coverage": per_member_coverage,
        "ranks": ranks,
        "sbc_pvalue": sbc_pvalue,
        "sbc_error": sbc_error,
        "sbc_gate_pass": bool(sbc_error < SBC_ERROR_GATE and sbc_pvalue > 0.05),
        "cvae_mse": cvae_mse,
        "baseline_mse": baseline_mse,
        "n_val": int(len(val_idx)),
        "n_models": int(k_members),
        "alpha": alpha,
    }
