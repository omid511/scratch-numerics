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


# ─── Training ────────────────────────────────────────────────────────


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
      * ``kl_anneal_epochs>0`` ramps the KL weight linearly from zero over
        that many ELBO epochs before full weight.

    Only ``dataset["splits"]["train"]`` indices are touched.

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
    x_train = log_freqs[train_idx]
    y_train = fields[train_idx]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    encoder = FreqOnlyEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
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
                    torch.nn.init.normal_(prm, mean=0.0, std=0.01)

    def _epochs(n_epochs, params, step_fn, tag):
        optim = torch.optim.Adam(params, lr=lr)
        history = []
        for epoch in range(n_epochs):
            perm = rng.permutation(len(x_train))
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(x_train), batch_size):
                idx = perm[start:start + batch_size]
                bx = torch.as_tensor(x_train[idx], dtype=torch.float32)
                by = torch.as_tensor(y_train[idx], dtype=torch.float32)
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
                    f"[train_field_cvae] {tag} epoch {epoch + 1}/{n_epochs} "
                    f"loss {history[-1]:.6f}",
                    flush=True,
                )
        return history

    # Phase 1: autoencoder pre-training (z ~ N(0, I)).
    def _ae_step(bx, by):
        c = encoder.forward_tensor(bx)
        cc = c if cond_decoder else torch.zeros(bx.shape[0], 0)
        z = torch.randn(bx.shape[0], decoder.d_z)
        pred = decoder.forward_tensor(z, cc)
        return torch.mean((pred - by) ** 2)

    kl_weight = [1.0 if kl_anneal_epochs == 0 else 0.0]
    step_counter = [0]

    history_ae = _epochs(
        epochs_ae,
        list(encoder.mlp.parameters()) + list(decoder.mlp.parameters()),
        _ae_step,
        "ae",
    )

    # Phase 2: ELBO with free-bits KL (matches train.train_posterior).
    def _elbo_step(bx, by):
        if kl_anneal_epochs > 0:
            kl_weight[0] = min(1.0, (step_counter[0] + 1) / kl_anneal_epochs)
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
        kl = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.5), dim=-1))
        step_counter[0] += 1
        return recon + kl_weight[0] * kl

    history_post = _epochs(
        epochs_post,
        (
            list(encoder.mlp.parameters())
            + list(decoder.mlp.parameters())
            + list(posterior.trunk.parameters())
            + list(posterior.mu_head.parameters())
            + list(posterior.logvar_head.parameters())
        ),
        _elbo_step,
        "elbo",
    )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "posterior": posterior,
        "history": {"ae": history_ae, "posterior": history_post},
        "grid_shape": (gy, gx),
        "n_modes": n_modes,
        "splits": splits,
        "options": {"cond_decoder": cond_decoder,
                    "kl_anneal_epochs": kl_anneal_epochs},
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

    log_freq_row: (n_modes,) → (n_samples, gy, gx) decoded retention fields.
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

    Models is the dict returned by :func:`train_field_cvae` (or any mapping
    exposing ``encoder``/``decoder``/``posterior`` with the same call
    conventions). Gates:

    SP3 (interval coverage): for each val observation draw ``n_samples``
    posterior fields q(z|c)·decoder, form per-pixel central ``(1 - alpha)``
    intervals, and average the fraction of pixels whose true retention lies
    inside. Gate: coverage > 0.80.

    SBC (sample-based): rank of the true mean severity within each val
    observation's posterior-predictive severity ensemble (ties count as "not
    below", standard SBC). Uniformity assessed two ways over the exact
    discrete support {0..n_samples}: a chi-square goodness-of-fit p-value
    (large p ⇒ calibrated) and the calibration error mean(|bin proportion -
    uniform proportion|). Gate: error < 0.10.

    Point accuracy: posterior-mean-field MSE against the truth, reported next
    to a :class:`DirectRegressionBaseline` (MLP log-freqs → field) fitted on
    the train split.

    Returns dict with keys: ``coverage``, ``coverage_gate_pass``, ``ranks``,
    ``sbc_pvalue``, ``sbc_error``, ``sbc_gate_pass``, ``cvae_mse``,
    ``baseline_mse``, ``n_val``, ``alpha``, ``n_samples``.
    """
    encoder = models["encoder"]
    decoder = models["decoder"]
    posterior_m = models["posterior"]

    fields = np.asarray(dataset["fields"], dtype=np.float64)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float64)
    severity = np.asarray(dataset["severity"], dtype=np.float64)
    val_idx = np.asarray(dataset["splits"]["val"], dtype=int)
    if len(val_idx) == 0:
        raise ValueError("validation split is empty")

    rng = np.random.default_rng(seed)
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0

    options = models.get("options", {}) if isinstance(models, dict) else {}
    cond_decoder = bool(options.get("cond_decoder", True))

    covered_pixels = []
    severity_ensembles = []
    mean_field_preds = []
    for i in val_idx:
        ens = _field_ensemble(
            encoder, decoder, posterior_m, log_freqs[i], n_samples, rng,
            cond_decoder=cond_decoder,
        )
        lower = np.quantile(ens, lower_q, axis=0)
        upper = np.quantile(ens, upper_q, axis=0)
        truth = fields[i]
        covered_pixels.append(float(np.mean((truth >= lower) & (truth <= upper))))
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
