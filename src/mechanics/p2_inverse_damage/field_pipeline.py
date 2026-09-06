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
    sbc_rank_uniformity_pvalue_exact,
)
from .posterior import ConditionalPosterior

# Gate threshold from the P2 charter (SP-gate evaluation).
SP3_COVERAGE_GATE = 0.80   # SP3: 90% per-pixel interval coverage must exceed 0.80

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
        if "meas_mode_shapes" in data.files:
            mode_shapes = np.asarray(
                data["meas_mode_shapes"], dtype=np.float64
            )
        elif "mode_shapes" in data.files:
            mode_shapes = np.asarray(data["mode_shapes"], dtype=np.float64)
        else:
            mode_shapes = None
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
    if freqs.ndim != 2:
        raise ValueError(f"meas_frequencies must be (n, n_modes); got {freqs.shape}")
    if severity.ndim != 1:
        raise ValueError(f"meas_severity must be (n,); got {severity.shape}")
    gy, gx = fields.shape[1], fields.shape[2]
    n_modes = freqs.shape[1]
    overlap = (set(splits["train"]) & set(splits["val"]))
    overlap |= set(splits["train"]) & set(splits["test"])
    overlap |= set(splits["val"]) & set(splits["test"])
    if overlap:
        raise ValueError(f"splits overlap at indices {sorted(overlap)}")
    # Index universe: every row belongs to exactly one split (design-level
    # discipline); catches dropped/duplicated/out-of-range rows at load.
    for name in ("train", "val", "test"):
        idx = list(splits[name])
        if len(set(idx)) != len(idx):
            raise ValueError(f"split {name!r} contains duplicate indices")
        bad = [i for i in idx if not isinstance(i, (int, np.integer)) or i < 0 or i >= n]
        if bad:
            raise ValueError(f"split {name!r} has out-of-range indices: {bad[:8]}")
    universe = sorted(splits["train"] + splits["val"] + splits["test"])
    if universe != list(range(n)):
        raise ValueError(
            f"splits must partition all {n} rows exactly once; "
            f"union covers {len(universe)} entries"
        )

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
        if summaries.ndim != 2 or summaries.shape[1] % n_modes != 0:
            raise ValueError(
                f"summaries must be (n, n_modes*d_block); got {summaries.shape}"
            )
        out["summaries"] = summaries
    if mode_shapes is not None:
        if mode_shapes.shape[0] != n:
            raise ValueError(
                f"sample-count mismatch: fields {n}, "
                f"mode_shapes {mode_shapes.shape}"
            )
        if mode_shapes.ndim != 4 or mode_shapes.shape[1] != n_modes:
            raise ValueError(
                f"mode_shapes must be (n, n_modes, gy, gx); got {mode_shapes.shape}"
            )
        if (mode_shapes.shape[2], mode_shapes.shape[3]) != (gy, gx):
            raise ValueError(
                f"mode_shapes grid {mode_shapes.shape[2:]} disagrees with "
                f"fields grid {(gy, gx)}"
            )
        out["mode_shapes"] = mode_shapes
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

        gen = torch.Generator().manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, generator=gen)
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

        gen = torch.Generator().manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, generator=gen)
                nn.init.zeros_(m.bias)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_modes + n_modes*d_summary_block) → c: (B, d_c)."""
        return self.mlp(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """NumPy inference path. (batch, d_in) → (batch, d_c)."""
        x_t = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.mlp(x_t).numpy()


class ModeShapeCNNEncoder(nn.Module):
    """Encode stacked mode shapes (B, n_modes, gy, gx) → c ∈ R^{d_c}.

    Small conv trunk: Conv2d(n_modes→16→32→16, 3x3, stride 2, pad 1)
    with ReLU, then a Linear head to d_c (built lazily on the first
    forward so any gy×gx works). Input rows are the per-sample mode
    shapes evaluated on the FIELD grid.
    """

    def __init__(self, n_modes: int = 6, d_c: int = 64, seed: int = 42):
        super().__init__()
        self.n_modes = n_modes
        self.d_c = d_c
        self.seed = int(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(n_modes, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc: nn.Linear | None = None
        gen = torch.Generator().manual_seed(seed)
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, generator=gen)
                nn.init.zeros_(m.bias)

    def _ensure_fc(self, h: torch.Tensor) -> None:
        if self.fc is None:
            flat = h.shape[1] * h.shape[2] * h.shape[3]
            self.fc = nn.Linear(flat, self.d_c).to(h.device)
            gen0 = torch.Generator().manual_seed(self.seed)
            nn.init.kaiming_normal_(self.fc.weight, generator=gen0)
            nn.init.zeros_(self.fc.bias)

    def forward_tensor(self, shapes: torch.Tensor) -> torch.Tensor:
        """shapes: (B, n_modes, gy, gx) → c: (B, d_c), grad-preserving."""
        h = self.conv(shapes)
        self._ensure_fc(h)
        return self.fc(h.flatten(1))

    def forward(self, shapes: np.ndarray) -> np.ndarray:
        """NumPy inference path. (batch, n_modes, gy, gx) → (batch, d_c)."""
        x_t = torch.as_tensor(shapes, dtype=torch.float32)
        with torch.no_grad():
            return self.forward_tensor(x_t).numpy()


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


def cnn_regression_probe(
    dataset: dict,
    *,
    d_c: int = 64,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Learnability anchor: direct CNN mode-shapes -> field regression.

    Trains the :class:`ModeShapeCNNEncoder` trunk plus a linear head to
    map canonicalized stacked mode shapes ``(n, n_modes, gy, gx)`` straight
    to retention fields by MSE on the train split — no latent, no ELBO.
    A few hundred steps suffice: if this probe cannot beat the
    frequencies-only reference (~0.0317 MSE), the shapes carry no
    exploitable signal and the CVAE route is hopeless; if it can
    (ridge-on-canonicalized-shapes reaches ~0.0248), any remaining gap is
    a pipeline problem, not a data problem.

    Requires ``dataset['mode_shapes']`` (canonicalized at emission since
    the sign/scale fix in :mod:`mechanics.p2_inverse_damage.damage_data`).
    Returns ``{"val_mse": ..., "train_mse": ..., "epochs": ...}``.
    """
    if "mode_shapes" not in dataset:
        raise ValueError("cnn_regression_probe requires dataset['mode_shapes']")
    fields = np.asarray(dataset["fields"], dtype=np.float32)
    shapes = np.asarray(dataset["mode_shapes"], dtype=np.float32)
    splits = dataset["splits"]
    train_idx = np.asarray(splits["train"], dtype=int)
    val_idx = np.asarray(splits["val"], dtype=int)
    gy, gx = fields.shape[-2], fields.shape[-1]

    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(seed)
    encoder = ModeShapeCNNEncoder(n_modes=shapes.shape[1], d_c=d_c, seed=seed)
    # Linear head built eagerly so its parameters are known before training.
    encoder._ensure_fc(encoder.conv(torch.zeros(1, shapes.shape[1], gy, gx)))
    head = nn.Linear(d_c, gy * gx)
    nn.init.kaiming_normal_(head.weight, generator=tgen)
    nn.init.zeros_(head.bias)
    params = (
        list(encoder.parameters()) + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)
    x_tr = torch.as_tensor(shapes[train_idx])
    y_tr = torch.as_tensor(fields[train_idx].reshape(-1, gy * gx))
    for epoch in range(epochs):
        perm = rng.permutation(len(x_tr))
        for s in range(0, len(x_tr), batch_size):
            b = perm[s:s + batch_size]
            pred = head(encoder.forward_tensor(x_tr[b]))
            loss = torch.mean((pred - y_tr[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def _mse(idx):
        with torch.no_grad():
            pred = head(
                encoder.forward_tensor(torch.as_tensor(shapes[idx]))
            ).numpy().reshape(-1, gy, gx)
        return float(np.mean((pred - fields[idx]) ** 2))

    return {"train_mse": _mse(train_idx), "val_mse": _mse(val_idx),
            "epochs": epochs}

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
    kl_beta: float = 1.0,
    free_bits: bool = True,
    use_summaries: bool = False,
    posterior_init_std: float = 0.5,
    conditioning: str = "freq_only",
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
        0.5 default starts the per-dim KL above the free-bits floor with
        a live gradient. (The 0.01 default put mu/logvar outputs inside
        the clamp (kl_per_dim < 0.5 everywhere) from step zero, so
        ``torch.clamp`` passed zero gradient and the posterior could never
        escape collapse; the pre-default failure was the opposite extreme:
        std=default gave mu~O(10), KL~2.7e4 nats at step zero.)
      * ``kl_beta`` scales the KL term after per-latent-dim normalization:
        the summed KL is divided by ``d_z`` so a mean-MSE reconstruction
        (~1e-3) is not drowned by the summed free-bits floor (~0.5*d_z).
        ``loss = recon + kl_weight * kl_beta * mean(sum(kl_per_dim)/d_z)``.
        Pass ``kl_beta=d_z`` to recover the legacy unnormalized scale.
      * ``conditioning='cnn'`` encodes the FULL mode shapes
        ``(n, n_modes, gy, gx)`` (field-grid aligned; requires
        ``dataset['mode_shapes']``) through :class:`ModeShapeCNNEncoder`;
        ``'freq_only'`` (default) and ``'freq_summary'`` preserve the
        previous behavior. ``use_summaries=True`` is a legacy alias for
        ``conditioning='freq_summary'``.

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

    use_summaries = use_summaries or conditioning == "freq_summary"
    cond = "freq_summary" if use_summaries else (
        "cnn" if conditioning == "cnn" else "freq_only"
    )
    n, gy, gx = fields.shape
    n_modes = log_freqs.shape[1]
    if cond == "freq_summary":
        if "summaries" not in dataset:
            raise ValueError("summaries conditioning requires dataset['summaries']")
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
    elif cond == "cnn":
        if "mode_shapes" not in dataset:
            raise ValueError("cnn conditioning requires dataset['mode_shapes']")
        mode_shapes = np.asarray(dataset["mode_shapes"], dtype=np.float32)
        if mode_shapes.ndim != 4 or mode_shapes.shape[0] != n:
            raise ValueError(
                f"mode_shapes must be (n, n_modes, gy, gx); got {mode_shapes.shape}"
            )
        x_train = mode_shapes[train_idx]
        encoder = ModeShapeCNNEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
    else:
        x_train = log_freqs[train_idx]
        encoder = FreqOnlyEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
    y_train = fields[train_idx]

    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(seed)

    # The decoder ALWAYS carries the d_c conditioning block: Phase 1 must
    # feed it the real measurement-derived c (see _ae_step below). With
    # cond_decoder=False only the Phase-2 ELBO de-conditions.
    decoder = DamageDecoder(d_z=d_z, d_c=d_c,
                            grid_size=(gy, gx), seed=seed)
    posterior = ConditionalPosterior(d_z=d_z, d_c=d_c, seed=seed)
    # Small-variance re-init of the posterior heads: their default init
    # produced mu~O(10) against the N(0,I) prior -> KL ~2.7e4 nats at step
    # zero, drowning the reconstruction term (empirically measured).
    for head in (posterior.trunk, posterior.mu_head, posterior.logvar_head):
        if hasattr(head, "parameters"):
            for prm in head.parameters():
                if prm.requires_grad and prm.ndim > 1:
                    torch.nn.init.normal_(prm, mean=0.0, std=posterior_init_std, generator=tgen)


    # Phase 1: autoencoder pre-training (z ~ N(0, I)).
    def _ae_step(bx, by):
        # Phase 1 trains encoder + decoder NORMALLY with the real
        # measurement-derived c: the reconstruction path c -> decoder(z, c)
        # is the ONLY route by which a CNN shape encoder receives live
        # gradient here. De-conditioning is an anti-collapse device for
        # Phase 2 (force information through z); applying it in Phase 1 as
        # well left the encoder with zero gradient and its features random.
        c = encoder.forward_tensor(bx)
        z = torch.randn(bx.shape[0], decoder.d_z, generator=tgen)
        pred = decoder.forward_tensor(z, c)
        return torch.mean((pred - by) ** 2)

    comps: list[list[float]] = []
    if not free_bits:
        kl_weight = [kl_weight_final]
    else:
        kl_weight = [kl_weight_final if kl_anneal_epochs == 0 else 0.0]
    step_counter = [0]

    # Eagerly materialize the CNN encoder's lazy FC head BEFORE the
    # optimizer parameter snapshot — _ensure_fc is self-guarded (idempotent),
    # so call it unconditionally. (A previous guard keyed on an _fc_built
    # attribute that is never set made this a no-op: the FC readout trained
    # frozen at random init through entire runs — review finding.)
    if hasattr(encoder, "_ensure_fc"):
        encoder._ensure_fc(
            encoder.conv(torch.zeros(
                1, x_train.shape[1], gy, gx,
            ))
        )
    enc_params = (
        list(encoder.mlp.parameters())
        if hasattr(encoder, "mlp") else list(encoder.parameters())
    )
    history_ae = _run_epochs(
        epochs_ae, x_train, y_train,
        enc_params + list(decoder.mlp.parameters()),
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
        # Asymmetry rationale: de-conditioning ONLY in Phase 2 forces all
        # measurement information through z (anti-collapse remedy), while
        # Phase 1 already shaped the encoder/decoder pair with live c
        # gradients. The zeros block matches the decoder's d_c-wide input.
        cc = (
            c if cond_decoder
            else torch.zeros(bx.shape[0], d_c)
        )
        mu, log_var = posterior.forward_tensor(c)
        # Bounded log-variance: an unbounded branch lets KL explode through
        # exp(log_var) early in training and stall the ELBO (observed ~11k
        # nats flat). +-10 nats is far beyond any calibrated posterior here.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu, generator=tgen)
        pred = decoder.forward_tensor(z, cc)
        recon = torch.mean((pred - by) ** 2)
        kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
        if free_bits:
            kl_per_dim = torch.clamp(kl_per_dim, min=0.5)
        # Per-latent-dim mean so recon (mean MSE) and KL share a scale;
        # kl_beta restores the legacy summed scale when set to d_z.
        kl = torch.mean(torch.sum(kl_per_dim, dim=-1) / mu.shape[-1])
        comps.append([float(recon.detach()), float(kl.detach())])
        step_counter[0] += 1
        return recon + kl_weight[0] * kl_beta * kl

    history_post = _run_epochs(
        epochs_post, x_train, y_train,
        (
            enc_params
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
                    "conditioning": cond,
                    "kl_anneal_epochs": kl_anneal_epochs,
                    "use_summaries": use_summaries,
                    "posterior_init_std": posterior_init_std,
                    "kl_weight_final": kl_weight_final,
                    "kl_beta": kl_beta,
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
        # Deliberately FROZEN at init (gamma=1, beta=0): this norm is a
        # fixed feature stabilizer, not a learned component. Its affine
        # parameters are excluded from every optimizer parameter list by
        # construction (requires_grad_(False)), closing the audit finding
        # that LayerNorm affine params silently never trained anywhere
        # (I3-class failure mode). Recorded as decision D-row in
        # RESEARCH_LOG.md. If per-feature scale adaptation is ever wanted,
        # make it an explicit measured lever, not a silent side effect.
        self.trunk_norm.weight.requires_grad_(False)
        self.trunk_norm.bias.requires_grad_(False)
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

        gen = torch.Generator().manual_seed(seed)
        for module in (self.trunk, self.mu_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, generator=gen)
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.sigma_head[0].weight, generator=gen)
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
    kl_beta: float = 1.0,
    sigma_range: tuple[float, float] = (0.005, 0.5),
    use_summaries: bool = False,
    posterior_init_std: float = 0.5,
    conditioning: str = "freq_only",
    sigma_calibration: str = "none",
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
        linear anneal) is identical to :func:`train_field_cvae`, including
        per-latent-dim normalization (``kl_beta``) and the 0.5 default
        ``posterior_init_std`` that starts above the free-bits floor.
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
    cond = (
        "freq_summary" if use_summaries else
        ("cnn" if conditioning == "cnn" else "freq_only")
    )
    if cond == "freq_summary":
        if "summaries" not in dataset:
            raise ValueError("summaries conditioning requires dataset['summaries']")
        summaries = np.asarray(dataset["summaries"], dtype=np.float32)
        x_train = np.concatenate(
            [log_freqs[train_idx], summaries[train_idx]], axis=1
        )
    elif cond == "cnn":
        if "mode_shapes" not in dataset:
            raise ValueError("cnn conditioning requires dataset['mode_shapes']")
        mode_shapes = np.asarray(dataset["mode_shapes"], dtype=np.float32)
        if mode_shapes.ndim != 4 or mode_shapes.shape[0] != n:
            raise ValueError(
                f"mode_shapes must be (n, n_modes, gy, gx); got {mode_shapes.shape}"
            )
        x_train = mode_shapes[train_idx]
    else:
        x_train = log_freqs[train_idx]
    y_train = fields[train_idx]
    log_sigma_min = float(np.log(sigma_range[0]))
    log_sigma_max = float(np.log(sigma_range[1]))

    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(seed)

    encoder = (
        FreqSummaryEncoder(
            n_modes=n_modes,
            d_c=d_c,
            d_summary_block=summaries.shape[1] // n_modes,
            seed=seed,
        )
        if cond == "freq_summary"
        else (
            ModeShapeCNNEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
            if cond == "cnn"
            else FreqOnlyEncoder(n_modes=n_modes, d_c=d_c, seed=seed)
        )
    )
    # Same Phase-1/Phase-2 asymmetry as train_field_cvae: the decoder
    # always carries the d_c block; Phase 1 feeds it the real c, only the
    # Phase-2 ELBO de-conditions when cond_decoder=False.
    decoder = HeteroscedasticFieldDecoder(d_z=d_z,
                                          d_c=d_c,
                                          grid_size=(gy, gx),
                                          sigma_range=sigma_range, seed=seed)
    posterior = ConditionalPosterior(d_z=d_z, d_c=d_c, seed=seed)
    # Same small-variance posterior re-init as train_field_cvae: default
    # heads put KL ~O(10^4) nats at step zero and drown the NLL term.
    for head in (posterior.trunk, posterior.mu_head, posterior.logvar_head):
        if hasattr(head, "parameters"):
            for prm in head.parameters():
                if prm.requires_grad and prm.ndim > 1:
                    torch.nn.init.normal_(prm, mean=0.0, std=posterior_init_std, generator=tgen)

    # Phase 1: autoencoder pre-training on the mean head (z ~ N(0, I)).
    def _ae_step(bx, by):
        c = encoder.forward_tensor(bx)
        # Phase 1 uses the REAL measurement-derived c (live encoder
        # gradient); de-conditioning applies only in Phase 2.
        cc = c
        z = torch.randn(bx.shape[0], decoder.d_z, generator=tgen)
        pred_mu, _ = decoder.forward_tensor(z, cc)
        return torch.mean((pred_mu - by) ** 2)

    # Eagerly materialize the CNN encoder's lazy FC head BEFORE the
    # optimizer parameter snapshot — _ensure_fc is self-guarded (idempotent),
    # so call it unconditionally. (A previous guard keyed on an _fc_built
    # attribute that is never set made this a no-op: the FC readout trained
    # frozen at random init through entire runs — review finding.)
    if hasattr(encoder, "_ensure_fc"):
        encoder._ensure_fc(
            encoder.conv(torch.zeros(
                1, x_train.shape[1], gy, gx,
            ))
        )
    enc_params = (
        list(encoder.mlp.parameters())
        if hasattr(encoder, "mlp") else list(encoder.parameters())
    )
    kl_weight = [1.0 if kl_anneal_epochs == 0 else 0.0]
    step_counter = [0]


    history_ae = _run_epochs(
        epochs_ae, x_train, y_train,
        (enc_params
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
        cc = (
            c if cond_decoder
            else torch.zeros(bx.shape[0], d_c)
        )
        mu, log_var = posterior.forward_tensor(c)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        pred_mu, _ = decoder.forward_tensor(mu, cc)
        recon = torch.mean((pred_mu - by) ** 2)
        kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
        kl = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.5), dim=-1) / mu.shape[-1])
        comps.append([float(recon.detach()), float(kl.detach())])
        step_counter[0] += 1
        return recon + kl_weight[0] * kl_beta * kl

    comps: list[list[float]] = []

    _main_params = (
        enc_params
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
            cc = (
                c if cond_decoder
                else torch.zeros(bx.shape[0], d_c)
            )
            mu, _ = posterior.forward_tensor(c)
            pred_mu, _ = decoder.forward_tensor(mu, cc)
            x = torch.cat([mu, cc], dim=-1) if cc.shape[1] else mu
            h = decoder.trunk_norm(decoder.trunk(x))  # match deployment path
        sigma, log_sigma = decoder.sigma_tensor(h.detach())
        return torch.mean(
            log_sigma + (by - pred_mu) ** 2 / (2.0 * sigma**2)
        )

    history_post += _run_epochs(
        max(0, epochs_post - epochs_mu), x_train, y_train,
        list(decoder.sigma_head.parameters()),
        _nll_sigma_step,
        batch_size=batch_size, lr=0.1 * lr, rng=rng, progress=progress,
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
        cc = c_np if cond_decoder else np.zeros((len(lf), d_c))
        mu_z, log_var = posterior(c_np)
        mu_grid, log_sigma_grid = decoder.forward(mu_z, cc)
        sigma_grid = np.exp(log_sigma_grid)   # already bounded by construction

        return mu_grid, sigma_grid
    # Optional in-training sigma calibration: fit the same leave-one-out
    # conformal multiplier the SP-gate path applies post-hoc, but on the
    # TRAIN split so the deployed interval_predictor is self-calibrated
    # (evaluate_sp_gates scales its analytic intervals by this scalar).
    sigma_multiplier = None
    if sigma_calibration == "conformal":
        mu_tr, sig_tr = interval_predictor(x_train)
        pred_sev = 1.0 - mu_tr.mean(axis=(1, 2))
        n_pix = mu_tr.shape[1] * mu_tr.shape[2]
        stat_std = np.sqrt((sig_tr ** 2).sum(axis=(1, 2))) / n_pix
        truth_sev = np.asarray(
            dataset["severity"], dtype=np.float64)[train_idx]
        ks = _conformal_severity_multipliers(
            pred_sev, truth_sev, stat_std)
        sigma_multiplier = float(np.median(ks))
    elif sigma_calibration != "none":
        raise ValueError(
            f"unknown sigma_calibration {sigma_calibration!r}; "
            "expected 'none' or 'conformal'"
        )

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
                    "kl_beta": kl_beta,
                    "sigma_range": sigma_range,
                    "use_summaries": use_summaries,
                    "posterior_init_std": posterior_init_std,
                    "conditioning": cond},
        "interval_predictor": interval_predictor,
        "sigma_multiplier": sigma_multiplier,
    }


# ─── SP-gate evaluation ──────────────────────────────────────────────

#: Margin below the retention ceiling (1.0) within which a pristine cell
#: counts as covered. Sigmoid-field decoders saturate below 1.0, so exact
#: containment can never cover pristine truth; the margin (well inside the
#: 0.10 gap between the deepest damaged retention 0.9 and pristine 1.0)
#: keeps that comparison honest without touching damaged-cell scoring.
PRISTINE_BOUNDARY_TOL = 0.01

#: Minimum ensemble size for the SBC uniformity gate to be meaningful.
#: Below this the exact-MC p-value has almost no power and the gate is
#: reported invalid instead of passed.
SBC_MIN_SAMPLES = 20


def _boundary_aware_covered(
    lower: np.ndarray, upper: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """Coverage mask with explicit retention-ceiling handling.

    Cells inside their intervals count as covered; pristine cells
    (truth == 1.0, unreachable for sigmoid outputs) additionally count
    when the interval upper endpoint reaches the ceiling margin.
    Damaged cells always use exact containment.
    """
    inside = (np.asarray(truth) >= np.asarray(lower)) & (
        np.asarray(truth) <= np.asarray(upper)
    )
    pristine = np.asarray(truth) >= 1.0
    return inside | (
        pristine & (np.asarray(upper) >= 1.0 - PRISTINE_BOUNDARY_TOL)
    )


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
        # De-conditioned decoder still has a d_c-wide input block
        # (Phase 1 trains it with real c); feed structural zeros.
        dec_in_cond = np.zeros((n_samples, getattr(decoder, "d_c", c.shape[-1])))
    return np.asarray(decoder.forward(z, dec_in_cond))                  # (S, gy, gx)


def _severity(field_stack: np.ndarray) -> np.ndarray:
    """Per-sample mean-severity of stacked fields (S, gy, gx) → (S,)."""
    return 1.0 - field_stack.mean(axis=(-2, -1))


def _conformal_severity_multipliers(
    pred_severity: np.ndarray,
    truth_severity: np.ndarray,
    stat_std: np.ndarray,
    min_obs: int = 8,
) -> np.ndarray:
    """Leave-one-out conformal variance multipliers for the mean-severity
    statistic.

    The analytic SP-gate path draws severity ensembles as
    ``mu + sigma * eps`` with eps INDEPENDENT across pixels, so the
    implied spread of the joint mean-severity statistic is
    ``stat_std = sqrt(sum(sigma^2)) / n_pixels``. When pixel residuals
    are spatially correlated (or sigma collapses on hard observations)
    that spread understates the actual residual scale and SBC ranks pile
    up at {0, n_samples} — observed on data/p2/fields_cnn_ds.npz as a
    rank U-shape with per-pixel coverage still nominal.

    Remedy (conformal-style calibration fitted on the validation split
    ONLY — never test): for observation j draw the multiplier from the
    OTHER val observations' standardized mean-statistic residuals,

        k_j = sqrt( mean_{l != j} [ ((y_l - mu_stat_l) / stat_std_l)^2 ] )

    and resample its ensemble at ``sigma * k_j``. Leave-one-out keeps
    every rank honest: no observation's own residual inflates the scale
    it is judged against. Falls back to 1.0 (no recalibration) when the
    val split is too small to estimate a scale (< min_obs) or when any
    input is degenerate. A floor of 1e-6 on k^2 prevents zero-width
    ensembles, which would degenerate ranks via ties.
    """
    n = len(truth_severity)
    if (
        n < min_obs
        or not np.all(np.isfinite(pred_severity))
        or not np.all(np.isfinite(truth_severity))
        or not np.all(np.isfinite(stat_std))
        or np.any(stat_std <= 0.0)
    ):
        return np.ones(n)
    z = (np.asarray(truth_severity, dtype=np.float64)
         - np.asarray(pred_severity, dtype=np.float64)) / np.asarray(
            stat_std, dtype=np.float64)
    loo_ms = (float(z @ z) - z ** 2) / (n - 1)
    return np.sqrt(np.maximum(loo_ms, 1e-6))


def evaluate_sp_gates(
    models: dict,
    dataset: dict,
    *,
    n_samples: int = 50,
    alpha: float = 0.10,
    seed: int | None = None,
    baseline_epochs: int = 200,
    split: str = "val",
) -> dict:
    """Evaluate the P2 SP-gates on a chosen dataset split (default ``val``).

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

    SP3 (interval coverage, STRATIFIED): per-pixel central ``(1 - alpha)``
    intervals — quantiled ensembles on the ensemble path, analytic normal
    bounds on the analytic path. Coverage is reported separately for
    PRISTINE cells (true retention ≡ 1.0) and DAMAGED cells (truth < 1.0),
    plus the pooled value. The gate binds on DAMAGED-cell coverage only:
    pooling with pristine cells masks damage-cell miscalibration — the pooling artifact that let a
    2x-too-narrow-on-damage model pass at pooled coverage > 0.9. When the
    split contains no damaged cells the gate falls back to pooled coverage.

    SBC / SP4 (sample-based, BOTH paths): rank of the true mean severity
    within each val observation's severity ensemble of ``n_samples`` draws —
    decoded posterior fields q(z|c)·decoder on the ensemble path,
    ``mu + sigma * eps`` per pixel on the analytic path (ties count as "not
    below", standard SBC). The SOLE criterion is the Monte-Carlo EXACT
    uniformity p-value over the discrete support {0..n_samples}
    (:func:`sbc_rank_uniformity_pvalue_exact`): with sparse bins the
    asymptotic chi-square reference is invalid, and the mean |bin proportion
    - uniform| error statistic is vacuous as a gate (its null level ~0.009
    sits an order of magnitude under the old 0.10 threshold, so it could
    never fail). Gate: exact p-value > 0.05. ``sbc_error`` is still
    reported as a descriptive statistic only.

    NO post-hoc recalibration: severity ensembles are drawn at raw sigma.
    The earlier leave-one-out conformal multiplier consumed ground-truth
    severities of the evaluated split itself — on split="test" that is
    test-label leakage, and it rescued exactly the sigma miscalibration the
    SP4 gate exists to detect. The raw uncalibrated result is the honest
    one. In-training calibration fitted on TRAIN rows only
    (``models["sigma_multiplier"]``) remains part of the deployed model and
    still scales the analytic intervals.

    Point accuracy: mean-field MSE against the truth (ensemble mean or mu),
    reported next to a :class:`DirectRegressionBaseline` (MLP log-freqs →
    field) fitted on the train split, and next to a constant-field baseline
    (train-set mean field) that any trained model must beat.

    Returns dict with keys: ``coverage``, ``coverage_pristine``,
    ``coverage_damaged``, ``coverage_gate_pass``, ``ranks``,
    ``sbc_pvalue``, ``sbc_error``, ``sbc_gate_pass``, ``cvae_mse``,
    ``baseline_mse``, ``constant_field_mse``, ``n_val``, ``alpha``,
    ``n_samples``.

    ``split`` selects which entry of ``dataset["splits"]`` the gates are
    computed on. Audit use: pass ``split="test"`` on a model trained ONLY
    on the train rows — the caller is responsible for split discipline;
    nothing here prevents evaluating on a split the model saw.
    """
    predictor = models.get("interval_predictor")

    fields = np.asarray(dataset["fields"], dtype=np.float64)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float64)
    severity = np.asarray(dataset["severity"], dtype=np.float64)
    if split not in dataset["splits"]:
        raise ValueError(
            f"unknown split {split!r}; available: {sorted(dataset['splits'])}"
        )
    eval_idx = np.asarray(dataset["splits"][split], dtype=int)
    if len(eval_idx) == 0:
        raise ValueError(f"{split!r} split is empty")

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
    if options.get("conditioning") == "cnn":
        if "mode_shapes" not in dataset:
            raise ValueError("cnn conditioning requires dataset['mode_shapes']")
        X = np.asarray(dataset["mode_shapes"], dtype=np.float64)
    elif use_summaries:
        if "summaries" not in dataset:
            raise ValueError("use_summaries=True requires dataset['summaries']")
        X = np.concatenate(
            [log_freqs,
             np.asarray(dataset["summaries"], dtype=np.float64)],
            axis=1,
        )

    covered_maps = []
    severity_ensembles = []
    mean_field_preds = []
    if predictor is not None:
        mu_val, sigma_val = predictor(X[eval_idx])
        # In-training calibration (train_field_cvae_heteroscedastic with
        # sigma_calibration='conformal'): scale sigma by the multiplier
        # fitted on the train split BEFORE computing analytic SP3
        # intervals and SBC ensembles — conformal moves into training.
        trained_mult = models.get("sigma_multiplier")
        if trained_mult is not None:
            sigma_val = sigma_val * float(trained_mult)
        lower = mu_val - z_crit * sigma_val
        upper = mu_val + z_crit * sigma_val
        truth = fields[eval_idx]
        covered_map = _boundary_aware_covered(
            lower, upper, truth)   # (n, gy, gx)
        # SBC severity ensembles at RAW sigma — no post-hoc recalibration.
        # The removed LOO conformal multiplier consumed ground-truth
        # severities of the evaluated split (test-label leakage on
        # split="test") and rescued exactly the sigma miscalibration SP4
        # exists to detect.
        for j in range(len(eval_idx)):
            eps = rng.standard_normal((n_samples, *mu_val[j].shape))
            severity_ensembles.append(_severity(
                mu_val[j][np.newaxis] + sigma_val[j][np.newaxis] * eps))
        mean_field_preds = list(mu_val)

    else:
        encoder = models["encoder"]
        decoder = models["decoder"]
        posterior_m = models["posterior"]
        for i in eval_idx:
            ens = _field_ensemble(
                encoder, decoder, posterior_m, X[i], n_samples, rng,
                cond_decoder=cond_decoder,
            )
            lower = np.quantile(ens, lower_q, axis=0)
            upper = np.quantile(ens, upper_q, axis=0)
            truth = fields[i]
            covered_maps.append(_boundary_aware_covered(lower, upper, truth))
            severity_ensembles.append(_severity(ens))
            mean_field_preds.append(ens.mean(axis=0))

    if predictor is None:
        covered_map = np.asarray(covered_maps)
    truth_eval = fields[eval_idx]
    pristine = truth_eval >= 1.0
    damaged = ~pristine
    coverage = float(covered_map.mean())
    coverage_pristine = (
        float(covered_map[pristine].mean()) if pristine.any()
        else float("nan")
    )
    coverage_damaged = (
        float(covered_map[damaged].mean()) if damaged.any()
        else float("nan")
    )
    # SP3 binds on DAMAGED cells only: pooled coverage hides damage-side
    # miscalibration. Pristine scoring is boundary-aware (see
    # _boundary_aware_covered): sigmoid outputs saturate below the 1.0
    # ceiling, so exact containment alone would pin pristine coverage at
    # zero. No damaged cells -> fall back to pooled coverage.
    gate_coverage = coverage_damaged if damaged.any() else coverage

    ranks = sbc_ranks(severity_ensembles, severity[eval_idx])
    # SP4 sole criterion: the finite-sample-valid Monte-Carlo EXACT
    # uniformity p-value (the asymptotic chi-square is invalid at these
    # bin counts).
    sbc_pvalue = sbc_rank_uniformity_pvalue_exact(ranks, n_samples + 1)
    sbc_gate_valid = bool(n_samples >= SBC_MIN_SAMPLES)

    # Descriptive calibration error only — vacuous as a gate (its null
    # level ~0.009 sits an order of magnitude under the old 0.10 threshold).
    counts = np.bincount(ranks, minlength=n_samples + 1).astype(float)
    props = counts / counts.sum()
    sbc_error = float(np.mean(np.abs(props - 1.0 / (n_samples + 1))))

    cvae_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(mean_field_preds, eval_idx)]
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
    baseline_preds = baseline.predict(log_freqs[eval_idx])
    baseline_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(baseline_preds, eval_idx)]
    ))

    # Constant-field baseline: predict the train-set mean field for every
    # input. Any trained model must beat this to demonstrate skill.
    const_field = fields[train_idx].mean(axis=0)
    const_mse = float(np.mean(
        [posterior_mean_mse(const_field, fields[i]) for i in eval_idx]
    ))

    return {
        "constant_field_mse": const_mse,
        "coverage": coverage,
        "coverage_pristine": coverage_pristine,
        "coverage_damaged": coverage_damaged,
        "coverage_gate_pass": bool(gate_coverage > SP3_COVERAGE_GATE),
        "ranks": ranks,
        "sbc_pvalue": sbc_pvalue,
        "sbc_error": sbc_error,
        "sbc_gate_valid": sbc_gate_valid,
        "sbc_gate_pass": bool(sbc_gate_valid and sbc_pvalue > 0.05),
        "cvae_mse": cvae_mse,
        "baseline_mse": baseline_mse,
        "n_val": int(len(eval_idx)),
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


def _resolve_eval_inputs(dataset: dict, options: dict) -> np.ndarray:
    """Rebuild the model input rows an evaluator must feed predictors.

    Mirrors the conditioning switch in the trainers and
    :func:`evaluate_sp_gates`: ``cnn`` → stacked mode shapes,
    ``freq_summary``/``use_summaries`` → concat(log_freqs, summaries),
    otherwise plain log-frequencies. Raises ValueError when the required
    channel is absent from ``dataset``.
    """
    options = options or {}
    if options.get("conditioning") == "cnn":
        if "mode_shapes" not in dataset:
            raise ValueError("cnn conditioning requires dataset['mode_shapes']")
        return np.asarray(dataset["mode_shapes"], dtype=np.float64)
    if bool(options.get("use_summaries", False)):
        if "summaries" not in dataset:
            raise ValueError("use_summaries=True requires dataset['summaries']")
        return np.concatenate(
            [np.asarray(dataset["log_freqs"], dtype=np.float64),
             np.asarray(dataset["summaries"], dtype=np.float64)],
            axis=1,
        )
    return np.asarray(dataset["log_freqs"], dtype=np.float64)


def _decode_fields(decoder, z: np.ndarray, cc) -> np.ndarray:
    """Decode latent samples to fields for either decoder flavor.

    :class:`DamageDecoder` returns field grids directly while
    :class:`HeteroscedasticFieldDecoder` returns ``(mu, log_sigma)`` —
    the mu grids are the decoded fields in both cases.
    """
    out = decoder.forward(z, cc)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float64)


def _latent_severity_ensemble(
    member: dict,
    x_row: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    cond_decoder: bool = True,
) -> np.ndarray:
    """Severity ensemble via posterior-latent decoded fields.

    Draws ``z ~ q(z|c)`` and decodes full fields, preserving the learned
    spatial correlation. The analytic alternative (``mu + sigma*eps`` with
    independent per-pixel noise) understates the mean-severity spread
    whenever pixel residuals are correlated, so SBC ranks pile at the
    extremes even when per-pixel coverage is nominal.
    """
    encoder, decoder, posterior = (
        member["encoder"], member["decoder"], member["posterior"])
    c = np.asarray(encoder.forward(np.expand_dims(x_row, axis=0)))
    mu, log_var = posterior(c)
    eps = rng.standard_normal((n_samples, mu.shape[-1]))
    z = mu + np.exp(0.5 * log_var) * eps
    cc = (
        np.repeat(c, n_samples, axis=0) if cond_decoder
        else np.zeros((n_samples, c.shape[-1]))
    )
    return _severity(_decode_fields(decoder, z, cc))


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
    replacement from TRAIN rows only. Every row-aligned channel present
    (fields/log_freqs/freqs/severity/summaries/mode_shapes) is resampled
    together so conditioning variants (cnn/freq_summary) and in-training
    conformal calibration keep working; val/test rows are never touched.
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
            # here and remain untouched in the caller's dataset. Every
            # row-aligned channel is resampled together (fields AND the
            # conditioning/severity channels), so cnn/summary conditioning
            # and sigma_calibration='conformal' keep working.
            fit_dataset = dict(dataset)
            for key in ("fields", "log_freqs", "freqs", "severity",
                        "summaries", "mode_shapes"):
                if key in dataset:
                    fit_dataset[key] = np.asarray(dataset[key])[take]
            fit_dataset["splits"] = {"train": list(range(len(take))),
                                     "val": [], "test": []}
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
        # Conditioning echo from the first member so evaluators rebuild
        # the same model inputs (log_freqs vs summaries vs mode_shapes).
        "options": dict(members[0].get("options", {})),
    }
    return {"members": members, "combined": combined}


def evaluate_ensemble_sp_gates(
    ensemble: dict,
    dataset: dict,
    *,
    alpha: float = 0.10,
    seed: int | None = None,
    baseline_epochs: int = 200,
    split: str = "val",
    severity_mode: str = "analytic",
    n_latent_per_member: int = 10,
) -> dict:
    """SP-gates for a heteroscedastic ensemble on a chosen dataset split
    (default ``val``).

    BMA combination per evaluated observation:
      * predictive mean ``mu_bar = mean_k(mu_k)``;
      * per-pixel total variance by the law of total variance
        ``mean_k(sigma_k^2 + mu_k^2) - mu_bar^2``;
      * SP3 analytic interval ``mu_bar ± z_{alpha/2} * sqrt(total_var)``
        (z_{alpha/2} = 1.645 at alpha = 0.10), scored boundary-aware and
        stratified exactly like :func:`evaluate_sp_gates` (the gate binds
        on DAMAGED-cell coverage; pooled coverage with pristine cells
        would mask damage-side miscalibration).

    Model inputs mirror the members' conditioning (cnn → mode shapes,
    freq_summary → concat(log_freqs, summaries), else log-frequencies),
    read from the first member's echoed ``options``.

    SBC severity ensembles (``severity_mode="analytic"``, default) draw ONE
    field sample per member at RAW sigma (``mu_k + sigma_k * eps``),
    giving a K-sample severity ensemble per observation; ranks and gates
    as in :func:`evaluate_sp_gates` (exact-MC uniformity p > 0.05 is the
    sole SP4 criterion; no post-hoc recalibration). With
    ``severity_mode="latent"`` each member instead contributes
    ``n_latent_per_member`` posterior-latent decoded fields (which preserve
    spatial correlation, unlike independent-pixel sampling); members must
    then carry encoder/decoder/posterior. The SBC gate is reported invalid
    (``sbc_gate_valid=False``) below ``SBC_MIN_SAMPLES`` total draws, where
    the exact-MC p-value has almost no power.

    Point accuracy: BMA mean-field MSE next to a
    :class:`DirectRegressionBaseline` fitted on the train split.

    Returns the evaluate_sp_gates key set plus ``per_member_coverage``
    (each member's own analytic-interval coverage, transparency).

    ``split`` selects which entry of ``dataset["splits"]`` the gates are
    computed on. Audit use: pass ``split="test"`` on an ensemble trained
    ONLY on the train rows — the caller is responsible for split
    discipline; nothing here prevents evaluating on a seen split.
    """
    if severity_mode not in ("analytic", "latent"):
        raise ValueError(
            f"unknown severity_mode {severity_mode!r}; "
            "expected 'analytic' or 'latent'"
        )
    fields = np.asarray(dataset["fields"], dtype=np.float64)
    log_freqs = np.asarray(dataset["log_freqs"], dtype=np.float64)
    severity = np.asarray(dataset["severity"], dtype=np.float64)
    if split not in dataset["splits"]:
        raise ValueError(
            f"unknown split {split!r}; available: {sorted(dataset['splits'])}"
        )
    eval_idx = np.asarray(dataset["splits"][split], dtype=int)
    if len(eval_idx) == 0:
        raise ValueError(f"{split!r} split is empty")

    rng = np.random.default_rng(seed)
    z_crit = float(_normal.ppf(1.0 - alpha / 2.0))

    if "combined" in ensemble:
        predictors = ensemble["combined"]["member_predictors"]
        eval_options = dict(ensemble["combined"].get("options", {}))
        members = ensemble.get("members")
    else:
        # Duck-typed ensembles: members carrying interval_predictor only.
        members = ensemble["members"]
        predictors = [m["interval_predictor"] for m in members]
        eval_options = dict(members[0].get("options", {})) if members else {}
    X = _resolve_eval_inputs(dataset, eval_options)
    pred_stack = [predictor(X[eval_idx]) for predictor in predictors]
    mu_stack = np.stack([p[0] for p in pred_stack], axis=0)      # (K,n,gy,gx)
    sigma_stack = np.stack([p[1] for p in pred_stack], axis=0)
    k_members = mu_stack.shape[0]

    mu_bar, total_sigma = _bma_combine(mu_stack, sigma_stack)
    lower, upper = mu_bar - z_crit * total_sigma, mu_bar + z_crit * total_sigma
    truth = fields[eval_idx]
    covered_map = _boundary_aware_covered(lower, upper, truth)
    pristine = truth >= 1.0
    damaged = ~pristine
    coverage = float(covered_map.mean())
    coverage_pristine = (
        float(covered_map[pristine].mean()) if pristine.any()
        else float("nan")
    )
    coverage_damaged = (
        float(covered_map[damaged].mean()) if damaged.any()
        else float("nan")
    )
    # SP3 binds on DAMAGED cells only (see evaluate_sp_gates); with no
    # damaged cells fall back to pooled coverage.
    gate_coverage = coverage_damaged if damaged.any() else coverage

    # Transparency: each member judged alone on its own intervals.
    per_member_coverage = [
        float(_boundary_aware_covered(
            mu_k - z_crit * sig_k, mu_k + z_crit * sig_k, truth).mean())
        for mu_k, sig_k in zip(mu_stack, sigma_stack)
    ]

    # SBC: one sample per member -> K-sample severity ensembles at raw
    # sigma. No post-hoc conformal recalibration: it consumed the
    # evaluated split's ground-truth severities and masked the very
    # miscalibration SP4 exists to detect.
    severity_ensembles = []
    if severity_mode == "latent":
        if members is None or not all(
            all(k in m for k in ("encoder", "decoder", "posterior"))
            for m in members
        ):
            raise ValueError(
                "severity_mode='latent' requires every member to carry "
                "encoder/decoder/posterior"
            )
        member_opts = [
            dict(m.get("options", eval_options)) for m in members]
        member_conds = [bool(o.get("cond_decoder", True)) for o in member_opts]
        for j, i in enumerate(eval_idx):
            draws = [
                _latent_severity_ensemble(
                    m, X[i], n_latent_per_member, rng, cond_decoder=cd)
                for m, cd in zip(members, member_conds)
            ]
            severity_ensembles.append(np.concatenate(draws))
        n_bins_total = k_members * n_latent_per_member + 1
    else:
        eps = rng.standard_normal(sigma_stack.shape)
        samples = mu_stack + sigma_stack * eps                 # (K,n,gy,gx)
        for j in range(len(eval_idx)):
            severity_ensembles.append(_severity(samples[:, j]))
        n_bins_total = k_members + 1
    ranks = sbc_ranks(severity_ensembles, severity[eval_idx])
    sbc_pvalue = sbc_rank_uniformity_pvalue_exact(ranks, n_bins_total)
    sbc_gate_valid = bool((n_bins_total - 1) >= SBC_MIN_SAMPLES)
    counts = np.bincount(ranks, minlength=n_bins_total).astype(float)
    props = counts / counts.sum()
    sbc_error = float(np.mean(np.abs(props - 1.0 / n_bins_total)))

    cvae_mse = float(np.mean(
        [posterior_mean_mse(mu_bar[j], fields[i])
         for j, i in enumerate(eval_idx)]
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
    baseline_preds = baseline.predict(log_freqs[eval_idx])
    baseline_mse = float(np.mean(
        [posterior_mean_mse(pred, fields[i])
         for pred, i in zip(baseline_preds, eval_idx)]
    ))

    return {
        "coverage": coverage,
        "coverage_pristine": coverage_pristine,
        "coverage_damaged": coverage_damaged,
        "coverage_gate_pass": bool(gate_coverage > SP3_COVERAGE_GATE),
        "per_member_coverage": per_member_coverage,
        "ranks": ranks,
        "sbc_pvalue": sbc_pvalue,
        "sbc_error": sbc_error,
        "sbc_gate_valid": sbc_gate_valid,
        "sbc_gate_pass": bool(sbc_gate_valid and sbc_pvalue > 0.05),
        "cvae_mse": cvae_mse,
        "baseline_mse": baseline_mse,
        "n_val": int(len(eval_idx)),
        "alpha": alpha,
        "n_members": int(k_members),
        "severity_mode": severity_mode,
    }
