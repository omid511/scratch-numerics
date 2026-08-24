"""Gradient-flow regression gates for the P2 field pipelines (field_pipeline).

Three failure modes are pinned here, each one previously observed:

* I3-class silent freeze — the CNN mode-shape encoder's lazily-built FC
  readout was left OUT of the optimizer parameter snapshot (a broken guard
  made the eager ``_ensure_fc`` call a no-op), so it trained frozen at
  random init through entire runs. The repair routes measurement
  information into the encoder again; this suite turns red if the repair
  is ever reverted (or if any future lazy init re-creates the bug).
* D20 deliberate freeze — ``HeteroscedasticFieldDecoder.trunk_norm`` is a
  FIXED feature stabilizer: its affine parameters must stay at their
  init values AND outside every optimizer parameter list. A silent
  re-entry into an optimizer would turn an explicit measured lever into
  an untracked side effect.
* Generic forgotten-param regression — EVERY parameter inside each
  optimizer's actual parameter-list scope must change over training.
  This catches future lazy-init holes or missing param-list entries for
  ANY submodule, without hardcoding names.

The Adam constructor is spied so the assertions run against the EXACT
tensor lists the pipelines hand to their optimizers (no duplicated
scope bookkeeping that could drift from the source).
"""
import numpy as np
import torch

from mechanics.p2_inverse_damage.damage_data import split_designs
from mechanics.p2_inverse_damage.field_pipeline import (
    ModeShapeCNNEncoder,
    train_field_cvae,
    train_field_cvae_heteroscedastic,
)

# Toy-run geometry shared by every test (kept tiny for speed).
N_SAMPLES = 16
GY = GX = 8
N_MODES = 3
BATCH_SIZE = 8
EPOCHS_AE = 2
EPOCHS_POST = 2


# ─── Synthetic fixtures ──────────────────────────────────────────────


def _sin_dataset(n=N_SAMPLES, gy=GY, gx=GX, n_modes=N_MODES, seed=7):
    """Tiny synthetic field dataset dict matching load_field_dataset output.

    Fields are random smooth retention grids built from low-frequency sin
    products and scaled by a random per-sample severity; mode_shapes carry
    the same severity signal through mode-dependent wavenumber patterns;
    log-frequencies drop with severity (plus noise). Consistent in spirit
    with ``test_field_pipeline._synthetic_dataset`` but adds the
    ``mode_shapes`` channel required by ``conditioning='cnn'``.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, np.pi, gy), np.linspace(0.0, np.pi, gx),
        indexing="ij",
    )
    severity = rng.uniform(0.0, 0.6, size=n)
    fields = np.empty((n, gy, gx), dtype=np.float32)
    mode_shapes = np.empty((n, n_modes, gy, gx), dtype=np.float32)
    for i in range(n):
        # Smooth base pattern: product of low-freq sin modes, clipped so
        # retention stays in (0, 1].
        base = np.ones((gy, gx), dtype=np.float64)
        for m in range(n_modes):
            base *= 1.0 - 0.15 * np.sin((m + 1) * xx) * np.sin((m + 2) * yy)
        base = np.clip(base, 0.05, None)
        fields[i] = ((1.0 - severity[i]) * base).astype(np.float32)
        # Mode shapes: damage damps and reshapes the standing patterns.
        for m in range(n_modes):
            mode_shapes[i, m] = (
                (1.0 - severity[i])
                * np.sin((m + 1) * xx + 0.5 * severity[i])
                * np.cos((m + 2) * yy - 0.5 * severity[i])
            ).astype(np.float32)
    freqs = (
        50.0
        * (1.0 - 0.3 * severity)[:, None]
        * (1.0 + 0.05 * rng.standard_normal((n, n_modes)))
        * np.linspace(1.0, 4.0, n_modes)[None, :]
    )
    train, val, test = split_designs(n, seed=seed)
    return {
        "fields": fields,
        "freqs": freqs,
        "log_freqs": np.log(freqs),
        "severity": severity,
        "mode_shapes": mode_shapes,
        "splits": {"train": train, "val": val, "test": test},
    }


# ─── Optimizer spy ───────────────────────────────────────────────────


class _AdamSpy:
    """Intercept every ``torch.optim.Adam`` construction made by a run.

    ``captured`` collects one entry per optimizer, each the exact param
    list handed to Adam. ``initials`` maps ``id(param)`` → a clone of the
    tensor's value at the moment it FIRST entered any optimizer (i.e. the
    Phase-1 starting point for encoder/decoder params, the post-re-init
    starting point for the posterior heads).
    """

    def __init__(self):
        self.captured: list[list[torch.nn.Parameter]] = []
        self.initials: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._orig_init = torch.optim.Adam.__init__

    def __enter__(self):
        spy = self

        def patched(init_self, params, *args, **kwargs):
            plist = list(params)
            spy.captured.append(plist)
            for p in plist:
                if id(p) not in spy.initials:
                    spy.initials[id(p)] = (p, p.detach().clone())
            spy._orig_init(init_self, plist, *args, **kwargs)

        torch.optim.Adam.__init__ = patched
        return self

    def __exit__(self, *exc):
        torch.optim.Adam.__init__ = self._orig_init
        return False

    def run(self, train_fn, dataset, **kwargs):
        """Train under the spy and return (models_dict, self)."""
        kwargs.setdefault("progress", False)
        kwargs.setdefault("batch_size", BATCH_SIZE)
        kwargs.setdefault("epochs_ae", EPOCHS_AE)
        kwargs.setdefault("epochs_post", EPOCHS_POST)
        with self:
            out = train_fn(dataset, **kwargs)
        return out, self


def _moved_report(spy: _AdamSpy):
    """(changed_ids, unchanged [(name, max_abs_delta)]) over all scoped params."""
    changed, unchanged = set(), []
    for pid, (p, init) in spy.initials.items():
        delta = (p.detach() - init).abs().max().item()
        if delta > 0.0:
            changed.add(pid)
        else:
            unchanged.append((tuple(p.shape), delta))
    return changed, unchanged


# ─── 1. I3 regression pin: CNN encoder FC head must train ────────────


class TestCnnEncoderGradientPin:
    """The d84fee0 repair: eager ``_ensure_fc`` BEFORE the optimizer
    snapshot, so ``ModeShapeCNNEncoder.fc`` receives live gradients in
    BOTH phases. Reverting the repair (frozen-at-init FC readout) must
    turn this test red."""

    def test_fc_weight_changes_over_training(self):
        ds = _sin_dataset()
        # _ensure_fc reseeds torch to 0 internally, so this reproduces the
        # exact FC init the trained run starts from regardless of RNG state.
        ref = ModeShapeCNNEncoder(n_modes=N_MODES, d_c=64, seed=0)
        with torch.no_grad():
            ref._ensure_fc(ref.conv(torch.zeros(1, N_MODES, GY, GX)))
        init_w = ref.fc.weight.detach().clone()

        out, _ = _AdamSpy().run(
            train_field_cvae, ds,
            conditioning="cnn", cond_decoder=False,
        )

        enc = out["encoder"]
        # The lazy head must exist (was materialized before the snapshot).
        assert enc.fc is not None, "CNN encoder FC head never materialized"
        trained_w = enc.fc.weight.detach()
        assert trained_w.shape == init_w.shape
        assert not torch.equal(trained_w, init_w), (
            "ModeShapeCNNEncoder.fc.weight did not move over training — "
            "the lazy FC head is excluded from the optimizer param lists "
            "(I3-class silent-freeze regression)"
        )


# ─── 2. Frozen trunk_norm pin (decision D20) ─────────────────────────


class TestFrozenTrunkNorm:
    """HeteroscedasticFieldDecoder.trunk_norm is a deliberately FROZEN
    feature stabilizer (gamma=1, beta=0, requires_grad False)."""

    def test_trunk_norm_values_and_grad_flag_untouched(self):
        ds = _sin_dataset()
        out, _ = _AdamSpy().run(
            train_field_cvae_heteroscedastic, ds, conditioning="freq_only",
        )
        tn = out["decoder"].trunk_norm
        assert not tn.weight.requires_grad, (
            "trunk_norm.weight must stay requires_grad=False (D20)"
        )
        assert not tn.bias.requires_grad, (
            "trunk_norm.beta must stay requires_grad=False (D20)"
        )
        assert torch.all(tn.weight == 1.0), "frozen gamma was modified"
        assert torch.all(tn.bias == 0.0), "frozen beta was modified"


# ─── 3. Every trainable param in optimizer scope must move ───────────


class TestAllScopedParamsMove:
    """Generic gate: snapshot every tensor that enters ANY optimizer param
    list during a run and assert each CHANGED over 2 epochs. Catches any
    future lazy-init hole or forgotten param-list entry without hardcoding
    submodule names."""

    def test_train_field_cvae_freq_only_all_params_move(self):
        ds = _sin_dataset()
        _, spy = _AdamSpy().run(
            train_field_cvae, ds, conditioning="freq_only",
        )
        # Scope sanity: encoder.mlp + decoder.mlp + posterior trunk/heads
        # must all be represented in the captured optimizer lists.
        n_scoped = sum(len(lst) for lst in spy.captured)
        assert n_scoped > 0, "no optimizer parameter lists were captured"
        changed, unchanged = _moved_report(spy)
        assert len(changed) == len(spy.initials), (
            f"{len(unchanged)} parameter(s) inside the optimizer scope did "
            f"not move over training: {unchanged}"
        )

    def test_heteroscedastic_freq_only_all_params_move(self):
        ds = _sin_dataset()
        out, spy = _AdamSpy().run(
            train_field_cvae_heteroscedastic, ds, conditioning="freq_only",
        )
        n_scoped = sum(len(lst) for lst in spy.captured)
        assert n_scoped > 0, "no optimizer parameter lists were captured"
        # post.logvar_head is a DELIBERATE structural dead zone on this
        # path (lever-G bistability characterization): reconstruction runs
        # at the posterior-MEAN latent (mu only), so log_var is touched
        # solely through the KL term, whose gradient dies either at the
        # free-bits floor (small init) or the ±10-nat clamp (large init).
        # It must still be a MEMBER of the optimizer lists; it is exempted
        # from the must-have-moved check.
        dead_ids = {
            id(p) for p in out["posterior"].logvar_head.parameters()
        }
        scoped_ids = {id(p) for lst in spy.captured for p in lst}
        assert dead_ids <= scoped_ids, (
            "posterior.logvar_head fell out of the optimizer param lists"
        )
        changed, unchanged = _moved_report(spy)
        must_move = {pid for pid in spy.initials if pid not in dead_ids}
        missing = [spy.initials[pid][0].shape
                   for pid in must_move - changed]
        assert not missing, (
            f"{len(missing)} declared-trainable parameter(s) inside the "
            f"optimizer scope did not move over training: {missing}; "
            f"zero-delta report: {unchanged}"
        )

    def test_heteroscedastic_sigma_head_is_in_scope(self):
        """The Phase-2b sigma-calibration sub-phase must include the sigma
        head in its own optimizer (it trains alone there)."""
        ds = _sin_dataset()
        out, spy = _AdamSpy().run(
            train_field_cvae_heteroscedastic, ds, conditioning="freq_only",
        )
        scoped_ids = {
            id(p) for lst in spy.captured for p in lst
        }
        for name, p in out["decoder"].sigma_head.named_parameters():
            assert id(p) in scoped_ids, (
                f"decoder.sigma_head.{name} never entered an optimizer"
            )
            init = spy.initials[id(p)][1]
            assert not torch.equal(p.detach(), init), (
                f"decoder.sigma_head.{name} did not move over training"
            )


# ─── 4. Optimizer membership sanity: the freeze is intentional ───────


class TestOptimizerMembershipSanity:
    """Document the D20 freeze as intentional: trunk_norm affine params
    must NOT appear in ANY optimizer param list."""

    def test_trunk_norm_excluded_from_every_optimizer(self):
        ds = _sin_dataset()
        out, spy = _AdamSpy().run(
            train_field_cvae_heteroscedastic, ds, conditioning="freq_only",
        )
        frozen_ids = {
            id(p) for p in out["decoder"].trunk_norm.parameters()
        }
        assert frozen_ids, "trunk_norm unexpectedly has no parameters"
        for i, lst in enumerate(spy.captured):
            leaked = [j for j, p in enumerate(lst) if id(p) in frozen_ids]
            assert not leaked, (
                f"frozen trunk_norm params leaked into optimizer #{i} "
                f"at positions {leaked} (freeze is no longer enforced)"
            )

    def test_no_frozen_tensor_in_any_optimizer(self):
        """Stronger generic form: every tensor entering an optimizer must
        be trainable (requires_grad True)."""
        ds = _sin_dataset()
        _, spy = _AdamSpy().run(
            train_field_cvae, ds, conditioning="freq_only",
        )
        for i, lst in enumerate(spy.captured):
            for j, p in enumerate(lst):
                assert p.requires_grad, (
                    f"requires_grad=False tensor at optimizer #{i}, "
                    f"position {j}"
                )
