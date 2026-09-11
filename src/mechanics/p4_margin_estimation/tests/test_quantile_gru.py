"""Phase-1 quantile-GRU slice: ordering, training, pairing, consistency."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechanics.p4_margin_estimation.quantile_gru import (  # noqa: E402
    QuantileGRUModel,
    build_state_pairs,
    train_quantile_gru,
    train_quantile_gru_consistency,
)

C, T = 4, 32


def _clip(margin, design_id, velocity=100.0, seed=0, scale=2.0, mask=False, n_channels=C):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_channels, T)) * 0.1 + float(margin) * scale
    sig = np.concatenate([base, (rng.random((n_channels, T)) > 0.5).astype(float)], axis=0) if mask else base
    return SimpleNamespace(
        sensor_signals=sig, margin=float(margin), design_id=design_id,
        velocity=float(velocity), dt=0.001, time=np.arange(T) / 1000.0,
    )


def _trend_clips(design_id, margins, velocity=100.0, seed=0, **kw):
    return [_clip(m, design_id, velocity, seed=seed + i, **kw) for i, m in enumerate(margins)]


def _median_gap(model, pairs):
    model.eval()
    with torch.no_grad():
        X = torch.stack([torch.tensor(np.asarray(c.sensor_signals), dtype=torch.float32) for c in
                         [m for p in pairs for m in p]])
        med = model(X)[:, 1].reshape(-1, 2)
        return float(torch.mean(torch.abs(med[:, 0] - med[:, 1])).item())


class TestOrdering:
    def test_random_input_ordered(self):
        torch.manual_seed(0)
        out = QuantileGRUModel(n_channels=C, hidden_dim=8)(torch.randn(6, C, T))
        assert out.shape == (6, 3)
        assert bool(((out[:, 0] <= out[:, 1]) & (out[:, 1] <= out[:, 2])).all())


class TestPinballDecreases:
    def test_two_epoch_loss_drops(self):
        margins = np.linspace(-0.2, 0.2, 16)
        train = _trend_clips("D000001", margins[:8], seed=0) + _trend_clips("D000002", margins[8:], seed=100)
        val = _trend_clips("D000003", np.linspace(-0.15, 0.15, 8), seed=200)
        _, hist = train_quantile_gru(train, n_channels=C, hidden_dim=8, epochs=2,
                                     batch_size=8, seed=0, val_clips=val)
        assert hist["train_loss"][-1] < hist["train_loss"][0]
        assert len(hist["train_loss"]) == 2 and len(hist["val_loss"]) == 2


class TestBuildStatePairs:
    def test_groups_excludes_pairs_correctly(self):
        a = [_clip(0.1, "DA", 100.0, seed=i) for i in range(3)]  # 1 pair + odd leftover
        b = [_clip(0.1, "DA", 200.0, seed=10)]  # singleton
        c = [_clip(0.1, "DB", 100.0, seed=20 + i) for i in range(2)]  # 1 pair
        d = [_clip(0.1, "DA", 100.0000001, seed=30)]  # near-but-unequal velocity: singleton
        pairs = build_state_pairs(a + b + c + d)
        assert len(pairs) == 2
        assert pairs[0] == (a[0], a[1])
        assert pairs[1] == (c[0], c[1])
        for x, y in pairs:
            assert (x.design_id, x.velocity) == (y.design_id, y.velocity)
        flat = [m for p in pairs for m in p]
        assert not any(m is b[0] or m is d[0] for m in flat)  # singletons dropped
        assert not any(m is a[2] for m in flat)  # odd leftover dropped


class TestConsistency:
    def test_smaller_pair_median_gap(self):
        torch.manual_seed(0)
        train = [_clip(0.1, "D000001", 100.0, seed=i) for i in range(6)] + \
                [_clip(-0.1, "D000002", 200.0, seed=100 + i) for i in range(6)]
        val = _trend_clips("D000003", np.linspace(-0.1, 0.1, 6), seed=200)
        kw = dict(n_channels=C, hidden_dim=8, epochs=10, batch_size=4, seed=0, val_clips=val)
        plain, _ = train_quantile_gru(train, **kw)
        cons, hist = train_quantile_gru_consistency(train, lambda_cons=1.0, **kw)
        assert len(hist["train_loss"]) == 10
        pairs = build_state_pairs(train)
        assert _median_gap(cons, pairs) < _median_gap(plain, pairs)


class TestMaskChannels:
    def test_two_channel_inputs_accepted(self):
        torch.manual_seed(0)
        train = [_clip(0.1, "D000001", seed=i, mask=True) for i in range(4)] + \
                [_clip(-0.1, "D000002", seed=100 + i, mask=True) for i in range(4)]
        val = [_clip(0.05, "D000003", seed=200 + i, mask=True) for i in range(4)]
        model, hist = train_quantile_gru(train, n_channels=2 * C, hidden_dim=8, epochs=1,
                                         batch_size=4, seed=0, val_clips=val)
        out = model(torch.stack([torch.tensor(np.asarray(c.sensor_signals), dtype=torch.float32) for c in val]))
        assert out.shape == (4, 3)
        assert bool(((out[:, 0] <= out[:, 1]) & (out[:, 1] <= out[:, 2])).all())
        assert len(hist["train_loss"]) == 1


class TestOverlap:
    def test_train_val_overlap_raises(self):
        train = [_clip(0.1, "D000001", seed=i) for i in range(4)]
        val = [_clip(0.05, "D000001", seed=100 + i) for i in range(2)]
        with pytest.raises(ValueError):
            train_quantile_gru(train, n_channels=C, hidden_dim=8, epochs=1,
                               batch_size=4, seed=0, val_clips=val)
        with pytest.raises(ValueError):
            train_quantile_gru_consistency(train, n_channels=C, hidden_dim=8, epochs=1,
                                           batch_size=4, seed=0, val_clips=val)
