"""Consumer regressions for grouped fitting, validation weighting and prefixes."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mechanics.p4_margin_estimation.baselines import train_gru
from mechanics.p4_margin_estimation.optimization import fit_epochs
from mechanics.p4_margin_estimation.temporal import TemporalMarginModel
from mechanics.p4_margin_estimation.train import train


def test_validation_loss_independent_of_batch_partition():
    train_data = TensorDataset(torch.zeros(3, 1), torch.zeros(3, 1))
    val_data = TensorDataset(torch.zeros(5, 1), torch.tensor([[1.], [1.], [1.], [1.], [4.]]))
    losses = []
    for batch_size in (2, 5):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        def loss(model, batch):
            x, y = batch
            return (model(x) - y).square().mean()
        history = fit_epochs(model, DataLoader(train_data, batch_size=2),
            DataLoader(val_data, batch_size=batch_size), loss, epochs=1, lr=.001)
        losses.append(history['val_loss'][0])
    assert losses == pytest.approx([4., 4.])


def test_grouped_gru_seed_reproduces_predictions():
    clips = [SimpleNamespace(sensor_signals=np.random.default_rng(i).normal(size=(2, 8)),
             margin=(i - 2) / 10, design_id=f'D{i}') for i in range(5)]
    query = torch.tensor(np.stack([c.sensor_signals for c in clips]), dtype=torch.float32)
    kwargs = dict(n_channels=2, hidden_dim=4, epochs=2, seed=17, batch_size=2)
    a, _ = train_gru(clips, **kwargs)
    torch.manual_seed(999)
    b, _ = train_gru(clips, **kwargs)
    with torch.inference_mode():
        torch.testing.assert_close(a(query), b(query), atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['gru', 'tcn'])
def test_saturated_tail_cannot_change_prediction(kind):
    torch.manual_seed(7)
    model = TemporalMarginModel(kind, sequence_length=32).eval()
    signal = torch.randn(2, 16, 32)
    lengths = torch.tensor([5, 17])
    side = torch.randn(2, 175)
    altered = signal.clone()
    for i, length in enumerate(lengths):
        altered[i, :, length:] = 100 * torch.randn_like(altered[i, :, length:])
    with torch.inference_mode():
        before, after = model(signal, lengths, side), model(altered, lengths, side)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    assert torch.all(before[:, 0] <= before[:, 1]) and torch.all(before[:, 1] <= before[:, 2])


def test_explicit_tcn_split_rejects_calibration_design_leakage():
    def clip(design):
        return SimpleNamespace(sensor_signals=np.zeros((2, 8)), margin=.1,
                               design_id=design, velocity=1.)
    with pytest.raises(ValueError, match='design overlap'):
        train([], n_channels=2, hidden_dim=4, epochs=1,
              train_clips=[clip('a')], val_clips=[clip('b')], test_clips=[clip('a')])
