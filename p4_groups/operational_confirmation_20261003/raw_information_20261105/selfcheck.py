"""CPU semantic checks for the fixed raw-only model and native causal windows."""
import os
import sys
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != HERE]
sys.path[:0] = [str(ROOT / 'src'), str(BASE)]

import numpy as np
import torch
import raw_modal_data as D
sys.path.insert(0, str(HERE))
from model import RawOnlyModel


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    waves = rng.normal(size=(1, 16, D.SAMPLES)).astype(np.float32)
    valid = np.ones((1, D.SAMPLES), dtype=bool)
    valid[:, 22:32] = False
    rows = np.array([0, 0, 0])
    frames = np.array([0, 4, 123])
    original = D.raw_windows(waves, valid, rows, frames)
    masked = D.raw_windows(waves, valid, rows, frames, zero_waveform=True)
    assert original.shape == (3, 384, 18)
    assert np.count_nonzero(masked[:, :, :16]) == 0
    np.testing.assert_array_equal(original[:, :, 16:], masked[:, :, 16:])
    mutated = waves.copy()
    mutated[:, :, 128:] = 37.0
    np.testing.assert_array_equal(original[:1], D.raw_windows(mutated, valid, rows[:1], frames[:1]))

    model = RawOnlyModel().float()
    x = torch.from_numpy(original)
    y = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    assert model(x).shape == (3,)
    try:
        model(x[:, :-1])
    except ValueError:
        pass
    else:
        raise AssertionError('invalid shape accepted')
    full = (model(x) - y).square().sum() / len(y)
    full.backward()
    expected = [p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for block in (slice(0, 2), slice(2, 3)):
        ((model(x[block]) - y[block]).square().sum() / len(y)).backward()
    for p, reference in zip(model.parameters(), expected):
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        torch.testing.assert_close(p.grad, reference, rtol=2e-4, atol=2e-6)
    print('CPU_SELFCHECK_OK shape causal_future mask_channels global_gradient')


if __name__ == '__main__':
    main()
