"""Check grouped causal behavior and time actual full epochs before freezing.

Uses only fitting/checkpoint-validation designs. Does not score held designs or
run full fits. A frozen protocol's evidence is never overwritten.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT.parents[1] / 'src'))
import experiment_raw_tcn as T


def edge_proof():
    count = 0
    for dilation in (1, 2, 4, 8, 16, 32, 64):
        for length in (7, 65, 129, 384):
            layer = T._GroupedCausalConv(dilation).double()
            values = torch.randn(2, 32, length, dtype=torch.float64, requires_grad=True)
            original = layer(values)
            cut = length // 2
            changed = values.detach().clone()
            changed[:, :, cut:] *= -3
            torch.testing.assert_close(original[:, :, :cut], layer(changed)[:, :, :cut], rtol=0, atol=0)
            changed = values.detach().clone()
            changed[:, 0] *= -3
            torch.testing.assert_close(original[:, 8:], layer(changed)[:, 8:], rtol=0, atol=0)
            original[:, :, :cut].square().sum().backward()
            torch.testing.assert_close(values.grad[:, :, cut:], torch.zeros_like(values.grad[:, :, cut:]), rtol=0, atol=0)
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in layer.parameters()):
                raise FloatingPointError('Nonfinite grouped causal gradient')
            count += 1
    return count


def epoch_proof(cache, arm):
    torch.manual_seed(42)
    model = T.RawTCNModel().to(device='cuda', dtype=torch.float32)
    model.modal_gru.flatten_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss, mse = T.train_epoch(model, optimizer, cache, arm)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite full-epoch grouped gradient')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite grouped parameter after Adam')
    model.eval()
    start = time.perf_counter()
    validation_mae = T.validate(model, cache)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    if not np.isfinite([loss, mse, validation_mae]).all():
        raise FloatingPointError('Nonfinite grouped full-epoch metric')
    return dict(arm=arm, training_decisions=len(cache['proper_target']), validation_decisions=len(cache['valid_target']),
        microbatch=T.MICRO, parameters=sum(p.numel() for p in model.parameters()), groups=4,
        training_seconds=training_seconds, validation_seconds=validation_seconds,
        total_seconds=training_seconds + validation_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        objective=loss, mse=mse, validation_mae=validation_mae, finite_gradients=True, finite_adam_parameters=True)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA execution host required')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    edges = edge_proof()
    print('GROUPED_CAUSAL_EDGE_PROOF', edges, flush=True)
    protocol = T.R.contract()[0]
    data = T.D.load_data()
    waves, valid = T.D.load_measured()
    fitting = set(protocol['split']['train']) | (set(protocol['additional_designs']) - set(protocol['evaluation_folds'][0]))
    proper = np.array([r['design'] in fitting and r['noise'] in ('clean', 'iid1') for r in data['records']])
    validation = np.array([r['design'] in protocol['split']['validation'] and r['noise'] in ('clean', 'iid1') for r in data['records']])
    modal, _ = T.D.modal_inputs(data, proper)
    cache = T.R._prepare_fold_cache(0, proper, validation, modal, data['truth'], waves, valid)
    T.attach_criteria(cache, data)
    reports = []
    for arm in T.ARMS:
        report = epoch_proof(cache, arm)
        reports.append(report)
        print('FULL_EPOCH_PROOF', json.dumps(report), flush=True)
        torch.cuda.empty_cache()
    proof = dict(passed=True, edge_cases=edges, epochs=reports, source_sha256=T.R.sha(T.__file__),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, no_full_training_restart=True)
    if not (T.OUT / 'protocol.json').exists():
        T.R.save_json(T.OUT / 'execution_verification.json', proof)
    else:
        T.contract()
        print('FROZEN_VERIFICATION_UNCHANGED', flush=True)
    return proof


if __name__ == '__main__':
    run()
