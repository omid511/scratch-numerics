"""TRAIN-only contracts, design weights and common screen reporting."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'p4_groups/physical_warning_rank_20261003'))
import run_ranking as R
E, C, np = R.E, R.C, R.np
import torch
from torch import nn
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
spec = importlib.util.spec_from_file_location('gap_scores', E.BASE / 'analyze_results.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
PROTOCOL = json.loads((OUT / 'protocol.json').read_text())


def fingerprint(script):
    return dict(parent=E.signature(), sources={str(p.relative_to(ROOT)): C.sha256(p)
                for p in (OUT / 'protocol.json', OUT / 'gap_common.py', OUT / script)},
                torch=torch.__version__)


def weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1. / counts[inverse] / len(counts)


def scale_fit(x):
    mean, scale, active = fit_ridge_scaling(x)
    return dict(mean=mean, scale=scale, active=active)


def scaled(scaling, x):
    z = (x-scaling['mean'])/scaling['scale']
    z[:, ~scaling['active']] = 0.
    return z


def tensor(x):
    return torch.as_tensor(np.array(x, copy=True))


def net(width, nonlinear=True, seed=42):
    torch.manual_seed(seed)
    if nonlinear:
        return nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    return nn.Sequential(nn.Linear(width,1))


def warm_net(predictor):
    closure = {n:c.cell_contents for n,c in zip(predictor.__code__.co_freevars,predictor.__closure__)}
    estimator, transform = closure['estimator'], closure['transform']
    scaling = {n:c.cell_contents for n,c in zip(transform.__code__.co_freevars,transform.__closure__)}
    model = net(len(scaling['mean']))
    with torch.no_grad():
        for layer,w,b in zip((model[0],model[2],model[4]),estimator.coefs_,estimator.intercepts_):
            layer.weight.copy_(tensor(w.T)); layer.bias.copy_(tensor(b))
    return model, scaling


def neural_predict(model, scaling, x):
    with torch.no_grad():
        return model(tensor(scaled(scaling,x))).numpy()[:,0]


def calibrated(point, cy, cids, held_point):
    radius = C.calibration(point,cy,cids)
    cp = C.distribution(point,radius)
    hp = C.distribution(held_point,radius)
    thresholds = np.array([C.threshold(cp[:,3],cy,b) for b in C.BUDGETS])
    rates = [float((-cp[cy>0,3]<=t).mean()) for t in thresholds]
    assert np.all(np.array(rates)<=C.BUDGETS)
    return hp, thresholds, dict(radius=radius,calibration_false_warning_rates=rates)


def summary(p,t,y,ids):
    error=p[...,0]-y
    out=dict(slices={},warning={})
    for name,mask in A.slices(y).items():
        if not mask.any():
            out['slices'][name]=dict(physical_clips=0)
            continue
        out['slices'][name]=dict(physical_clips=int(mask.sum()),designs=len(np.unique(ids[mask])),
            **C.helpers.point(y[mask],p[:,mask,0],ids[mask]),
            signed_p95=float(np.quantile(error[:,mask],.95)),signed_p99=float(np.quantile(error[:,mask],.99)),
            exceedance={str(v):float((error[:,mask]>v).mean()) for v in (.005,.01,.02,.03,.05)})
    for b in C.BUDGETS:
        flags=-p[...,3]<=t[C.BUDGETS.index(b)]
        out['warning'][str(b)]={}
        for label,mask,miss in [('stable_false_warning',y>0,False),('unstable_miss',y<=0,True),
                               ('unstable_last1_miss',(y<0)&(y>-.01),True),
                               ('stable_last1_warning',(y>0)&(y<.01),False)]:
            events=~flags[:,mask] if miss else flags[:,mask]
            out['warning'][str(b)][label]=dict(rate=float(events.mean()),counts_per_view=events.sum(1).tolist(),
                physical_clips=int(mask.sum()),designs=len(np.unique(ids[mask])))
    return out


def evaluate(path, d, held, arms, predictions, thresholds, meta):
    names=d['names']; y=d['y'][held]; ids=d['ids'][held]
    output=dict(protocol=PROTOCOL,screen=meta,arms=arms,conditions={},families={})
    groups=[(n,[n],output['conditions']) for n in names]
    groups += [(k,v,output['families']) for k,v in A.FAMILIES.items()]
    for name,members,dest in groups:
        indices=[names.index(n) for n in members]
        dest[name]={arm:summary(predictions[i,indices],thresholds[i],y,ids) for i,arm in enumerate(arms)}
    C.save_json(path,output)
    for name in ('clean','iid1','correlated1','iid5','correlated5'):
        group=output['conditions']['clean'] if name=='clean' else output['families'][name]
        print('GAP_SCREEN_ENDPOINTS',path.stem,name,json.dumps({arm:{
            'stable_over03':v['slices']['stable0.01']['over03'],
            'unstable_over03':v['slices']['unstable0.01']['over03'],
            'mae':v['slices']['all']['design_mae'],
            'miss5':v['warning']['0.05']['unstable_last1_miss']['rate'],
            'false5':v['warning']['0.05']['stable_false_warning']['rate'],
            'miss1':v['warning']['0.01']['unstable_last1_miss']['rate'],
            'false1':v['warning']['0.01']['stable_false_warning']['rate']}
            for arm,v in group.items()}),flush=True)
    return output


def inner_members(fold,proper):
    path=E.INNER/f'inner_seed42_fold{fold}_part0.npz'
    with np.load(path) as z:
        fit,query=z['fit_rows'].copy(),z['query_rows'].copy()
        objects=C.cloudpickle.loads(z['components'].tobytes())
    assert set(fit)|set(query)==set(proper) and not set(fit)&set(query)
    return fit,query,objects['one'],objects['full']
