"""Two fixed model trials retaining the completed raw/modal/mask information.

Same causal input, splits, calibration, optimizer and checkpoint rule as R/M.
Only the waveform encoder changes to a groups=4 multiscale residual TCN.
Two matched-seed arms differ only in the supervised training objective.
No gate-bypass extraction, new acquisition, held scoring or qualification change.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import experiment_raw_modal as R
import analyze_raw_modal as A
from mechanics.p4_margin_estimation.tcn import CausalConv1d, ChannelLayerNorm, TCNBlock, TCNBackbone, TemporalSummary

D, S, H = R.D, R.S, R.H
ROOT, SCRIPT = R.ROOT, R.SCRIPT
OUT = SCRIPT / 'raw_tcn_grouped4_20261010'
ARMS = ('tcn_mse', 'tcn_risk')
SEEDS = (42, 43, 44)
MICRO = 512
WEIGHTS = (.5, .5, 2., 1., 1.)


class _GroupedCausalConv(CausalConv1d):
    """Native grouped convolution with the existing left-only causal padding."""
    def __init__(self, dilation):
        torch.nn.Module.__init__(self)
        self.pad = 2 * dilation
        self.conv = torch.nn.Conv1d(32, 32, 3, dilation=dilation, groups=4)


class _GroupedTCNBlock(TCNBlock):
    def __init__(self, dilation):
        torch.nn.Module.__init__(self)
        self.conv1 = _GroupedCausalConv(dilation)
        self.norm1 = ChannelLayerNorm(32)
        self.conv2 = _GroupedCausalConv(dilation)
        self.norm2 = ChannelLayerNorm(32)
        self.dropout = torch.nn.Dropout(0.)


class _GroupedTCNBackbone(TCNBackbone):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.input_proj = torch.nn.Conv1d(18, 32, 1)
        self.blocks = torch.nn.Sequential(*[_GroupedTCNBlock(2 ** i) for i in range(7)])
        self.receptive_field = 1 + 4 * sum(2 ** i for i in range(7))


class RawTCNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.modal_gru = torch.nn.GRU(176, 32, batch_first=True)
        self.raw_backbone = _GroupedTCNBackbone()
        self.summary = TemporalSummary(window=64)
        self.raw_projection = torch.nn.Sequential(torch.nn.Linear(96, 32), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, modal, raw):
        if modal.ndim != 3 or tuple(modal.shape[1:]) != (3, 176):
            raise ValueError('Expected modal [B,3,176]')
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (384,18) or len(raw) != len(modal):
            raise ValueError('Expected aligned raw [B,384,18]')
        _, hidden = self.modal_gru(modal)
        waveform = self.raw_projection(self.summary(self.raw_backbone(raw.transpose(1,2))))
        return self.head(torch.cat((hidden[-1], waveform), dim=1))[:,0]


def criteria(truth, current):
    # Derive eligibility from original float64 labels, not rounded GPU targets.
    return dict(boundary=(truth > -.02) & (truth < .05),
        sign=current & (abs(truth) >= .01) & (abs(truth) <= .05), far=truth >= .10)


def terms(point, target, masks):
    error = point-target
    b, sign, far = masks['boundary'], masks['sign'], masks['far']
    return torch.stack((error.square().sum(), error[b].square().sum(),
        torch.relu(error[b]-.02).square().sum(),
        torch.relu(.005-torch.sign(target[sign])*point[sign]).square().sum(),
        torch.relu(-error[far]-.005).square().sum()))


def denominators(masks, count, device):
    sizes = [count, int(masks['boundary'].sum()), int(masks['boundary'].sum()),
        int(masks['sign'].sum()), int(masks['far'].sum())]
    return torch.tensor(sizes, device=device, dtype=torch.float32).clamp_min(1), sizes


def scalar_loss(sums, normalizers, arm, risk_factors):
    if arm == 'tcn_mse':
        return sums[0]/normalizers[0]
    if arm != 'tcn_risk':
        raise ValueError(arm)
    return (sums*risk_factors).sum()


def blocks(count):
    for start in range(0, count, MICRO):
        yield slice(start, min(start+MICRO, count))


def attach_criteria(cache, data):
    for part in ('proper', 'valid'):
        rows, frames = cache[f'{part}_rows'], cache[f'{part}_frames']
        masks = criteria(data['truth'][rows,frames], data['observed'][rows,frames])
        normalizers, sizes = denominators(masks, len(rows), 'cuda')
        cache[f'{part}_criteria'] = {name:torch.as_tensor(mask, device='cuda') for name,mask in masks.items()}
        cache[f'{part}_normalizers'], cache[f'{part}_sizes'] = normalizers, sizes
        cache[f'{part}_risk_factors'] = torch.tensor(WEIGHTS, device='cuda', dtype=torch.float32)/normalizers


def train_epoch(model, optimizer, cache, arm):
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(5, device='cuda', dtype=torch.float32)
    for block in blocks(len(cache['proper_target'])):
        point = model(cache['proper_modal'][block], cache['proper_raw'][block])
        if not bool(torch.isfinite(point).all()):
            raise FloatingPointError('Nonfinite TCN point')
        if arm == 'tcn_mse':
            squared = (point-cache['proper_target'][block]).square().sum()
            (squared/cache['proper_normalizers'][0]).backward()
            total[0] += squared.detach()
        else:
            masks = {name:mask[block] for name,mask in cache['proper_criteria'].items()}
            sums = terms(point, cache['proper_target'][block], masks)
            scalar_loss(sums, cache['proper_normalizers'], arm, cache['proper_risk_factors']).backward()
            total += sums.detach()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite TCN gradient')
    optimizer.step()
    return float(scalar_loss(total, cache['proper_normalizers'], arm, cache['proper_risk_factors'])), float(total[0]/cache['proper_normalizers'][0])


def validate(model, cache):
    count = len(cache['valid_target'])
    absolute = torch.zeros((), device='cuda')
    with torch.no_grad():
        for block in blocks(count):
            point = model(cache['valid_modal'][block], cache['valid_raw'][block])
            if not bool(torch.isfinite(point).all()):
                raise FloatingPointError('Nonfinite TCN validation')
            absolute += abs(point-cache['valid_target'][block]).sum()
    return float(absolute/count)


def loss_smoke():
    y = np.array([.01,-.015,.12,.001])
    masks = {k:torch.from_numpy(v) for k,v in criteria(y,np.ones(4,bool)).items()}
    p = torch.tensor([.05,.01,.08,.001], dtype=torch.float64, requires_grad=True)
    target = torch.from_numpy(y)
    norms, sizes = denominators(masks, 4, 'cpu')
    norms = norms.to(dtype=p.dtype)
    factors = p.new_tensor(WEIGHTS)/norms
    sums = terms(p,target,masks)
    np.testing.assert_allclose(sums.detach().numpy(), [.003825,.002225,.000425,.000225,.001225], rtol=0, atol=1e-12)
    halves = [terms(p[s],target[s],{k:v[s] for k,v in masks.items()}) for s in (slice(0,2),slice(2,4))]
    torch.testing.assert_close(sum(halves), sums, rtol=0, atol=1e-12)
    loss = scalar_loss(sums,norms,'tcn_risk',factors); loss.backward()
    np.testing.assert_allclose(p.grad.numpy(), [.05,.03625,-.08,0], rtol=0, atol=1e-12)
    return dict(global_denominators=sizes, risk_loss=float(loss.detach()),
        boundary_optimism_and_wrong_sign_push_down=True, farther_stable_underprediction_pushes_up=True,
        microbatch_objective_identity=True)


def smoke(data, waves, valid):
    if not torch.cuda.is_available():
        raise RuntimeError('Original CUDA host required')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    design_ids = set(R.contract()[0]['split']['train'])
    proper = np.array([r['design'] in design_ids and r['noise'] in ('clean','iid1') for r in data['records']])
    modal, _ = D.modal_inputs(data, proper)
    rows, frames = R._decision_indices(proper, data['truth'].shape[1])
    rows, frames = rows[:MICRO], frames[:MICRO]
    raw = D.raw_windows(waves,valid,rows,frames)
    m = torch.as_tensor(modal[rows,frames], device='cuda', dtype=torch.float32)
    x = torch.as_tensor(raw, device='cuda', dtype=torch.float32)
    model = RawTCNModel().to(device='cuda', dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(m[:8],x[:8]); reversed_point = model(m[:8].flip(0),x[:8].flip(0)).flip(0)
        torch.testing.assert_close(out,reversed_point,rtol=0,atol=2e-6)
        model(m[8:16],x[8:16])
        torch.testing.assert_close(out,model(m[:8],x[:8]),rtol=0,atol=0)
        altered = x[:8].clone(); altered[:,128:,:16] *= -3
        a = model.raw_backbone(x[:8].transpose(1,2))[:,:,:128]
        b = model.raw_backbone(altered.transpose(1,2))[:,:,:128]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.train()
    target = torch.as_tensor(data['truth'][rows,frames],device='cuda',dtype=torch.float32)
    cm = criteria(data['truth'][rows,frames],data['observed'][rows,frames])
    norms, _ = denominators(cm, len(rows), 'cuda')
    factors = norms.new_tensor(WEIGHTS)/norms
    masks = {k:torch.as_tensor(v,device='cuda') for k,v in cm.items()}
    torch.cuda.reset_peak_memory_stats()
    loss = scalar_loss(terms(model(m,x),target,masks),norms,'tcn_risk',factors); loss.backward()
    if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('TCN actual-data backward failed')
    proof = dict(actual_decisions=len(rows), parameters=sum(p.numel() for p in model.parameters()),
        receptive_field=model.raw_backbone.receptive_field, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        causal_backbone=True, decision_reset=True, batch_order_invariant=True, actual_backward_finite=True,
        loss=loss_smoke(), gpu=torch.cuda.get_device_name())
    del model,m,x,target,modal
    torch.cuda.empty_cache()
    return proof


def freeze():
    if (OUT/'protocol.json').exists():
        raise RuntimeError('TCN protocol already frozen')
    old, preflight, _ = R.contract()
    verification_path = OUT / 'execution_verification.json'
    verification = json.loads(verification_path.read_text())
    if verification['source_sha256'] != R.sha(__file__) or verification.get('passed') is not True:
        raise RuntimeError('Run grouped causal/full-epoch verification on the current TCN source before freezing')
    data = D.load_data(); waves,valid = D.load_measured()
    proof = smoke(data,waves,valid)
    files = [Path(__file__), ROOT/'src/mechanics/p4_margin_estimation/tcn.py',
        SCRIPT/'verify_raw_tcn_execution.py', Path(A.__file__), R.OUT/'protocol.json',
        R.OUT/'preflight.json', R.OUT/'results.json', verification_path]
    protocol = dict(scope='User-authorized stronger model and localized task-aware training; development only',
        inputs={**old['inputs'], **{str(p.relative_to(ROOT)):R.sha(p) for p in files}},
        split=old['split'], additional_designs=old['additional_designs'], evaluation_folds=old['evaluation_folds'],
        primary_noise=old['primary_noise'], arms=list(ARMS), seeds=list(SEEDS),
        retained_inputs='Unchanged modal174+flag/age three slots; calibrated raw16+sample admissibility+relative time384 samples; no feature deletion or extra oracle input',
        architecture=dict(modal_gru=[176,32],raw_tcn_width=32,raw_tcn_blocks=7,kernel=3,groups=4,dilations=[1,2,4,8,16,32,64],
            receptive_field=509,dropout=0,summary='early64/late64/last',projection=[96,32],head=[64,16,1]),
        execution='Float32 deterministic native grouped dilated causal convolution; groups=4; full-frame loss; no precision reduction or dropped observations',
        training=dict(maximum_epochs=2000,patience=100,learning_rate=.003,weight_decay=.0001,microbatch=MICRO,
            fitting_noise=['clean','iid1'],optimizer_steps_per_epoch=1,checkpoint='Original clean/IID1 validation all-native-frame MAE, identical both arms',
            mse='All-native-frame mean squared margin error',
            risk='0.5*all MSE +0.5*boundary MSE +2*boundary over02 squared hinge +current eligible sign005 squared hinge +farther-stable under005 squared hinge',
            criteria='Original float64 truth: boundary(-.02,.05), sign current&.01<=abs(m)<=.05, farther-stable m>=.10; labels used only in training loss',
            normalization='Each term uses its full proper-frame stratum denominator; empty terms contribute zero without redistributing weights; no target whitening'),
        inherited_input_preflight=R.sha(R.OUT/'preflight.json'), inherited_candidate_input_bounds=preflight['passed'],smoke=proof,
        protected_progress_ledger_sha256=old['protected_progress_ledger_sha256'],
        limits=['Soft supervised penalties, not guaranteed hard constraints','No new acquisition/extraction/bound campaign',
            'No held scoring, qualification or production replacement; current/retreat/sign/optimism/farther-stable gates unchanged'])
    R.save_json(OUT/'protocol.json',protocol)
    print('RAW_TCN_FROZEN',json.dumps(proof),flush=True)


def contract():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for relative,digest in protocol['inputs'].items():
        if R.sha(ROOT/relative) != digest:
            raise RuntimeError(f'Pinned TCN input changed: {relative}')
    if not protocol['inherited_candidate_input_bounds'] or not protocol['smoke']['actual_backward_finite']:
        raise RuntimeError('TCN execution prerequisites failed')
    return protocol


def fit(data, protocol, fold, arm, seed, proper, validation, modal, scaling, waves, valid, cache):
    stem = f'fold{fold}_{arm}_seed{seed}'
    path, state_path = OUT/f'{stem}.npz', OUT/f'{stem}.pt'
    digest = R.sha(OUT/'protocol.json')
    held = np.isin([r['design'] for r in data['records']],protocol['split']['held'])
    if path.exists() or state_path.exists():
        if not path.exists() or not state_path.exists():
            raise RuntimeError(f'Incomplete TCN fit {stem}')
        with np.load(path,allow_pickle=False) as z:
            if str(z['protocol_sha256']) != digest or not np.array_equal(z['proper'],proper) or not np.array_equal(z['validation'],validation):
                raise RuntimeError(f'TCN fit provenance mismatch {stem}')
            point, metadata = z['predictions'].copy(),json.loads(str(z['metadata']))
        if not np.isnan(point[held]).all() or not np.isfinite(point[~held]).all():
            raise RuntimeError(f'TCN fit prediction scope mismatch {stem}')
        state = torch.load(state_path,map_location='cpu')
        if state['protocol_sha256'] != digest:
            raise RuntimeError(f'TCN checkpoint mismatch {stem}')
        return point,metadata
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = RawTCNModel().to(device='cuda',dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    best, stale, history, best_state, best_epoch = float('inf'),0,[],None,0
    fit_started = time.perf_counter()
    for epoch in range(2000):
        model.train(); objective, mse = train_epoch(model,optimizer,cache,arm)
        model.eval(); value = validate(model,cache)
        if not np.isfinite((objective,mse,value)).all():
            raise FloatingPointError(f'Nonfinite TCN metrics {stem}')
        history.append(dict(epoch=epoch+1,objective=objective,train_mse=mse,validation_mae=value))
        if value < best:
            best,stale,best_epoch,best_state = value,0,epoch+1,R._cpu_state(model.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print('RAW_TCN_EPOCH', fold, arm, seed, epoch + 1, 'objective', objective,
                'validation_mae', value, 'best_epoch', best_epoch, 'best_mae', best,
                'elapsed_seconds', time.perf_counter() - fit_started, flush=True)
        if stale >= 100:
            break
    model.load_state_dict(best_state); model.eval()
    point = R._stream_predictions(model,modal,waves,valid,data['truth'].shape,~held,False)
    if not np.isnan(point[held]).all() or not np.isfinite(point[~held]).all():
        raise RuntimeError('TCN prediction protection failed')
    metadata = dict(fold=fold,arm=arm,seed=seed,best_epoch=best_epoch,epochs=len(history),validation_mae=best,history=history,
        parameters=sum(p.numel() for p in model.parameters()),proper_sizes=cache['proper_sizes'],validation_sizes=cache['valid_sizes'],
        optimizer_steps_per_epoch=1,microbatch=MICRO,protocol_sha256=digest)
    R._atomic_torch(state_path,dict(state=best_state,protocol_sha256=digest,fold=fold,arm=arm,seed=seed))
    R._atomic_npz(path,predictions=point,proper=proper,validation=validation,prediction_rows=~held,
        mean=scaling[0],scale=scaling[1],active=scaling[2],metadata=json.dumps(metadata),protocol_sha256=np.array(digest))
    print('RAW_TCN_FIT',fold,arm,seed,best_epoch,best,flush=True)
    return point,metadata


def train():
    protocol = contract()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fit fallback')
    torch.set_num_threads(1); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    data = D.load_data(); waves,valid = D.load_measured()
    records = data['records']; designs = np.array([r['design'] for r in records]); noises = np.array([r['noise'] for r in records])
    validation = np.isin(designs,protocol['split']['validation']) & np.isin(noises,('clean','iid1'))
    points,fits = [],[]
    for fold,ids in enumerate(protocol['evaluation_folds']):
        fitting = set(protocol['split']['train']) | (set(protocol['additional_designs'])-set(ids))
        if fitting & (set(ids)|set(protocol['split']['validation'])|set(protocol['split']['held'])):
            raise RuntimeError('TCN fitting/evaluation design overlap')
        proper = np.isin(designs,list(fitting)) & np.isin(noises,('clean','iid1'))
        modal,scaling = D.modal_inputs(data,proper)
        cache = R._prepare_fold_cache(fold,proper,validation,modal,data['truth'],waves,valid)
        attach_criteria(cache,data)
        for arm in ARMS:
            for seed in SEEDS:
                point,metadata = fit(data,protocol,fold,arm,seed,proper,validation,modal,scaling,waves,valid,cache)
                points.append(point); fits.append(metadata)
        R._release_fold_cache(); del cache,modal
    if len(fits) != 18:
        raise RuntimeError('TCN fixed-fit count mismatch')
    R._atomic_npz(OUT/'predictions.npz',predictions=np.stack(points),
        folds=np.array([f['fold'] for f in fits]),arms=np.array([f['arm'] for f in fits]),seeds=np.array([f['seed'] for f in fits]),
        protocol_sha256=np.array(R.sha(OUT/'protocol.json')))
    R.save_json(OUT/'training.json',dict(fit_count=len(fits),fits=fits,protocol_sha256=R.sha(OUT/'protocol.json'),
        predictions_sha256=R.sha(OUT/'predictions.npz'),no_held_predictions=True,gpu=torch.cuda.get_device_name()))
    print('RAW_TCN_TRAINING_COMPLETE',len(fits),flush=True)


def score():
    protocol = contract()
    if (OUT/'results.json').exists():
        raise RuntimeError('Completed TCN scores already exist')
    data = D.load_data(); records = data['records']; designs = np.array([r['design'] for r in records])
    strata = H.strata(data); strata['farther_stable15'] = data['truth'] >= .15
    _,valid = D.load_measured()
    cumulative = np.zeros((len(records),valid.shape[1]+1),np.int32)
    np.cumsum(valid,axis=1,dtype=np.int32,out=cumulative[:,1:])
    ends = 128+64*np.arange(data['truth'].shape[1])
    counts = cumulative[:,ends]-cumulative[:,np.maximum(0,ends-384)]; del cumulative
    raw_strata = dict(raw_no_samples=counts==0,raw_partial_window=(counts>0)&(counts<384),raw_full_window=counts==384,
        raw_present_no_modal=(counts>0)&~data['observed'])
    strata.update(raw_strata); strata.update({f'boundary_{k}':v&strata['boundary'] for k,v in raw_strata.items()})
    for profile in sorted({r['profile'] for r in records}):
        pm = np.array([r['profile']==profile for r in records])[:,None]
        strata[f'profile_{profile}'] = pm & strata['all']; strata[f'boundary_profile_{profile}'] = pm & strata['boundary']
    metrics = json.loads((R.OUT/'results.json').read_text())['metrics']
    with np.load(OUT/'predictions.npz',allow_pickle=False) as z:
        prediction = z['predictions'].copy(); folds,arms,seeds = z['folds'],z['arms'],z['seeds']
    held = np.isin(designs,protocol['split']['held'])
    if not np.isnan(prediction[:,held]).all() or not np.isfinite(prediction[:,~held]).all():
        raise RuntimeError('TCN held protection failed at scoring')
    checks = 0
    for fold,ids in enumerate(protocol['evaluation_folds']):
        for noise in sorted({r['noise'] for r in records if r['design'] in ids}):
            condition = f'fold{fold}/evaluation/{noise}'
            pop = np.array([r['design'] in ids and r['noise']==noise for r in records])[:,None]
            for arm in ARMS:
                indices = [np.flatnonzero((folds==fold)&(arms==arm)&(seeds==seed)).item() for seed in SEEDS]
                point = prediction[indices]
                metrics[condition][arm] = {name:S.summary(point,data['truth'],pop&mask,designs) for name,mask in strata.items()}
                if noise in protocol['primary_noise']:
                    for name in ('all','boundary'):
                        for design in ids:
                            selected = pop & strata[name] & (designs==design)[:,None]
                            direct = abs(point[:,selected]-data['truth'][selected]).mean(axis=1)
                            saved = metrics[condition][arm][name]['per_design'][design]
                            np.testing.assert_allclose(direct,saved['mae_by_seed'],rtol=0,atol=1e-12)
                            if int(selected.sum()) != saved['frames']:
                                raise RuntimeError('TCN original denominator mismatch')
                            checks += 1
    R.save_json(OUT/'results.json',dict(metrics=metrics,arms=[*A.ARMS,*ARMS],protocol_sha256=R.sha(OUT/'protocol.json'),
        independent_original_denominator_checks=checks,operational_predictor_changed=False))
    report(protocol,metrics,checks)
    print('RAW_TCN_SCORED',len(metrics),checks,flush=True)


def report(protocol, metrics, checks):
    primary = {}
    for noise in protocol['primary_noise']:
        by_arm,paired = {},{}
        for arm in (*A.ARMS,*ARMS):
            whole = A.joined(metrics,noise,arm,'all'); boundary = A.joined(metrics,noise,arm,'boundary')
            by_arm[arm] = dict(slices={name:A.mae(A.joined(metrics,noise,arm,name)) for name in
                ('all','boundary','boundary_current','profile_retreat','farther_stable','farther_stable15','raw_present_no_modal','raw_no_samples')},
                optimism=A.event(metrics,noise,arm,'boundary','optimism_over02'),
                stable_sign=A.event(metrics,noise,arm,'current','stable_wrong_sign'),
                unstable_sign=A.event(metrics,noise,arm,'current','unstable_exclusion_wrong_sign'),
                no_sample_all=A.contribution(whole,A.joined(metrics,noise,arm,'raw_no_samples')),
                no_sample_boundary=A.contribution(boundary,A.joined(metrics,noise,arm,'boundary_raw_no_samples')))
        for arm in ARMS:
            paired[arm] = {name:{control:A.paired(A.joined(metrics,noise,arm,name),A.joined(metrics,noise,control,name))
                for control in (*A.ARMS,'tcn_mse') if control != arm}
                for name in ('all','boundary','boundary_current','profile_retreat','farther_stable','raw_present_no_modal')}
        primary[noise] = dict(arms=by_arm,paired=paired)
        print('RAW_TCN_RESULT',noise,{arm:{name:by_arm[arm]['slices'][name]['mean'] for name in ('all','boundary')}
            for arm in ('raw','mask',*ARMS)},flush=True)
    if R.sha(ROOT/'P4_PROGRESS.json') != protocol['protected_progress_ledger_sha256']:
        raise RuntimeError('Protected progress ledger changed')
    R.save_json(OUT/'comparison_summary.json',dict(primary=primary,independent_original_denominator_checks=checks,
        held_predictions_absent=True,qualification_and_progress_ledger_unchanged=True,
        sources={name:R.sha(OUT/name) for name in ('protocol.json','training.json','results.json')},
        bootstrap='2000 seed42 paired design resamples preserving all3 seeds; no refit/seed uncertainty or multiplicity correction',
        limits='Development point-error evidence only; not blind/physical transfer, qualified warning or safety guarantees'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase',choices=('freeze','train','score'))
    phase = parser.parse_args().phase
    globals()[phase]()
