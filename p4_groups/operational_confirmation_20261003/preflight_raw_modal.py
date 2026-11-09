"""Mandatory execution checks for the fixed R/M experiment; no model search."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import raw_modal_data as D
from experiment_raw_modal import RawModalModel
import analyze_gate_bypass_receipt as A

PRIMARY = ('clean', 'iid1', 'correlated1', 'iid1_fresh20261020', 'correlated1_fresh20261020', 'iid1_fresh20261021', 'correlated1_fresh20261021')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checks(data, waves, valid):
    # Actual measured data, independent explicit slicing including early warmup.
    frames = np.array([0, 1, 4, 20, 60, 100, 123])
    rows = np.zeros(len(frames), dtype=int)
    actual = D.raw_windows(waves, valid, rows, frames)
    for index, frame in enumerate(frames):
        end = 128+64*int(frame)
        expected = np.zeros((384, 18), np.float32)
        expected[:, 17] = np.arange(-383, 1, dtype=np.float32)*np.float32(.001)
        for position, sample in enumerate(range(end-384, end)):
            if sample >= 0 and valid[0, sample]:
                expected[position, :16] = waves[0, :, sample]
                expected[position, 16] = 1
        np.testing.assert_array_equal(actual[index, :, :17], expected[:, :17])
        np.testing.assert_allclose(actual[index, :, 17], expected[:, 17], atol=3e-8, rtol=0)
    assert actual.dtype == np.float32 and actual.shape == (len(frames), 384, 18)
    assert np.isfinite(actual).all()
    assert not (abs(actual[..., :16]) >= 50).any()
    control = D.raw_windows(waves, valid, rows, frames, zero_waveform=True)
    erased = actual.copy(); erased[..., :16] = 0
    np.testing.assert_array_equal(control, erased)
    # Inject physically forbidden values only into a throwaway copied trajectory.
    copied_wave = np.asarray(waves[:1]).copy()
    copied_valid = np.asarray(valid[:1]).copy()
    copied_valid[0, [100, 101, 102]] = False
    copied_wave[0, :, 100] = np.nan
    copied_wave[0, :, 101] = np.inf
    copied_wave[0, :, 102] = 50
    excluded = D.raw_windows(copied_wave, copied_valid, np.array([0]), np.array([0]))
    assert np.isfinite(excluded).all()
    assert np.count_nonzero(excluded[0, 356:359, :17]) == 0
    before = D.raw_windows(copied_wave, copied_valid, np.array([0]), np.array([4]))
    copied_wave[:, :, 384:] *= -19
    copied_valid[:, 384:] = ~copied_valid[:, 384:]
    after = D.raw_windows(copied_wave, copied_valid, np.array([0]), np.array([4]))
    np.testing.assert_array_equal(before, after)
    records = data['records']
    fit = np.array([r['design'] in ('D000007','D000033','D000068','D000087') and r['noise'] in ('clean','iid1') for r in records])
    slots, scaling = D.modal_inputs(data, fit)
    old_features = data['features'].copy()
    current = (old_features-scaling[0])/scaling[1]
    current[..., ~scaling[2]] = 0
    current[~data['observed']] = 0
    current = np.concatenate((current, data['observed'][..., None], data['age'][..., None]), axis=-1).astype(np.float32)
    for frame in (0, 1, 2, 4, 60):
        for slot, lag in enumerate((4,2,0)):
            expected = current[:, frame-lag] if frame >= lag else np.zeros_like(current[:,0])
            np.testing.assert_array_equal(slots[:,frame,slot], expected)
    np.testing.assert_array_equal(old_features, data['features'])
    torch.manual_seed(42)
    model = RawModalModel().to(dtype=torch.float32).eval()
    with torch.no_grad():
        m = torch.from_numpy(slots[rows, frames].copy())
        r = torch.from_numpy(actual.copy())
        output = model(m, r).numpy()
        order = np.array([6,2,0,5,1,4,3])
        reordered = model(m[order], r[order]).numpy()[np.argsort(order)]
        np.testing.assert_allclose(output, reordered, rtol=0, atol=2e-6)
        model(m, torch.from_numpy(control.copy()))
        repeated = model(m, r).numpy()
        np.testing.assert_array_equal(output, repeated)
    assert output.shape == (len(rows),) and np.isfinite(output).all()
    return dict(actual_calibrated_apertures=len(frames), warmup_and_alignment=True,
        no_future_samples=True, forbidden_values_excluded=True, zero_waveform_input_identity=True,
        reset_and_decision_order_invariant=True, cached_modal_values_unchanged=True,
        parameter_count=sum(p.numel() for p in model.parameters()), oldest_sample_ms=383)


def witness(data, waves, valid):
    directory = A.OUT/'raw_response_partition'
    acquisition = json.loads((D.OUT/'acquisition.json').read_text())
    calibration_offsets = np.asarray(acquisition['calibration_offsets'], dtype=float)
    calibration_scales = np.asarray(acquisition['calibration_scales'], dtype=float)
    offsets = []
    for path in sorted(directory.glob('masks_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            rows = z['row_indices'].copy()
            count = int(z['sample_count'])
            command = np.unpackbits(z['command'], axis=1, count=count, bitorder='little').astype(bool)
            finite = np.unpackbits(z['finite'], axis=1, count=count, bitorder='little').astype(bool)
            cap = np.unpackbits(z['cap'], axis=1, count=count, bitorder='little').astype(bool)
            np.testing.assert_allclose(calibration_offsets[rows], z['offsets'], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_allclose(calibration_scales[rows], z['scales'], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_array_equal(valid[rows], ~command & finite & ~cap)
            # Independent original calibration artifacts; retain the 1e-8 RMS floor.
            for local, row in enumerate(rows):
                if not valid[row,:51].all():
                    continue
                leading = np.asarray(waves[row,:,:51], dtype=float)
                np.testing.assert_allclose(leading.mean(axis=1), 0, atol=3e-7, rtol=0)
                rms = np.sqrt((leading**2).mean(axis=1))
                above_floor = calibration_scales[row] > 1e-8
                np.testing.assert_allclose(rms[above_floor], 1, atol=3e-7, rtol=0)
                assert (rms[~above_floor] <= 1+3e-7).all()
            offsets.extend(rows.tolist())
    assert sorted(offsets) == list(range(len(data['records'])))
    return dict(rejection_mask_rows=len(offsets), leading_calibration_rows=len(offsets))


def run():
    if not torch.cuda.is_available():
        raise RuntimeError('Original CUDA execution host required')
    D.OUT.mkdir(exist_ok=True)
    if (D.OUT/'protocol.json').exists() or (D.OUT/'preflight.json').exists():
        raise RuntimeError('Frozen execution preflight already exists; do not overwrite')
    source = json.loads((D.SOURCE/'protocol.json').read_text())
    data = D.load_data()
    waves, valid = D.load_measured()
    assert waves.shape == (len(data['records']),16,8001) and valid.shape == (len(data['records']),8001)
    architecture = checks(data,waves,valid)
    provenance = witness(data,waves,valid)
    records = data['records']
    designs = np.array([r['design'] for r in records]); noises = np.array([r['noise'] for r in records])
    reports, failed = {}, []
    for fold, ids in enumerate(source['evaluation_folds']):
        fitting = set(source['split']['train']) | (set(source['additional_designs'])-set(ids))
        proper = np.isin(designs,list(fitting)) & np.isin(noises,('clean','iid1'))
        assert not fitting & (set(ids) | set(source['split']['validation']) | set(source['split']['held']))
        slots, _ = D.modal_inputs(data,proper)
        for noise in PRIMARY:
            rows = np.flatnonzero(np.isin(designs,ids) & (noises==noise))
            frames = data['truth'].shape[1]
            row_indices, frame_indices = np.repeat(rows,frames), np.tile(np.arange(frames),len(rows))
            raw = np.empty((len(row_indices),384,18),np.float32)
            for begin in range(0,len(row_indices),1024):
                end=begin+1024
                raw[begin:end] = D.raw_windows(waves,valid,row_indices[begin:end],frame_indices[begin:end])
            # Relative sample times are constant for all rows, not absolute endpoint labels.
            np.testing.assert_array_equal(raw[...,17], np.broadcast_to(raw[0,:,17],raw[...,17].shape))
            modal = slots[rows].reshape(len(rows),frames,-1)
            # Omitting strictly constant coordinates preserves EXACT input equality.
            real = np.concatenate((modal, raw[...,:17].reshape(len(rows),frames,-1)),axis=-1)
            mask = np.concatenate((modal,raw[...,16].reshape(len(rows),frames,-1)),axis=-1)
            y,d = data['truth'][rows],designs[rows]
            no_sample = ~raw[...,16].any(axis=1).reshape(len(rows),frames)
            for stratum in ('all','boundary'):
                selected = np.ones(y.shape,bool) if stratum=='all' else (y>-.02)&(y<.05)
                weights=A.original_population_weights(selected,d)
                target=(.010 if noise=='clean' else .015) if stratum=='all' else (.005 if noise=='clean' else .010)
                floor_r=A.weighted_collision_floor(real,y,weights,d)
                floor_m=A.weighted_collision_floor(mask,y,weights,d)
                floor_gap=A.weighted_collision_floor(mask,y,weights*no_sample,d)
                key=f'fold{fold}/{noise}/{stratum}'
                reports[key]=dict(candidate_r=floor_r,control_m=floor_m,no_sample_bound=floor_gap,
                    no_sample_weight=float(weights[no_sample].sum()),target=target,frames=int(selected.sum()),
                    original_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
                if floor_r['floor'] > target:
                    failed.append(dict(condition=key,floor=floor_r['floor'],target=target))
            del raw,real,mask
        print('RAW_MODAL_PREFLIGHT_FOLD',fold,flush=True)
    inputs=[D.SOURCE/'dataset.npz',D.SOURCE/'protocol.json',D.OUT/'acquisition.json',D.OUT/'waveforms.npy',D.OUT/'admissible.npy',
        D.SCRIPT/'raw_modal_data.py',D.SCRIPT/'experiment_raw_modal.py',Path(__file__),
        D.SCRIPT/'score_additive_spectral.py',D.SCRIPT/'experiment_multivel_history.py',D.ROOT/'src/mechanics/p4_margin_estimation/baselines.py',
        Path(D.E.__file__),Path(D.E.M.__file__),Path(A.__file__)]
    protocol=dict(scope='Fixed bounded measured raw-content R versus capacity/mask/gap-refitting M; grouped development only',
        inputs={str(path.relative_to(D.ROOT)):sha(path) for path in inputs},split=source['split'],additional_designs=source['additional_designs'],evaluation_folds=source['evaluation_folds'],
        arms=['raw','mask'],architecture=architecture,raw_width=18,raw_samples=384,lags=[4,2,0],dt=.001,
        modal_source='Original cached accepted174 plus unchanged flag/age; no gate-bypass auxiliary estimates',
        raw_encoding='16calibrated waveform values zero at inadmissible samples +complete-channel admissibility +relative sample time; no absolute endpoint/schedule/simulator features',
        training=dict(seeds=[42,43,44],maximum_epochs=2000,patience=100,learning_rate=.003,weight_decay=.0001,microbatch=2048,
            fitting_noise=['clean','iid1'],loss='Full row/frame mean MSE, full-gradient accumulation, one optimizerstep per epoch',selection='Original2 checkpoint-validation designs clean/IID1 all native-frame MAE'),
        primary_noise=list(PRIMARY),protected_progress_ledger_sha256=sha(D.ROOT/'P4_PROGRESS.json'),
        limits=['Point estimator only; no qualification/uncertainty/warning/runtime/physical-transfer credit','No held-design predictions/scoring','M floor is a diagnostic control and does not veto candidateR'])
    save(D.OUT/'protocol.json',protocol)
    result=dict(passed=not failed,architecture=architecture,provenance=provenance,collision_reports=reports,failed_candidate_bounds=failed,
        protocol_sha256=sha(D.OUT/'protocol.json'),preflight_source_sha256=sha(__file__),
        fit_authorization='18fixed fits immediately permitted iff passed; no tuning or extravariants',qualification_unchanged=True)
    save(D.OUT/'preflight.json',result)
    print('RAW_MODAL_PREFLIGHT_COMPLETE',json.dumps(dict(passed=result['passed'],candidate_checks=len(reports),failed=failed)),flush=True)
    if failed:
        raise RuntimeError('Actual candidateR input collision floor forecloses primary target')


if __name__=='__main__':
    run()
