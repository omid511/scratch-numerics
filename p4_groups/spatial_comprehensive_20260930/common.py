"""Fixed column contracts and artifact helpers for the spatial ingredient study."""
import importlib.util
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'p4_groups/two_mode_spatial_20260930'
VERIFY=ROOT/'p4_groups/blend_verification_20260930'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'p4_groups/correlated_training_20260929')]
import cloudpickle
import numpy as np
import sklearn
from run_augmentation_experiment import sha256
from experiment_p4_tabular import fit_network
spec=importlib.util.spec_from_file_location('spatial_analysis_helpers',PRIOR/'analyze_results.py')
helpers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
summarize,arm_difference,paired_rate_difference=helpers.summarize,helpers.arm_difference,helpers.paired_rate_difference
spec=importlib.util.spec_from_file_location('spatial_prepare_helpers',PRIOR/'prepare_features.py')
preparation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
base_feature_row=preparation.base_feature_row
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import estimate_modes,modal_feature_row,modal_spatial_row,modal_spatial_addition_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
MASKS=['F','P','C','S','PC','PS','CS','PCS']
TRAIN_NAMES=['clean','iid1_seed928','iid5_seed928','iid1_seed929']
CONF_NAMES=['iid1_new943','correlated1_new943_944','iid5_new943','correlated5_new943_944',
            'iid1_new945','correlated1_new945_946','iid5_new945','correlated5_new945_946']


def columns(mask):
    if mask=='one': return np.arange(107)
    if mask in ('PCS','blend'): return np.arange(174)
    assert mask in MASKS
    extra=[171,173]
    if 'P' in mask: extra+=list(range(107,123))
    if 'C' in mask: extra+=list(range(123,171))
    if 'S' in mask: extra+=[172]
    return np.r_[np.arange(107),np.sort(extra)].astype(int)


def save_json(path,value):
    path.with_suffix('.tmp').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)


def save_npz(path,**value):
    with path.with_suffix('.tmp').open('wb') as handle: np.savez_compressed(handle,**value)
    path.with_suffix('.tmp').replace(path)


def load_data():
    with np.load(PRIOR/'features.npz') as z:
        train=np.concatenate([z['training107'],z['training_additions']],axis=2)
        evaluation=np.concatenate([z['evaluation107'],z['evaluation_additions']],axis=2)
        y,ids,rows=z['y'],z['ids'],z['source_indices']
        names=z['conditions'].tolist(); layout,pairs=z['sensor_indices'],z['sensor_pairs']
    with np.load(VERIFY/'features.npz') as z:
        for k,v in [('y',y),('ids',ids),('source_indices',rows)]: np.testing.assert_array_equal(z[k],v)
        evaluation=np.concatenate([evaluation,z['features']]); names+=z['conditions'].tolist()
    with np.load(PRIOR/'predictions.npz') as z: folds=z['row_fold']
    return train,evaluation,y,ids,rows,names,folds,layout,pairs


def source_signature():
    return {'protocol':sha256(OUT/'protocol.json'),'common':sha256(Path(__file__)),
            'prior_features':sha256(PRIOR/'features.npz'),'verification_features':sha256(VERIFY/'features.npz'),
            'fit_source':sha256(ROOT/'experiment_p4_tabular.py'),
            'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'cloudpickle':cloudpickle.__version__}}


def model_path(arm,seed,fold):
    if arm=='one': return PRIOR/f'spatial107_seed{seed}_fold{fold}.npz'
    if arm=='PCS': return PRIOR/f'two_mode174_seed{seed}_fold{fold}.npz'
    return OUT/f'{arm}_seed{seed}_fold{fold}.npz'


def trusted_predictor(path):
    with np.load(path) as z:
        meta=json.loads(str(z['metadata']))
        if not meta['success']: raise RuntimeError(f'nonconverged predictor: {path}')
        return cloudpickle.loads(z['predictor'].tobytes())  # Own trusted local artifacts only.


def query(q,rows,arm,addition_rows=None):
    cols=columns(arm.removeprefix('sham_'))
    if addition_rows is None: return q[rows][:,cols]
    return np.column_stack([q[rows,:107],q[addition_rows][:,cols[cols>=107]]])


def extract(measured,dt,layout,pairs,ridge):
    signal=np.clip(causal_calibration_normalize(measured,calibration_samples=51)[layout],-50,50).astype(np.float32)
    modes=estimate_modes(signal,dt,include_shapes=True)
    row=np.r_[base_feature_row(signal,dt,modes,ridge),modal_spatial_row(modes,pairs),modal_spatial_addition_row(modes,pairs)]
    return row,modes


def fitted_model(x,y,seed,path,signature,**arrays):
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint']))==signature,'stale fitted model'
            for k,v in arrays.items(): np.testing.assert_array_equal(z[k],v)
            return cloudpickle.loads(z['predictor'].tobytes()),json.loads(str(z['metadata']))
    predict,meta=fit_network(x,y,seed=seed,alpha=1.,max_iter=10000,max_fun=200000)
    save_npz(path,predictor=np.frombuffer(cloudpickle.dumps(predict),dtype=np.uint8),metadata=json.dumps(meta),
             fingerprint=json.dumps(signature),**arrays)
    print('Fit',path.name,json.dumps(meta),flush=True)
    return predict,meta
