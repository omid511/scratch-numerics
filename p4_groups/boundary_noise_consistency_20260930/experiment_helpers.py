"""Frozen data/feature contracts for the three-arm localized consistency study."""
import importlib.util
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'p4_groups/spatial_comprehensive_20260930'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(BASE)]
spec=importlib.util.spec_from_file_location('frozen_spatial_contract',BASE/'common.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np,cloudpickle=base.np,base.cloudpickle
sha256,save_json,save_npz=base.sha256,base.save_json,base.save_npz
fit_network=base.fit_network
extract,PhysicsFeatureRidge=base.extract,base.PhysicsFeatureRidge
summarize,arm_difference,paired_rate_difference=base.summarize,base.arm_difference,base.paired_rate_difference
spec=importlib.util.spec_from_file_location('frozen_paired_metrics',BASE/'assess_matrix.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
point,paired_metrics=metrics.point,metrics.paired_metrics
ARMS=['supervised','global','localized'];SEEDS=[42,43,44]
FRESH=['iid1_new949','correlated1_new949_950','iid5_new949','correlated5_new949_950',
       'iid1_new951','correlated1_new951_952','iid5_new951','correlated5_new951_952']


def load_data():
    train,evaluation,y,ids,rows,names,folds,layout,pairs=base.load_data()
    with np.load(BASE/'confirmation_features.npz') as z:
        for key,value in [('y',y),('ids',ids),('source_indices',rows),('row_fold',folds)]:np.testing.assert_array_equal(z[key],value)
        evaluation=np.concatenate([evaluation,z['features']]);names+=z['conditions'].tolist()
    assert train.shape==(4,len(y),174) and len(np.unique(ids))==77
    return train,evaluation,y,ids,rows,names,folds,layout,pairs


def signature():
    paths={'protocol':OUT/'protocol.json','helpers':Path(__file__),
        'training_features':base.PRIOR/'features.npz','cached_verification':base.VERIFY/'features.npz',
        'cached_confirmation':BASE/'confirmation_features.npz','fit_source':ROOT/'experiment_p4_tabular.py',
        'objective':ROOT/'src/mechanics/p4_margin_estimation/skip_mlp.py',
        'scaling':ROOT/'src/mechanics/p4_margin_estimation/baselines.py',
        'modal':ROOT/'src/mechanics/p4_margin_estimation/modal_features.py'}
    return {'sources':{k:sha256(p) for k,p in paths.items()},
            'versions':base.source_signature()['versions']}


def reference_manifest():
    models={p.name:sha256(p) for a in ('one','PCS') for s in SEEDS for f in range(5) for p in [base.model_path(a,s,f)]}
    value={'models':models,'versions':base.source_signature()['versions']};path=OUT/'reference_manifest.json'
    if path.exists():assert json.loads(path.read_text())==value,'frozen reference changed'
    else:save_json(path,value)
    return value


def predictor(arm,seed,fold):
    path=base.model_path(arm,seed,fold) if arm in ('one','PCS') else OUT/f'{arm}_seed{seed}_fold{fold}.npz'
    return base.trusted_predictor(path)


def slices(y):
    masks={'all':np.ones(len(y),bool),'far':y>=.15}
    for eps in (.01,.02,.05):masks[f'stable{eps}']=(y>0)&(y<eps);masks[f'unstable{eps}']=(y<0)&(y>-eps)
    return masks
