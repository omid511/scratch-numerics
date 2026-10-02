"""Compare warm full inference; blend shares one modal extraction."""
import json
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/'p4_groups/two_mode_spatial_20260930'
sys.path[:0] = [str(OUT),str(ROOT),str(ROOT/'src'),str(ROOT/'p4_groups/correlated_training_20260929')]
import cloudpickle
import numpy as np
import sklearn
from threadpoolctl import threadpool_info,threadpool_limits
from prepare_draws import extract_row
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from run_augmentation_experiment import sha256


def inference(measured,dt,layout,pairs,ridge,predictors,arm):
    query = extract_row(measured,dt,layout,pairs,ridge,two_mode=arm!=0)
    if arm==0:
        prediction = float(predictors[0](query[None])[0])
    elif arm==1:
        prediction = float(predictors[1](query[None])[0])
    else:
        a = float(predictors[0](query[None,:107])[0])
        b = float(predictors[1](query[None])[0])
        prediction = .5*(a+b)
    return prediction,query


def main():
    manifest = json.loads((OUT/'model_manifest.json').read_text())
    assert manifest['versions'] == {'numpy':np.__version__,'sklearn':sklearn.__version__,'cloudpickle':cloudpickle.__version__}
    with np.load(OUT/'features.npz') as z:
        ids,rows,layout,pairs = z['ids'],z['source_indices'],z['sensor_indices'],z['sensor_pairs']
        ci = z['conditions'].tolist().index('iid1_fresh939')
        reference = z['features'][ci]
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:
        dts = z['dts'][rows]
    selected = np.array([np.flatnonzero(ids==d)[0] for d in np.unique(ids)])
    measured = []
    for k in selected:
        with np.load(ROOT/'p4_sensor_candidates'/f'{ids[k]}.npz') as z:
            j = int(np.flatnonzero(z['source_indices']==rows[k])[0])
            raw,sigma = z['raw'][j],z['noise_scale'][j]
        white = np.random.default_rng(np.random.SeedSequence([939,int(rows[k])])).normal(size=raw.shape)
        measured.append(raw+.01*sigma*white)
    predictors,signatures = [],{}
    for arm in ('spatial107','two_mode174'):
        name = f'{arm}_seed42_fold0.npz'
        assert sha256(PRIOR/name)==manifest['models'][name]
        with np.load(PRIOR/name) as z:
            predictors.append(cloudpickle.loads(z['predictor'].tobytes()))  # Hash-pinned trusted local models only.
        signatures[name] = manifest['models'][name]
    arms = ['spatial107','two_mode174','blend50']
    expected = [predictors[0](reference[selected,:107]),predictors[1](reference[selected])]
    expected.append(.5*(expected[0]+expected[1]))
    ridge = PhysicsFeatureRidge()
    for ai in range(3):
        for j,k in enumerate(selected):
            value,query = inference(measured[j],float(dts[k]),layout,pairs,ridge,predictors,ai)
            np.testing.assert_allclose(query,reference[k,:107] if ai==0 else reference[k],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(value,expected[ai][j],atol=1e-12,rtol=1e-12)
    durations = np.empty((3,10,len(selected)))
    outputs = np.empty_like(durations)
    for repeat in range(10):
        for j,k in enumerate(selected):
            for ai in [((repeat+j)+a)%3 for a in range(3)]:
                start = time.perf_counter_ns()
                value,_ = inference(measured[j],float(dts[k]),layout,pairs,ridge,predictors,ai)
                durations[ai,repeat,j] = (time.perf_counter_ns()-start)/1e6
                outputs[ai,repeat,j] = value
    for ai in range(3):
        np.testing.assert_allclose(outputs[ai],np.broadcast_to(expected[ai],outputs[ai].shape),atol=1e-12,rtol=1e-12)
    result = {'scope':json.loads((OUT/'protocol.json').read_text())['timing'],
              'platform':platform.platform(),'python':platform.python_version(),'threadpools':threadpool_info(),
              'physical_clips':len(selected),'repetitions':10,
              'arms':{a:{'median_ms':float(np.median(durations[ai])),
                         'p95_ms':float(np.quantile(durations[ai],.95)),
                         'max_ms':float(durations[ai].max())} for ai,a in enumerate(arms)},
              'blend_paired_overhead_ms':{ref:{'median':float(np.median(durations[2]-durations[ri])),
                                             'mean':float((durations[2]-durations[ri]).mean()),
                                             'p95':float(np.quantile(durations[2]-durations[ri],.95))}
                                          for ri,ref in enumerate(arms[:2])},
              'hashes':{'measurement':sha256(Path(__file__)),'features':sha256(OUT/'features.npz'),
                        'extractor':sha256(OUT/'prepare_draws.py'),'manifest':sha256(OUT/'model_manifest.json'),
                        'models':signatures}}
    path = OUT/'inference_measurements.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle,durations_ms=durations,predictions=outputs,source_indices=rows[selected],
                            ids=ids[selected],arms=arms)
    path.with_suffix('.tmp').replace(path)
    result['hashes']['measurements'] = sha256(path)
    path = OUT/'inference_timing.json'
    path.with_suffix('.tmp').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)
    print('Shared-extraction timing',json.dumps(result['arms']),
          'blend overhead',result['blend_paired_overhead_ms'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
