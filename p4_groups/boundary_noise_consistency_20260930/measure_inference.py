"""Measure actual frozen-feature inference; loss localization adds no inference routing."""
from experiment_helpers import *
import platform
import time
from threadpoolctl import threadpool_info,threadpool_limits


def inference(measured,dt,layout,pairs,ridge,predictors,arm):
    signal=np.clip(base.causal_calibration_normalize(measured,calibration_samples=51)[layout],-50,50).astype(np.float32)
    modes=base.estimate_modes(signal,dt,include_shapes=True)
    row=np.r_[base.base_feature_row(signal,dt,modes,ridge),base.modal_spatial_row(modes,pairs)]
    if arm!='one':row=np.r_[row,base.modal_spatial_addition_row(modes,pairs)]
    if arm=='blend':value=.5*(predictors['one'](row[None,:107])[0]+predictors['PCS'](row[None])[0])
    else:value=predictors[arm](row[None])[0]
    return float(value),row


def main():
    manifest=reference_manifest();arms=['one','PCS','blend']+ARMS
    with np.load(OUT/'fresh_features.npz') as z:
        ids,rows,layout,pairs=z['ids'],z['source_indices'],z['sensor_indices'],z['sensor_pairs'];reference=z['features'][0]
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:dt=z['dts'][rows]
    selected=np.array([np.flatnonzero(ids==d)[0] for d in np.unique(ids)]);measured=[]
    for k in selected:
        with np.load(ROOT/'p4_sensor_candidates'/f'{ids[k]}.npz') as z:
            j=int(np.flatnonzero(z['source_indices']==rows[k])[0]);raw,sigma=z['raw'][j],z['noise_scale'][j]
        white=np.random.default_rng(np.random.SeedSequence([949,int(rows[k])])).normal(size=raw.shape)
        measured.append(raw+.01*sigma*white)
    predictors={a:predictor(a,42,0) for a in arms if a!='blend'}
    expected={a:p(reference[selected,:107] if a=='one' else reference[selected]) for a,p in predictors.items()}
    expected['blend']=.5*(expected['one']+expected['PCS']);ridge=PhysicsFeatureRidge()
    for a in arms:
        for j,k in enumerate(selected):
            value,row=inference(measured[j],float(dt[k]),layout,pairs,ridge,predictors,a)
            np.testing.assert_allclose(row,reference[k,:107] if a=='one' else reference[k],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(value,expected[a][j],atol=1e-12,rtol=1e-12)
    duration=np.empty((len(arms),10,len(selected)));outputs=np.empty_like(duration)
    for repeat in range(10):
        for j,k in enumerate(selected):
            for ai in [((repeat+j)+offset)%len(arms) for offset in range(len(arms))]:
                start=time.perf_counter_ns();value,_=inference(measured[j],float(dt[k]),layout,pairs,ridge,predictors,arms[ai])
                duration[ai,repeat,j]=(time.perf_counter_ns()-start)/1e6;outputs[ai,repeat,j]=value
    for ai,a in enumerate(arms):np.testing.assert_allclose(outputs[ai],np.broadcast_to(expected[a],outputs[ai].shape),atol=1e-12,rtol=1e-12)
    save_npz(OUT/'inference_measurements.npz',durations_ms=duration,predictions=outputs,source_indices=rows[selected],arms=arms)
    hashes={a:sha256(base.model_path(a,42,0) if a in ('one','PCS') else OUT/f'{a}_seed42_fold0.npz') for a in predictors}
    result={'platform':platform.platform(),'python':platform.python_version(),'threadpools':threadpool_info(),
        'scope':json.loads((OUT/'protocol.json').read_text())['timing'],
        'arms':{a:{'median_ms':float(np.median(duration[i])),'p95_ms':float(np.quantile(duration[i],.95)),
                   'max_ms':float(duration[i].max()),'p95_below10ms':bool(np.quantile(duration[i],.95)<10)} for i,a in enumerate(arms)},
        'hashes':{'models':hashes,'source':sha256(Path(__file__)),'features':sha256(OUT/'fresh_features.npz'),
                  'measurements':sha256(OUT/'inference_measurements.npz')}}
    save_json(OUT/'inference_timing.json',result);print('Full inference timing',json.dumps(result['arms']),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
