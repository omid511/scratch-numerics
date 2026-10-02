"""Warm raw-sensor-to-native-prediction timing, with one shared modal estimate."""
from common import *
import platform
import time
from threadpoolctl import threadpool_info,threadpool_limits


def inference(measured,dt,layout,pairs,ridge,predictors,arm):
    signal=np.clip(causal_calibration_normalize(measured,calibration_samples=51)[layout],-50,50).astype(np.float32)
    modes=estimate_modes(signal,dt,include_shapes=True)
    row=np.r_[base_feature_row(signal,dt,modes,ridge),modal_spatial_row(modes,pairs)]
    if arm!='one':row=np.r_[row,modal_spatial_addition_row(modes,pairs)]
    if arm=='blend':value=.5*(predictors['one'](row[None,:107])[0]+predictors['PCS'](row[None])[0])
    else:value=predictors[arm](row[None,columns(arm)])[0]
    return float(value),row


def main():
    selection=json.loads((OUT/'selection.json').read_text());arms=['one','PCS','blend']+selection['selected']
    with np.load(OUT/'confirmation_features.npz') as z:
        ids,rows,layout,pairs=z['ids'],z['source_indices'],z['sensor_indices'],z['sensor_pairs']
        reference=z['features'][z['conditions'].tolist().index('iid1_new943')]
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:dt=z['dts'][rows]
    chosen=np.array([np.flatnonzero(ids==d)[0] for d in np.unique(ids)]);measured=[]
    for k in chosen:
        with np.load(ROOT/'p4_sensor_candidates'/f'{ids[k]}.npz') as z:
            j=int(np.flatnonzero(z['source_indices']==rows[k])[0]);raw,sigma=z['raw'][j],z['noise_scale'][j]
        white=np.random.default_rng(np.random.SeedSequence([943,int(rows[k])])).normal(size=raw.shape)
        measured.append(raw+.01*sigma*white)
    predictors={};hashes={}
    for arm in arms:
        if arm=='blend':continue
        path=model_path(arm,42,0);hashes[path.name]=sha256(path)
        if arm in selection['selected']:assert hashes[path.name]==selection['model_hashes'][arm]['42_0']
        predictors[arm]=trusted_predictor(path)
    expected={a:predictors[a](reference[chosen][:,columns(a)]) for a in predictors}
    expected['blend']=.5*(expected['one']+expected['PCS']);ridge=PhysicsFeatureRidge()
    for a in arms:
        for j,k in enumerate(chosen):
            value,row=inference(measured[j],float(dt[k]),layout,pairs,ridge,predictors,a)
            np.testing.assert_allclose(row,reference[k,:107] if a=='one' else reference[k],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(value,expected[a][j],atol=1e-12,rtol=1e-12)
    duration=np.empty((len(arms),10,len(chosen)));outputs=np.empty_like(duration)
    for repeat in range(10):
        for j,k in enumerate(chosen):
            for ai in [((repeat+j)+offset)%len(arms) for offset in range(len(arms))]:
                start=time.perf_counter_ns();value,_=inference(measured[j],float(dt[k]),layout,pairs,ridge,predictors,arms[ai])
                duration[ai,repeat,j]=(time.perf_counter_ns()-start)/1e6;outputs[ai,repeat,j]=value
    for ai,a in enumerate(arms):np.testing.assert_allclose(outputs[ai],np.broadcast_to(expected[a],outputs[ai].shape),atol=1e-12,rtol=1e-12)
    save_npz(OUT/'inference_measurements.npz',durations_ms=duration,predictions=outputs,source_indices=rows[chosen],ids=ids[chosen],arms=arms)
    result={'platform':platform.platform(),'python':platform.python_version(),'threadpools':threadpool_info(),
        'scope':json.loads((OUT/'protocol.json').read_text())['timing'],
        'reduced_extraction_note':'Reduced timing includes constructing the full67-column addition before selecting its subset.',
        'arms':{a:{'median_ms':float(np.median(duration[i])),'p95_ms':float(np.quantile(duration[i],.95)),
                   'max_ms':float(duration[i].max()),'p95_below10ms':bool(np.quantile(duration[i],.95)<10)} for i,a in enumerate(arms)},
        'paired_overhead_vs_one':{a:{'median_ms':float(np.median(duration[i]-duration[0])),
                                     'p95_ms':float(np.quantile(duration[i]-duration[0],.95))} for i,a in enumerate(arms[1:],1)},
        'hashes':{'models':hashes,'execution':sha256(Path(__file__)),'common':sha256(OUT/'common.py'),
                  'features':sha256(OUT/'confirmation_features.npz'),'measurements':sha256(OUT/'inference_measurements.npz')}}
    save_json(OUT/'inference_timing.json',result);print('Full pipeline latency',json.dumps(result['arms']),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
