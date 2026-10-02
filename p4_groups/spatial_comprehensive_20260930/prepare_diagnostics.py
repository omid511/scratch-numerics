"""Replay mode estimates; solver identity is retained only in diagnostic arrays."""
from common import *
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
NAMES=['clean','iid1_fresh939','iid5_fresh939','correlated_fresh939_940',
       'iid1_fresh941','iid5_fresh941','correlated_fresh941_942']
FIELDS=['estimated_modes','visible_modes','fit_residual','usable_prefix','second_energy','alpha_gap','relative_frequency_gap',
        'lead_truth_index','second_truth_index','lead_alpha_error','second_alpha_error','critical_recovered',
        'all_mode_frequency_coverage','lead_is_critical','similarity','second_available','pair_available']


def diagnostic(modes,truth,row):
    out=np.full(len(FIELDS),np.nan)
    a,f,e=modes['alpha'],modes['frequency'],modes['energy_fraction']
    visible=np.flatnonzero(e>=.01);order=visible[np.lexsort((f[visible],-a[visible]))]
    out[:4]=[len(a),len(visible),modes['relative_fit_error'],modes['usable_fraction']]
    out[7:9]=-1;out[11:14]=0;out[14:]=row[-2],row[-3],row[-1]
    if len(order)>1:
        i,j=order[:2];out[4:7]=e[j],a[i]-a[j],2*abs(f[i]-f[j])/(f[i]+f[j])
    if len(a):
        freq=truth.imag/(2*np.pi);cost=abs(f[:,None]-freq[None])/np.maximum(2.,.05*freq)[None]
        est,actual=linear_sum_assignment(cost);valid=cost[est,actual]<=1
        mapping={int(i):int(j) for i,j in zip(est[valid],actual[valid])}
        out[11]=int(0 in mapping.values());out[12]=len(mapping)/len(truth)
        for n,index in enumerate(order[:2]):
            if int(index) in mapping:
                target=mapping[int(index)];out[7+n]=target;out[9+n]=a[index]-truth[target].real
        out[13]=int(len(order)>0 and mapping.get(int(order[0]))==0)
    return out


def main():
    _,evaluation,y,ids,rows,names,folds,layout,pairs=load_data()
    lookup={int(row):i for i,row in enumerate(rows)}
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:dt=z['dts'][rows]
    signature={**source_signature(),'execution':sha256(Path(__file__)),
               'modal_source':sha256(ROOT/'src/mechanics/p4_margin_estimation/modal_features.py')}
    values=np.full((len(NAMES),len(y),len(FIELDS)),np.nan);max_delta=0.;ridge=PhysicsFeatureRidge()
    for did in np.unique(ids):
        source=ROOT/'p4_sensor_candidates'/f'{did}.npz';sig={**signature,'raw':sha256(source)}
        path=OUT/f'diagnostic_{did}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint']))==sig
                source_rows,data,delta=z['source_indices'],z['diagnostics'],float(z['replay_delta'])
        else:
            with np.load(source) as z:raw,sigma,source_rows,truth=z['raw'],z['noise_scale'],z['source_indices'],z['poles']
            data=np.full((len(NAMES),len(source_rows),len(FIELDS)),np.nan);delta=0.
            for j,index in enumerate(source_rows):
                k=lookup[int(index)]
                white={s:np.random.default_rng(np.random.SeedSequence([s,int(index)])).normal(size=raw[j].shape) for s in (939,941)}
                shared={s:np.random.default_rng(np.random.SeedSequence([s,int(index)])).normal(size=(1,raw[j].shape[1])) for s in (940,942)}
                for ci,name in enumerate(NAMES):
                    if name=='clean':measured=raw[j]
                    else:
                        seed=939 if '939' in name else 941;level=.05 if name.startswith('iid5') else .01
                        noise=white[seed] if name.startswith('iid') else np.sqrt(.2)*white[seed]+np.sqrt(.8)*shared[seed+1]
                        measured=raw[j]+level*sigma[j]*noise
                    row,modes=extract(measured,float(dt[k]),layout,pairs,ridge)
                    ref=evaluation[names.index(name),k]
                    np.testing.assert_allclose(row,ref,atol=1e-10,rtol=1e-10)
                    delta=max(delta,float(abs(row-ref).max()));data[ci,j]=diagnostic(modes,truth[j],row)
            save_npz(path,source_indices=source_rows,diagnostics=data,replay_delta=delta,fingerprint=json.dumps(sig))
            print('Diagnosed',did,flush=True)
        target=np.array([lookup[int(row)] for row in source_rows]);values[:,target]=data;max_delta=max(max_delta,delta)
    save_npz(OUT/'diagnostics.npz',diagnostics=values,fields=FIELDS,conditions=NAMES,y=y,ids=ids,
             source_indices=rows,row_fold=folds,replay_delta=max_delta,fingerprint=json.dumps(signature))
    print('Diagnostic replay completed',values.shape,'max174 delta',max_delta,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
