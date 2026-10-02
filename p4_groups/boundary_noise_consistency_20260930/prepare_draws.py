"""Preregistered fresh iid/correlated draws; no structural simulation or tuning."""
from experiment_helpers import *
from threadpoolctl import threadpool_limits


def main():
    _,_,y,ids,rows,_,folds,layout,pairs=load_data();manifest=reference_manifest()
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:dt=z['dts'][rows]
    sig={**signature(),'execution':sha256(Path(__file__)),'references':manifest}
    lookup={int(row):i for i,row in enumerate(rows)};data=np.full((8,len(y),174),np.nan);ridge=PhysicsFeatureRidge()
    for did in np.unique(ids):
        source=ROOT/'p4_sensor_candidates'/f'{did}.npz';fingerprint={**sig,'raw':sha256(source)};path=OUT/f'draws_{did}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint']))==fingerprint,'stale acquisition draw'
                source_rows,features=z['source_indices'],z['features']
        else:
            with np.load(source) as z:raw,sigma,source_rows=z['raw'],z['noise_scale'],z['source_indices']
            features=np.empty((8,len(source_rows),174))
            for j,index in enumerate(source_rows):
                k=lookup[int(index)]
                for draw,seed in enumerate((949,951)):
                    white=np.random.default_rng(np.random.SeedSequence([seed,int(index)])).normal(size=raw[j].shape)
                    common=np.random.default_rng(np.random.SeedSequence([seed+1,int(index)])).normal(size=(1,raw[j].shape[1]))
                    corr=np.sqrt(.2)*white+np.sqrt(.8)*common
                    for local,(level,noise) in enumerate([(.01,white),(.01,corr),(.05,white),(.05,corr)]):
                        features[4*draw+local,j]=extract(raw[j]+level*sigma[j]*noise,float(dt[k]),layout,pairs,ridge)[0]
            save_npz(path,features=features,source_indices=source_rows,fingerprint=json.dumps(fingerprint))
            print('Fresh acquisition',did,flush=True)
        target=np.array([lookup[int(row)] for row in source_rows])
        assert set(target)==set(np.flatnonzero(ids==did))
        data[:,target]=features
    assert np.isfinite(data).all()
    save_npz(OUT/'fresh_features.npz',features=data,y=y,ids=ids,source_indices=rows,row_fold=folds,
             sensor_indices=layout,sensor_pairs=pairs,conditions=FRESH,fingerprint=json.dumps(sig))
    print('Completed fresh acquisition',data.shape,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
