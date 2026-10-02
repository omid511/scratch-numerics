"""Generate only preregistered confirmation draws after finalist lock."""
from common import *
from threadpoolctl import threadpool_limits


def main():
    selection=json.loads((OUT/'selection.json').read_text())
    _,_,y,ids,rows,_,folds,layout,pairs=load_data()
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:dt=z['dts'][rows]
    sig={**source_signature(),'execution':sha256(Path(__file__)),'selection':sha256(OUT/'selection.json'),
         'modal_source':sha256(ROOT/'src/mechanics/p4_margin_estimation/modal_features.py')}
    lookup={int(row):i for i,row in enumerate(rows)};data=np.full((8,len(y),174),np.nan);ridge=PhysicsFeatureRidge()
    for did in np.unique(ids):
        source=ROOT/'p4_sensor_candidates'/f'{did}.npz';signature={**sig,'raw':sha256(source)};path=OUT/f'confirmation_{did}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint']))==signature
                source_rows,features=z['source_indices'],z['features']
        else:
            with np.load(source) as z:raw,sigma,source_rows=z['raw'],z['noise_scale'],z['source_indices']
            features=np.empty((8,len(source_rows),174))
            for j,index in enumerate(source_rows):
                k=lookup[int(index)]
                for draw,seed in enumerate((943,945)):
                    white=np.random.default_rng(np.random.SeedSequence([seed,int(index)])).normal(size=raw[j].shape)
                    common=np.random.default_rng(np.random.SeedSequence([seed+1,int(index)])).normal(size=(1,raw[j].shape[1]))
                    corr=np.sqrt(.2)*white+np.sqrt(.8)*common
                    for local,(level,noise) in enumerate([(.01,white),(.01,corr),(.05,white),(.05,corr)]):
                        features[4*draw+local,j]=extract(raw[j]+level*sigma[j]*noise,float(dt[k]),layout,pairs,ridge)[0]
            save_npz(path,features=features,source_indices=source_rows,fingerprint=json.dumps(signature))
            print('Confirmed acquisition',did,flush=True)
        target=np.array([lookup[int(row)] for row in source_rows]);data[:,target]=features
    assert np.isfinite(data).all()
    save_npz(OUT/'confirmation_features.npz',features=data,y=y,ids=ids,source_indices=rows,row_fold=folds,
             sensor_indices=layout,sensor_pairs=pairs,conditions=CONF_NAMES,fingerprint=json.dumps(sig))
    print('New confirmation acquisition completed',data.shape,'locked',selection['selected'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
