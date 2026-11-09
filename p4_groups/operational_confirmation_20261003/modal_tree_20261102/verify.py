import json, hashlib
from pathlib import Path
import numpy as np
root=Path('/home/omid5/mechanics'); base=root/'p4_groups/operational_confirmation_20261003'; out=base/'modal_tree_20261102'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=load(out/'protocol.json'); summary=load(out/'comparison_summary.json'); results=load(out/'results.json'); comp=load(base/'raw_modal_20261010/results.json')['metrics']
with np.load(base/'support_modal_20261009_v2/dataset.npz',allow_pickle=False) as z:
    records=json.loads(str(z['records'])); truth=z['truth']; obs=z['observed']
design=np.array([r['design'] for r in records]); noise=np.array([r['noise'] for r in records]); profile=np.array([r['profile'] for r in records]); ph=sha(out/'protocol.json'); print('protocol',ph,'progress',sha(root/'P4_PROGRESS.json'))
assert sha(root/'P4_PROGRESS.json')=='ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'
assert all(sha(root/k)==v for k,v in p['inputs'].items()); assert summary['protocol_sha256']==results['protocol_sha256']==ph
points=[]
for f in range(3):
    receipt=load(out/f'fold{f}_complete.json'); assert receipt==results['fit_receipts'][f]; assert receipt['protocol_sha256']==ph and receipt['fold']==f and receipt['seed']==42
    assert receipt['model_sha256']==sha(out/f'fold{f}_model.pkl') and receipt['predictions_sha256']==sha(out/f'fold{f}_predictions.npz')
    ids=set(p['split']['train']) | (set(p['additional_designs'])-set(p['evaluation_folds'][f])); forbidden=set(p['evaluation_folds'][f])|set(p['split']['validation'])|set(p['split']['held']); assert not ids&forbidden
    with np.load(out/f'fold{f}_predictions.npz',allow_pickle=False) as z:
        proper=np.isin(design,list(ids)) & np.isin(noise,['clean','iid1']); held=np.isin(design,p['split']['held']); assert np.array_equal(proper,z['proper']) and np.array_equal(held,z['held']); assert str(z['protocol_sha256'])==ph
        point=z['predictions']; assert np.isnan(point[held]).all() and np.isfinite(point[~held]).all(); assert proper.sum()==receipt['proper_rows']==240 and proper.sum()*124==receipt['training_decisions']; points.append(point.copy())
    print('fold',f,'proper',proper.sum(),'held_rows',held.sum(),'held_nan',held.sum()*124,'seconds',receipt['elapsed_seconds'],'rss_kib',receipt['peak_rss_kib'],'model',receipt['model_sha256'],'pred',receipt['predictions_sha256'])
assert sha(out/'fold0_predictions.npz')=='56d56bb7897029ab9e12f3b1fda4323350d5b1dd9307a232001d7c8e6c4f55b3'
controls=('raw','mask','content_gru','ordered_concat','unordered_pool','framewise_reference'); checks=0; maxdiff=0
for n in p['primary_noise']:
    by={s:{} for s in ('all','boundary','profile_retreat','farther_stable')}; events={k:[0,0] for k in ('optimism','stable_wrong_sign','unstable_wrong_sign')}
    for f,ids in enumerate(p['evaluation_folds']):
        for d in ids:
            idx=np.flatnonzero((design==d)&(noise==n)); y=truth[idx]; q=points[f][idx]; o=obs[idx]; pr=profile[idx,None]
            masks={'all':np.ones(y.shape,bool),'boundary':(y>-.02)&(y<.05),'profile_retreat':np.broadcast_to(pr=='retreat',y.shape),'farther_stable':y>=.10}
            for s,m in masks.items():
                count=int(m.sum()); value=float(np.abs(q[m]-y[m]).mean()); by[s][d]=value
                saved=results['metrics'][f'fold{f}/evaluation/{n}']['modal_tree'][s]['per_design'][d]; diff=abs(value-saved['mae_by_seed'][0]); maxdiff=max(maxdiff,diff); assert diff<=1e-12 and count==saved['frames']; checks+=1
                for c in controls:
                    ref=comp[f'fold{f}/evaluation/{n}'][c][s]['per_design'][d]; assert count==ref['frames']; checks+=1
            for name,m,e in [('optimism',masks['boundary'],q-y>.02),('stable_wrong_sign',o&(y>=.01)&(y<=.05),q<=0),('unstable_wrong_sign',o&(y<=-.01)&(y>=-.05),q>0)]: events[name][0]+=int((m&e).sum()); events[name][1]+=int(m.sum())
    saved=summary['primary'][n]; vals={s:float(np.mean(list(v.values()))) for s,v in by.items()}
    for s,v in vals.items():
        diff=abs(v-saved['arms']['modal_tree']['slices'][s]['mean']); maxdiff=max(maxdiff,diff); assert diff<=1e-12; checks+=1
    for name,(ev,count) in events.items():
        v=saved['arms']['modal_tree'][name]; assert v['frames']==count and v['events_by_seed']==[ev] and abs(v['mean']-ev/count)<=1e-12; checks+=1
    print('CONDITION',n,'all',vals['all'],'boundary',vals['boundary'],'targets',saved['targets'],'events',events)
    for s in ('profile_retreat','farther_stable'):
        ds=[]
        for c in controls:
            refs={}
            for f,ids in enumerate(p['evaluation_folds']):
                rr=comp[f'fold{f}/evaluation/{n}'][c][s]['per_design']; assert set(rr)==set(ids); refs.update(rr)
            delta=float(np.mean([by[s][d]-x for d in by[s] for x in refs[d]['mae_by_seed']])); recorded=saved['tree_minus_comparators'][s][c]['mean']; diff=abs(delta-recorded); maxdiff=max(maxdiff,diff); assert diff<=1e-12; checks+=1; ds.append(c+':'+format(delta,'.9f'))
        print(' DELTA',s,' '.join(ds))
print('CHECKS',checks,'maxdiff',maxdiff,'input_hashes',len(p['inputs']),'folds',len(points),'conditions',len(p['primary_noise']),'designs_per_condition',12)
