"""Pooled two-fold descriptive screens and paired design uncertainty."""
from gap_common import *


def compare():
    d=E.setup(); output=dict(protocol=PROTOCOL,scope='Two prespecified held TRAIN folds, not five-fold or blind confirmation.',screens={})
    for name in ('w1','s1','n1'):
        arrays=[]; levels=[]; held=[]; folds=[]
        for screen in PROTOCOL['screens']:
            fold=screen['fold']
            with np.load(OUT/f'{name}_fold{fold}.npz') as z:
                arms=z['arms'].tolist(); rows=z['held_rows']; arrays.append(z['predictions']); held.extend(rows)
                levels.append(np.repeat(z['thresholds'][...,None],len(rows),axis=-1)); folds.extend([fold]*len(rows))
        p=np.concatenate(arrays,axis=2); t=np.concatenate(levels,axis=2); held=np.array(held); y=d['y'][held]; ids=d['ids'][held]
        assert len(np.unique(held))==len(held)
        result=dict(physical_clips=len(y),designs=len(np.unique(ids)),families={})
        comparisons=[(arm,'rank_products') for arm in arms[2:]] if name=='w1' else [(arm,'blend_frozen') for arm in arms[1:]]
        if name=='w1':
            comparisons += [(arm,'blend_frozen') for arm in arms[2:]]
            comparisons += [('nonlinear_b1','affine_b1'),('nonlinear_b5','affine_b5'),('nonlinear_b1','classification_b1'),('nonlinear_b5','classification_b5')]
        elif name=='s1': comparisons += [('average_tail','mse'),('worst_tail','mse'),('worst_tail','average_tail')]
        else: comparisons += [(kind+'_own','unweighted_own') for kind in ('diagonal','full','oracle')]
        groups=[('clean',['clean'])]+list(A.FAMILIES.items())
        for family,members in groups:
            indices=[d['names'].index(n) for n in members]
            value=dict(arms={arm:summary(p[i,indices],t[i],y,ids) for i,arm in enumerate(arms)},paired={})
            for candidate,reference in comparisons:
                ai,ri=arms.index(candidate),arms.index(reference); pair=dict(warning={},optimism={})
                for budget in (.01,.05):
                    wc=-p[ri,indices,:,3]<=t[ri,C.BUDGETS.index(budget)]
                    wp=-p[ai,indices,:,3]<=t[ai,C.BUDGETS.index(budget)]
                    pair['warning'][str(budget)]={}
                    for label,mask,miss in [('stable_false_warning',y>0,False),('unstable_last1_miss',(y<0)&(y>-.01),True)]:
                        a,b=(~wc[:,mask],~wp[:,mask]) if miss else (wc[:,mask],wp[:,mask])
                        pair['warning'][str(budget)][label]=C.helpers.paired_rate_difference(a,b,ids[mask])
                if name!='w1':
                    for label,mask in [('stable_last1',(y>0)&(y<.01)),('unstable_last1',(y<0)&(y>-.01)),('stable_last5',(y>0)&(y<.05))]:
                        ec=p[ri,indices][:,mask,0]-y[mask]; ep=p[ai,indices][:,mask,0]-y[mask]
                        pair['optimism'][label]=C.helpers.paired_rate_difference(ec>.03,ep>.03,ids[mask])
                value['paired'].setdefault(candidate,{})[reference]=pair
            result['families'][family]=value
            if not family.startswith('replication'):
                print('POOLED_SCREEN',name,family,json.dumps({arm:dict(
                    stable_over03=v['slices']['stable0.01']['over03'],unstable_over03=v['slices']['unstable0.01']['over03'],
                    far_under05=v['slices']['far_stable']['under05'],mae=v['slices']['all']['design_mae'],
                    miss5=v['warning']['0.05']['unstable_last1_miss']['rate'],false5=v['warning']['0.05']['stable_false_warning']['rate'],
                    miss1=v['warning']['0.01']['unstable_last1_miss']['rate'],false1=v['warning']['0.01']['stable_false_warning']['rate']) for arm,v in value['arms'].items()}),flush=True)
        output['screens'][name]=result
    C.save_json(OUT/'comparison.json',output)
    print('INDEPENDENT_SCREEN_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): compare()
