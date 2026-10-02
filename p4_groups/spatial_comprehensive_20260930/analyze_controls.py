"""Cached clean/noise/gain/timebase comparisons against same-width shams."""
from common import *
from assess_matrix import load_predictions,point,paired_metrics
from threadpoolctl import threadpool_limits


def main():
    data=load_predictions();_,_,y,ids,rows,names,folds,_,_=load_data()
    selection=json.loads((OUT/'selection.json').read_text())
    with np.load(OUT/'predictions_controls.npz') as z:
        np.testing.assert_array_equal(z['source_indices'],rows)
        np.testing.assert_array_equal(z['conditions'],names)
        for i,a in enumerate(z['arms'].tolist()):data[a]=(z['training_predictions'][i],z['heldout_predictions'][i])
    masks={'all':np.ones(len(y),bool),'far':y>=.15}
    for eps in (.01,.02,.05):masks[f'stable{eps}']=(y>0)&(y<eps);masks[f'unstable{eps}']=(y<0)&(y>-eps)
    result={'conditions':{},'hashes':{'controls':sha256(OUT/'predictions_controls.npz'),
            'analysis':sha256(Path(__file__)),'selection':sha256(OUT/'selection.json')}}
    for ci,c in enumerate(names):
        records={}
        for arm in selection['selected']:
            records[arm]={}
            for label,m in masks.items():
                target,groups=y[m],ids[m];real=data[arm][1][:,ci,:][...,m]
                records[arm][label]={'real':point(target,real,groups),
                    'sham':point(target,data['sham_'+arm][1][:,ci,:][...,m],groups),
                    'paired_differences':{a:paired_metrics(target,data[a][1][:,ci,:][...,m],real,groups)
                                          for a in ('one','PCS','blend','sham_'+arm)}}
        result['conditions'][c]=records
    save_json(OUT/'control_results.json',result)
    print('Completed paired matched-width control comparisons on all21 cached conditions',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
