"""Paired warning-rate comparisons at training-calibrated budgets, not retuning."""
from common import *
from threadpoolctl import threadpool_limits


def main():
    with np.load(OUT/'warning_predictions.npz') as z:
        p,t,budgets=z['predictions'],z['thresholds'],z['budgets']
        arms,names=z['arms'].tolist(),z['conditions'].tolist()
        y,ids,folds=z['y'],z['ids'],z['row_fold'];cal=z['calibration_rates']
    np.testing.assert_array_equal(p[arms.index('blend')],.5*(p[arms.index('one')]+p[arms.index('PCS')]))
    assert np.all(cal<=budgets[None,None,None,:])
    masks={'false_warning_all':y>0,'missed_unstable_all':y<=0,
           'false_warning_last1':(y>0)&(y<.01),'missed_unstable_last1':(y<=0)&(y>-.01)}
    result={'conditions':{},'calibration_rates':cal.tolist(),'arms':arms,'budgets':budgets.tolist(),
            'hashes':{'predictions':sha256(OUT/'warning_predictions.npz'),'analysis':sha256(Path(__file__))}}
    for ci,c in enumerate(names):
        result['conditions'][c]={}
        for bi,label in enumerate(['zero']+[str(b) for b in budgets]):
            warned=[]
            for ai,a in enumerate(arms):
                levels=0. if bi==0 else t[ai,:,folds,bi-1].T
                warned.append(p[ai,:,ci,:]<=levels)
            record={}
            for ai,arm in enumerate(arms):
                if arm=='one':continue
                record[arm]={}
                for metric,m in masks.items():
                    a,b=warned[0][:,m],warned[ai][:,m]
                    if metric.startswith('missed'):a,b=~a,~b
                    record[arm][metric]={'reference_rate':float(a.mean()),'candidate_rate':float(b.mean()),
                        'paired':paired_rate_difference(a,b,ids[m])}
            result['conditions'][c][label]=record
    save_json(OUT/'warning_comparisons.json',result)
    print('Paired warning tradeoffs completed without threshold retuning',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
