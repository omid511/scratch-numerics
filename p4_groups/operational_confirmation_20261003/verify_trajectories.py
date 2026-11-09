"""Replay actual saved continuous acquisitions and the held advisory scorecard."""
import trajectory_runner as O
import json
np,C=O.np,O.C


def verify(*,data=None,infer_views=None):
    d=O.E.setup() if data is None else data
    sets=O.partitions(d['ids']); fp=O.signature(); models,legacy=O.fit_baseline(d,sets,fp)
    with open(O.OUT/'trajectory_calibration.json') as f: archived=json.load(f)
    assert archived['fingerprint']==fp
    policy=O.load_trajectories(sets['policy']); calibration=O.load_trajectories(sets['calibration']); held=O.load_trajectories(sets['held'])
    thresholds,record=O.operational_thresholds(policy,calibration); thresholds.update(legacy)
    assert thresholds==archived['thresholds'] and record==archived['calibration']
    audits={did:json.loads((O.OUT/f'audit_{did}.json').read_text()) for did in sets['held']}
    replays=[]; maximum=0.; unsupported=[]; fit_quality={}
    for did in sets['held']:
        rows=[r for r in held if r['descriptor']['design']==did]
        if not audits[did]['acquisition_supported']:
            assert all(not r['available'].any() and np.isnan(r['point']).all() and np.all(r['reason']=='unsupported_acquisition') for r in rows)
            unsupported.append(did); continue
        assert audits[did]['rom_valid'] and audits[did]['bandwidth_supported']
        assert {r['track'] for r in audits[did]['response']}=={'fixed','ramp','forced_ramp'}
        rawfile=O.OUT/f'raw_{did}_fast_cross.npz'
        with np.load(rawfile) as z:
            assert json.loads(str(z['fingerprint']))==fp
            raw=z['response'].copy(); physical_valid=z['valid'].copy()
        assert np.array_equal(np.isfinite(raw).all(axis=0),physical_valid)
        replayed=None
        if infer_views is not None:
            noises=[O.TP.continuous_noise(raw,'fast_cross','initial_only',0,int(did[1:]),n)[0][:,:1512] for n in O.TP.NOISE]
            replayed=dict(zip(O.TP.NOISE,infer_views(raw[:,:1512],noises)))
        for noise in ('clean','correlated5','dropout'):
            row=next(r for r in rows if r['descriptor']['profile']=='fast_cross' and r['descriptor']['forcing']=='initial_only' and r['descriptor']['replicate']==0 and r['descriptor']['noise']==noise)
            injected,_=O.TP.continuous_noise(raw,'fast_cross','initial_only',0,int(did[1:]),noise)
            replay=O.infer_measured(raw[:,:1512],injected[:,:1512],models,d['pairs']) if replayed is None else replayed[noise]
            count=len(replay['times'])
            np.testing.assert_array_equal(replay['times'],row['times'][:count])
            np.testing.assert_array_equal(replay['identified'],row['identified'][:count])
            for key in ('point','score','fit_error'):
                np.testing.assert_allclose(replay[key],row[key][:count],atol=1e-10,rtol=1e-10,equal_nan=True)
            valid=np.isfinite(replay['score']); error=0. if not valid.any() else float(abs(replay['score'][valid]-row['score'][:count][valid]).max())
            maximum=max(maximum,error); replays.append(dict(design=did,noise=noise,actual_prefix_windows=count,maximum_score_error=error))
            if noise=='dropout':
                affected=np.flatnonzero((row['times']>=5.)&(row['times']<5.5))
                assert len(affected) and not row['identified'][affected].any() and np.isnan(row['point'][affected]).all()
        for noise in O.TP.NOISE:
            values=np.concatenate([r['fit_error'][np.isfinite(r['fit_error'])] for r in rows if r['descriptor']['noise']==noise])
            fit_quality[f'{did}/{noise}']=None if not len(values) else dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    # Missing observations never generate a safe output, including deadline misses.
    for row in held:
        assert np.isfinite(row['point'][row['available']]).all()
        assert np.all(row['delivery'][row['available']]-row['times'][row['available']]<=.1)
        mask=(row['reason']=='missing_input')|(row['reason']=='unsupported_acquisition')
        assert not row['available'][mask].any() and np.isnan(row['point'][mask]).all()
    before=json.loads((O.OUT/'trajectory_scorecard.json').read_text())
    after=O.scorecard(held,thresholds,audits,fp)
    assert before==after
    assert O.signature()==fp
    C.save_json(O.OUT/'trajectory_verification.json',dict(raw_acquisition_replays=replays,maximum_score_error=maximum,
        unsupported_held_designs=unsupported,empirical_policy_calibration_exact=True,held_scorecard_recomputed_exact=True,
        fit_error_diagnostics=fit_quality,fingerprint=fp,
        limitation='Numerical identification availability does not certify estimation accuracy. Full-order audits are short fixed/ramp/forced-response qualifications, not hardware validation.'))
    print('O1_TRAJECTORY_VERIFICATION_COMPLETE',len(replays),'raw acquisition replays',len(unsupported),'unsupported held designs',maximum,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): verify()
