"""Screen the fixed observation-dependent hybrid without selecting on held data."""
import json
from pathlib import Path
import numpy as np
import temporal_convergence as C


def run():
    result = json.loads((C.OUT / 'results.json').read_text())
    assert result['source_sha256'] == C.sha256(Path(C.__file__))
    assert result['protocol'] == json.loads((C.OUT / 'protocol.json').read_text())
    for name, expected in result['helper_source_sha256'].items():
        assert C.sha256(C.SCRIPT / name) == expected
    with np.load(C.OUT / 'predictions.npz') as z:
        assert json.loads(str(z['protocol'])) == result['protocol']
        records = json.loads(str(z['records']))
        assert records == result['records']
        rows = np.array([r['split'] == 'validation' for r in records])
        dataset = {k: z[k][rows].copy() for k in ('truth', 'observed', 'age')}
        dataset['records'] = [r for r in records if r['split'] == 'validation']
        point = z['predictions'][:, rows].copy()
    assert np.isfinite(point).all()
    # Fixed causal rule: measured-window flag only; no target-dependent switching.
    hybrid = np.where(dataset['observed'], point[0], point[1])
    evaluation = {kind: C.evaluate(prediction, dataset)['validation'] for kind, prediction in
                  (('framewise', point[0]), ('gru', point[1]), ('hybrid', hybrid))}
    for kind in ('framewise', 'gru'):
        assert evaluation[kind] == result['models'][kind]['evaluation']['validation']
    better = evaluation['hybrid']['all']['mae'] < evaluation['framewise']['all']['mae']
    payload = dict(rule='Accepted fresh input -> framewise; missing input -> GRU extrapolation',
                   scope='Fixed-rule validation screen, no fitted gate or warning policy',
                   source_sha256=C.sha256(Path(__file__)), result_sha256=C.sha256(C.OUT / 'results.json'),
                   evaluation=evaluation, validation_mae_better_than_framewise=better,
                   held_hybrid_evaluated=False,
                   decision='Validation MAE passes; conditional risks still require review' if better else
                   'Reject this hybrid: validation MAE worsens versus framewise; do not select on held gains',
                   limitations=['Validation was already used for checkpoint selection; this is not blind confirmation',
                                'Gap predictions remain inferred, never measured availability',
                                'No new warning threshold or safety qualification'])
    C.T.M.G.O.C.save_json(C.OUT / 'hybrid_validation_screen.json', C.T.M.clean_json(payload))
    for kind, values in evaluation.items():
        print('HYBRID_VALIDATION', kind, json.dumps({key: values[key] for key in
              ('all', 'missing', 'stable_last2pct', 'unstable', 'by_design')}))
    print('HYBRID_DECISION', payload['decision'])


if __name__ == '__main__':
    run()
