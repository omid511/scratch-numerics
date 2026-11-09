"""Descriptive paired scoring; repeated development designs are not blind evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'additive_spectral_20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(point, truth, mask, designs):
    """Average seed-wise, equal-design MAE; never score an averaged prediction."""
    per_design = {}
    for design in sorted(set(designs.tolist())):
        selected = mask & (designs == design)[:, None]
        if selected.any():
            error = point[:, selected] - truth[selected][None, :]
            per_design[design] = dict(frames=int(selected.sum()),
                mae_by_seed=np.mean(abs(error), axis=1).tolist(),
                signed_error_by_seed=error.mean(axis=1).tolist())
    total = int(mask.sum())
    equal_design = np.mean([r['mae_by_seed'] for r in per_design.values()], axis=0) if per_design else None
    unstable = mask & (truth < 0)
    sign_stable = mask & (truth >= .01) & (truth <= .05)
    sign_unstable = mask & (truth <= -.01) & (truth >= -.05)
    def event(selected, comparison):
        count = int(selected.sum())
        events = comparison(point[:, selected], truth[selected][None, :]).sum(axis=1)
        return dict(frames=count, events_by_seed=events.tolist(),
                    rate_by_seed=(events / count).tolist() if count else None)
    return dict(frames=total, designs=len(per_design), per_design=per_design,
        equal_design_mae_by_seed=equal_design.tolist() if equal_design is not None else None,
        equal_design_mae=float(equal_design.mean()) if equal_design is not None else None,
        optimism_over02=event(mask, lambda p, y: p - y > .02),
        unstable_positive_point=event(unstable, lambda p, y: p > 0),
        stable_wrong_sign=event(sign_stable, lambda p, y: p <= 0),
        unstable_exclusion_wrong_sign=event(sign_unstable, lambda p, y: p > 0))


def masks(truth, observed, times, bins):
    boundary = (truth > -.02) & (truth < .05)
    rate = abs(np.gradient(truth, times, axis=1))
    transitions = np.zeros(truth.shape, dtype=bool)
    transitions[:, 1:] = observed[:, 1:] & observed[:, :-1] & np.any(bins[:, 1:] != bins[:, :-1], axis=2)
    set_changes = np.zeros_like(transitions)
    set_changes[:, 1:] = transitions[:, 1:] & np.any(np.sort(bins[:, 1:], axis=2) != np.sort(bins[:, :-1], axis=2), axis=2)
    result = dict(all=np.ones_like(observed), current=observed, gap=~observed,
        boundary=boundary, boundary_current=boundary & observed, boundary_gap=boundary & ~observed,
        stable_side=(truth >= .01) & (truth <= .05),
        unstable_exclusion=(truth <= -.01) & (truth >= -.05),
        farther_stable10=truth >= .10, farther_stable15=truth >= .15,
        stationary=rate <= 1e-10, slow_ramp=(rate > 1e-10) & (rate <= .04), fast_ramp=rate > .04,
        bin_transition=transitions, bin_set_change=set_changes,
        mode_slot_reordering=transitions & ~set_changes, current_no_bin_transition=observed & ~transitions)
    for low, high in ((-.02, 0), (0, .01), (.01, .02), (.02, .05), (.05, .10)):
        result[f'margin_{low}_{high}'] = (truth >= low) & (truth < high)
    return result


def run():
    if (OUT / 'verification.json').exists():
        raise RuntimeError('Independently verified scores are immutable')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    training = json.loads((OUT / 'training.json').read_text())
    assert training['dataset_sha256'] == sha(OUT / 'dataset.npz')
    assert training['predictions_sha256'] == sha(OUT / 'predictions.npz')
    with np.load(OUT / 'dataset.npz', allow_pickle=False) as z:
        truth, observed, times, bins = (z[name] for name in ('truth', 'observed', 'times', 'bins'))
        records = json.loads(str(z['records']))
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as z:
        points = z['predictions']; arms = json.loads(str(z['arms'])); seeds = z['seeds'].tolist()
    assert points.shape == (len(arms), len(seeds), *truth.shape)
    assert np.isfinite(points).all()
    designs = np.array([r['design'] for r in records])
    strata = masks(truth, observed, times, bins)
    metrics, profiles, matched_margin_profiles, transitions = {}, {}, {}, {}
    conditions = []
    for split in ('train', 'validation', 'held'):
        for noise in sorted(set(r['noise'] for r in records if r['split'] == split)):
            condition = f'{split}/{noise}'
            conditions.append(condition)
            population = np.array([r['split'] == split and r['noise'] == noise for r in records])[:, None]
            metrics[condition] = {arm: {name: summary(point, truth, population & mask, designs)
                for name, mask in strata.items()} for arm, point in zip(arms, points)}
            profiles[condition] = {}
            matched_margin_profiles[condition] = {}
            for profile in ('cross', 'stable', 'retreat'):
                profile_mask = population & np.array([r['profile'] == profile for r in records])[:, None]
                profiles[condition][profile] = {arm: {name: summary(point, truth, profile_mask & strata[name], designs)
                    for name in ('all', 'boundary', 'current', 'gap')} for arm, point in zip(arms, points)}
                matched_margin_profiles[condition][profile] = {arm: {name: summary(point, truth, profile_mask & mask, designs)
                    for name, mask in strata.items() if name.startswith('margin_')} for arm, point in zip(arms, points)}
            selected = population & strata['bin_transition']
            transitions[condition] = {}
            for arm, point in zip(arms, points):
                signed_jump = np.zeros_like(point, dtype=np.float64)
                signed_jump[:, :, 1:] = np.diff(point - truth[None, :, :], axis=2)
                values = signed_jump[:, selected]
                transitions[condition][arm] = dict(frames=int(selected.sum()),
                    mean_abs_error_jump_by_seed=np.mean(abs(values), axis=1).tolist() if selected.any() else None,
                    p95_abs_error_jump_by_seed=np.quantile(abs(values), .95, axis=1).tolist() if selected.any() else None)
    paired = {}
    for arm in ('spectral_distribution', 'spatial_concentration'):
        control = 'shuffled_spectral_control' if arm.startswith('spectral') else 'shuffled_spatial_control'
        paired[arm] = {}
        for condition in conditions:
            if not condition.startswith('held/'):
                continue
            paired[arm][condition] = {}
            for stratum in ('all', 'boundary', 'current', 'boundary_current', 'gap', 'stationary', 'slow_ramp', 'fast_ramp', 'bin_transition'):
                result = {}
                for comparator in ('baseline174', control):
                    a = metrics[condition][arm][stratum]
                    b = metrics[condition][comparator][stratum]
                    per_design = {d: (np.array(a['per_design'][d]['mae_by_seed']) - b['per_design'][d]['mae_by_seed']).tolist()
                                  for d in a['per_design']}
                    result[comparator] = dict(delta_by_design_seed=per_design,
                        mean_delta=float(np.mean(list(per_design.values()))) if per_design else None,
                        improved_every_design_seed=bool(per_design and np.all(np.array(list(per_design.values())) < 0)))
                paired[arm][condition][stratum] = result
    result = dict(scope=protocol['scope'], protocol_sha256=sha(OUT / 'protocol.json'),
        dataset_sha256=sha(OUT / 'dataset.npz'), predictions_sha256=sha(OUT / 'predictions.npz'),
        scoring_source_sha256=sha(__file__), seeds=seeds, arms=arms,
        physical_trajectories=48, held_physical_trajectories=12, fitting_designs=protocol['split']['train'],
        held_designs=protocol['split']['held'], metrics=metrics, profiles=profiles,
        matched_margin_profiles=matched_margin_profiles, bin_transitions=transitions, paired=paired,
        limitation='Descriptive paired results on two repeatedly inspected held development designs; no significance, blind qualification, safety claim or averaging predictions across seeds',
        bin_transition_definition='Changes in existing strength-ordered modal-frequency nearest bins; includes mode-slot reordering, reported separately from changes in the sorted bin set',
        operational_predictor_changed=False, progress_ledger_changed=False)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for condition in conditions:
        if condition.startswith('held/'):
            print('ADDITIVE_HELD', condition, json.dumps({arm: {s: metrics[condition][arm][s]['equal_design_mae']
                for s in ('all', 'current', 'boundary', 'boundary_current')} for arm in arms}), flush=True)
    print('ADDITIVE_SCORED', len(conditions), 'conditions', flush=True)


if __name__ == '__main__':
    run()
