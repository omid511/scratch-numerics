"""Training-only P1 budget comparisons, residual-scale selection and model freeze."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, spearmanr

from p1_improve import (ROOT, SEED, error_rows, preserve_stage_source,
                        summarize_rows, training_data, utc)
from p1_design import file_sha256, write_json
from p1_holdout import _finite_quantile
from p1_improved_models import (ImprovedLatentGP, KernelGP, load_model,
                                make_model, save_model)
from p1_surrogate import (BoundaryLatentDecoder, ParameterScaler,
                          grouped_run_splits, write_records)


class BoundaryKernelGP(ImprovedLatentGP):
    """Original boundary-envelope PCA with the same two-kernel tuning budget."""
    def fit(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        self.training_rows = rows.copy()
        self.training_runs = np.unique(dataset.run_ids[rows])
        self.training_modes = np.unique(dataset.mode_ids[rows])
        if len(self.training_runs) < 2:
            raise ValueError('At least two training geometries required')
        self.parameter_names, self.grid_shape = dataset.parameter_names, dataset.grid_shape
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.decoder = BoundaryLatentDecoder(self.latent_dim, self.seed).fit(dataset, rows)
        latent = self.decoder.encode(dataset.correction[rows])
        x = self.scaler.transform(dataset.parameters[rows])
        relative = (dataset.f_hf[rows]-dataset.f_lf[rows])/dataset.f_hf[rows]
        for mode in self.training_modes:
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            self.field_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], latent[local], seed=self.seed+int(mode))
            self.frequency_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], relative[local], seed=self.seed+1000+int(mode))
        return self

    def save(self, directory, dataset):
        super().save(directory, dataset)
        path = Path(directory)/'model.json'
        metadata = json.loads(path.read_text())
        metadata.update(model_type='boundary_kernel_gp',
                        representation_metric='Original boundary-envelope-divided PCA')
        write_json(path, metadata, overwrite=True)

    @classmethod
    def load(cls, directory):
        model = super().load(directory)
        with np.load(Path(directory)/'model.npz', allow_pickle=False) as archive:
            model.decoder = BoundaryLatentDecoder.from_arrays(
                archive, model.latent_dim, model.seed)
        return model


def budget_model(config):
    if config['family'] == 'boundary_kernel_pca':
        return BoundaryKernelGP(config['kernel'], config['latent_dim'],
                                config['seed'], config['max_iterations'])
    return make_model(config)


def load_budget_model(directory):
    config = json.loads((Path(directory)/'configuration.json').read_text())
    return (BoundaryKernelGP.load(directory) if config['family'] == 'boundary_kernel_pca'
            else load_model(directory))


def budget_candidates(protocol, repeat):
    common = {'seed': 42+repeat, 'max_iterations': protocol['selection']['gp_max_iterations']}
    result = {}
    for family in ('boundary_kernel_pca', 'output_pca', 'direct_hf', 'cokriging'):
        result[family] = [dict(name=f'{family}_{kernel}', family=family, kernel=kernel,
                              latent_dim=48, **common) for kernel in ('rbf', 'matern32')]
    result['neural_inr'] = [dict(name=f'neural_inr_lr{rate:g}', family='neural_inr',
                                learning_rate=rate, steps=protocol['selection']['neural_steps'],
                                width=64, **common) for rate in (1e-3, 3e-4)]
    return result


def target_counts(dataset, rows):
    return {mode: int(np.sum(dataset.mode_ids[rows] == mode)) for mode in range(1, 11)}


def budget_validation_splits(dataset, runs, seed):
    """Two 75/25 geometry holdouts; modal stratification uses this budget only."""
    runs = np.asarray(runs, dtype=int)
    rows = np.flatnonzero(np.isin(dataset.run_ids, runs))
    counts = target_counts(dataset, rows)
    minimum = 2 if min(counts.values()) >= 2 else 1
    rng = np.random.default_rng(seed)
    size = max(2, int(np.ceil(len(runs)*.25)))
    result, seen = [], set()
    for _ in range(1000):
        validation_runs = np.sort(rng.choice(runs, size, replace=False))
        key = tuple(validation_runs)
        if key in seen:
            continue
        training = np.flatnonzero(np.isin(dataset.run_ids, runs[~np.isin(runs, validation_runs)]))
        validation = np.flatnonzero(np.isin(dataset.run_ids, validation_runs))
        if min(target_counts(dataset, training).values()) < minimum:
            continue
        result.append((training, validation))
        seen.add(key)
        if len(result) == 2:
            return result
    # An untrainable modal budget remains an explicit negative result, rather
    # than selecting a new HF subset after inspecting outside-budget labels.
    return [(np.flatnonzero(np.isin(dataset.run_ids, runs[~np.isin(runs, v)])),
             np.flatnonzero(np.isin(dataset.run_ids, v)))
            for v in np.array_split(rng.permutation(runs), 4)[:2]]


def paired_interval(dataset, rows, baseline, prediction, seed=SEED):
    left = error_rows(dataset, rows, baseline)
    right = error_rows(dataset, rows, prediction)
    differences = []
    for run in sorted(set(dataset.run_ids[rows])):
        differences.append(np.median([r['field_rms'] for r in left if r['run'] == run])
                           -np.median([r['field_rms'] for r in right if r['run'] == run]))
    differences = np.asarray(differences)
    draw = np.random.default_rng(seed).integers(0, len(differences), (10000, len(differences)))
    samples = np.median(differences[draw], axis=1)
    return {'geometry_paired_median_difference': float(np.median(differences)),
            'bootstrap_95_percentile_interval': np.quantile(samples, [.025, .975]).tolist(),
            'positive_direction': 'Original fixed PCA48/RBF has larger error',
            'scope': 'Historical cases, not untouched final evidence; geometry resampling.'}


def learning_curves(output):
    path = output/'learning_curves'
    path.mkdir(exist_ok=False)
    protocol = json.loads((output/'protocol.json').read_text())
    train, complete, roles = training_data()
    evaluation = np.flatnonzero([roles[int(run)] == 'test' for run in complete.run_ids])
    registration = {'registered_at_utc': utc(), 'budget_protocol': protocol['learning_curves'],
        'candidates_per_family': 2, 'validation_geometry_holdouts': 2,
        'holdout_fraction': .25, 'primary_selection': 'Geometry median field RMS',
        'candidate_source': 'Fixed RBF/Matern32 alternatives, neural learning rates .001/.0003',
        'budget_original': 'Original envelope PCA48; RBF is additionally retained as the untuned audited-family baseline.',
        'isolation': 'Every fit, representation and hyperparameter comparison uses only this HF-budget subset. No winner imported from the 48-run model selection.',
        'subsets': 'Three fixed random nested permutations; do not redraw an unsuccessful modal-coverage budget.',
        'modal_stratification': 'Validation partitions may use accepted-label availability only within the already counted training budget; sparse/unserved modes remain failures.',
        'evaluation': 'Previously inspected historical test; no use in tuning or final-model selection.',
        'repeat_limit': 'At budget48 all repeats share the same HF geometries; seeds/validation splits vary, not independent HF datasets.',
        'sources': {'study': file_sha256(__file__)}}
    write_json(path/'protocol.json', registration)
    summary, all_records = [], []
    for repeat in range(protocol['learning_curves']['repeats']):
        permutation = np.random.default_rng(SEED+100+repeat).permutation(train.runs)
        for budget in protocol['learning_curves']['hf_train_budgets']:
            runs = permutation[:budget]
            rows = np.flatnonzero(np.isin(train.run_ids, runs))
            splits = budget_validation_splits(train, runs, SEED+1000+repeat*100+budget)
            trial = path/f'repeat{repeat}_hf{budget}'
            trial.mkdir()
            write_json(trial/'geometry_plan.json', {'training_runs': runs.tolist(),
                'target_counts': target_counts(train, rows),
                'validation_splits': [{'training_runs': np.unique(train.run_ids[a]).tolist(),
                                       'validation_runs': np.unique(train.run_ids[b]).tolist()}
                                      for a, b in splits]})
            predictions = {}
            for family, configs in budget_candidates(protocol, repeat).items():
                directory = trial/family
                directory.mkdir()
                candidate_reports = []
                for config in configs:
                    started = time.monotonic()
                    records = []
                    unsupported = []
                    minimum = 1 if family == 'neural_inr' else 2
                    for fold, (fit_rows, validation) in enumerate(splits):
                        counts = target_counts(train, fit_rows)
                        missing = [mode for mode, count in counts.items() if count < minimum]
                        if missing:
                            unsupported.append({'fold': fold, 'modes': missing,
                                                'required_training_targets_per_mode': minimum})
                            continue
                        model = budget_model(config).fit(train, fit_rows)
                        predicted = model.predict_rows(train, validation)
                        records.extend(error_rows(train, validation, predicted, fold=fold))
                    candidate_reports.append({'configuration': config,
                        'validation_metrics': summarize_rows(records) if not unsupported else None,
                        'unsupported_folds': unsupported,
                        'tuning_seconds': time.monotonic()-started})
                    print(f'BUDGET repeat{repeat} HF{budget} {config["name"]}: tuning {candidate_reports[-1]["tuning_seconds"]:.1f}s', flush=True)
                eligible = [r for r in candidate_reports if r['validation_metrics'] is not None]
                result = {'family': family, 'hf_budget': budget, 'repeat': repeat,
                          'candidates': candidate_reports,
                          'tuning_seconds': sum(r['tuning_seconds'] for r in candidate_reports)}
                if not eligible:
                    result.update(status='untrainable_tuning_budget', metrics=None)
                else:
                    winner = min(eligible, key=lambda r: r['validation_metrics']['geometry_median_field_rms']['median'])
                    config = winner['configuration']
                    started = time.monotonic()
                    model = budget_model(config).fit(train, rows)
                    result['fit_seconds'] = time.monotonic()-started
                    started = time.monotonic()
                    prediction = model.predict_rows(complete, evaluation)
                    result['inference_seconds'] = time.monotonic()-started
                    predictions[family] = prediction
                    records = error_rows(complete, evaluation, prediction, family=family,
                                         hf_budget=budget, repeat=repeat)
                    all_records.extend(records)
                    result.update(status='complete', selected=config, metrics=summarize_rows(records))
                    saved = dict(config, budget_implementation_sha256=file_sha256(__file__))
                    save_model(model, directory/'model', train, saved)
                    np.savez_compressed(directory/'historical_prediction.npz',
                        run_ids=complete.run_ids[evaluation], mode_ids=complete.mode_ids[evaluation], **prediction)
                write_json(directory/'results.json', result)
                summary.append(result)
            # The fixed original RBF family is not given hidden extra tuning.
            base_config = budget_candidates(protocol, repeat)['boundary_kernel_pca'][0]
            base_directory = trial/'original_fixed_rbf'
            minimum_counts = target_counts(train, rows)
            if min(minimum_counts.values()) >= 2:
                tuned = predictions.get('boundary_kernel_pca')
                tuned_result = next(r for r in summary[::-1] if r['family'] == 'boundary_kernel_pca')
                if tuned is not None and tuned_result['selected']['kernel'] == 'rbf':
                    baseline = tuned
                    source = trial/'boundary_kernel_pca/model'
                    import shutil
                    shutil.copytree(source, base_directory/'model')
                    fit_seconds = tuned_result['fit_seconds']
                else:
                    started = time.monotonic()
                    base_model = budget_model(base_config).fit(train, rows)
                    fit_seconds = time.monotonic()-started
                    baseline = base_model.predict_rows(complete, evaluation)
                    save_model(base_model, base_directory/'model', train,
                               dict(base_config, budget_implementation_sha256=file_sha256(__file__)))
                base_records = error_rows(complete, evaluation, baseline, family='original_fixed_rbf',
                                          hf_budget=budget, repeat=repeat)
                all_records.extend(base_records)
                base_result = {'family': 'original_fixed_rbf', 'hf_budget': budget, 'repeat': repeat,
                    'status': 'complete', 'metrics': summarize_rows(base_records),
                    'fit_seconds': fit_seconds, 'tuning_seconds': 0., 'configuration': base_config}
                write_json(base_directory/'results.json', base_result)
                summary.append(base_result)
                for family, prediction in predictions.items():
                    write_json(trial/family/'paired_original_comparison.json',
                               paired_interval(complete, evaluation, baseline, prediction))
            write_json(trial/'results.json', [r for r in summary if r['repeat'] == repeat and r['hf_budget'] == budget])
    write_records(path/'historical_mode_rows.csv', all_records)
    write_json(path/'results.json', {'trials': summary, 'protocol': registration})
    plot_learning_curves(path, summary)


def plot_learning_curves(path, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families = sorted({r['family'] for r in reports})
    for metric, label, name in [('geometry_median_field_rms', 'Geometry-median normalized-field RMS', 'field'),
                                ('geometry_median_frequency_error_pct', 'Geometry-median frequency error (%)', 'frequency')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for family in families:
            budgets, medians, lows, highs = [], [], [], []
            for budget in sorted({r['hf_budget'] for r in reports}):
                values = [r['metrics'][metric]['median'] for r in reports
                          if r['family'] == family and r['hf_budget'] == budget and r['status'] == 'complete']
                if values:
                    budgets.append(budget)
                    medians.append(float(np.median(values)))
                    lows.append(min(values)); highs.append(max(values))
            if budgets:
                line, = ax.plot(budgets, medians, marker='o', label=family)
                ax.fill_between(budgets, lows, highs, alpha=.12, color=line.get_color())
        ax.set(xlabel='Training HF geometries', ylabel=label,
               title='Historical benchmark; three nested repeats (band = min/max, not confidence interval)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(path/f'{name}_learning_curve.png', dpi=160)
        plt.close(fig)


@dataclass
class LocalResidualScale:
    neighbors: int = 5
    field_floor: float = 1e-6
    frequency_floor_hz: float = 1e-3

    def fit(self, dataset, rows, prediction):
        rows = np.asarray(rows, dtype=int)
        self.parameter_names = dataset.parameter_names
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.parameters = self.scaler.transform(dataset.parameters[rows])
        self.mode_ids = dataset.mode_ids[rows].copy()
        self.run_ids = dataset.run_ids[rows].copy()
        self.interior_mask = dataset.interior_mask.copy()
        self.field_squared_error = (prediction['correction']-dataset.correction[rows])[:, self.interior_mask]**2
        self.frequency_squared_relative_error = ((prediction['frequency']-dataset.f_hf[rows])/dataset.f_hf[rows])**2
        if not np.isfinite(self.field_squared_error).all() or not np.isfinite(self.frequency_squared_relative_error).all():
            raise ValueError('Residual scales require finite out-of-fold predictions')
        return self

    def predict(self, inputs, rows, prediction):
        rows = np.asarray(rows, dtype=int)
        query = self.scaler.transform(inputs.parameters[rows])
        field = np.zeros((len(rows), *inputs.grid_shape))
        frequency = np.empty(len(rows))
        for mode in np.unique(inputs.mode_ids[rows]):
            training = np.flatnonzero(self.mode_ids == mode)
            if not len(training):
                raise ValueError(f'No out-of-fold residual scale for mode{mode}')
            local = np.flatnonzero(inputs.mode_ids[rows] == mode)
            distance = np.linalg.norm(query[local, None, :]-self.parameters[training][None, :, :], axis=2)
            nearest = np.argsort(distance, axis=1, kind='stable')[:, :min(self.neighbors, len(training))]
            global_field = self.field_squared_error[training].mean(axis=0)
            global_frequency = float(self.frequency_squared_relative_error[training].mean())
            for index, indices in zip(local, nearest):
                selected = training[indices]
                distances = np.linalg.norm(query[index]-self.parameters[selected], axis=1)
                bandwidth = max(float(np.median(distances)), .05)
                weights = np.exp(-(distances-distances.min())/bandwidth)
                weights /= weights.sum()
                variance = weights@self.field_squared_error[selected]+.01*global_field
                field[index, self.interior_mask] = np.maximum(np.sqrt(variance), self.field_floor)
                relative = np.sqrt(weights@self.frequency_squared_relative_error[selected]+.01*global_frequency)
                frequency[index] = max(float(relative*prediction['frequency'][index]), self.frequency_floor_hz)
        return dict(prediction, correction_std=field, frequency_std=frequency)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/'residuals.npz', parameters=self.parameters,
            mode_ids=self.mode_ids, run_ids=self.run_ids, interior_mask=self.interior_mask,
            field_squared_error=self.field_squared_error,
            frequency_squared_relative_error=self.frequency_squared_relative_error)
        write_json(directory/'scale.json', {'neighbors': self.neighbors,
            'field_floor': self.field_floor, 'frequency_floor_hz': self.frequency_floor_hz,
            'parameter_names': self.parameter_names, 'scaler': self.scaler.state(),
            'meaning': 'Training-only cross-fitted local residual RMS; not pure epistemic uncertainty.'})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory/'scale.json').read_text())
        model = cls(metadata['neighbors'], metadata['field_floor'], metadata['frequency_floor_hz'])
        model.parameter_names = tuple(metadata['parameter_names'])
        model.scaler = ParameterScaler.from_state(metadata['scaler'])
        with np.load(directory/'residuals.npz', allow_pickle=False) as saved:
            for key in saved.files:
                setattr(model, key, saved[key])
        return model


def raw_scale(prediction, interior_mask):
    field = np.asarray(prediction['correction_std']).copy()
    frequency = np.asarray(prediction['frequency_std']).copy()
    if not np.isfinite(field).all() or not np.isfinite(frequency).all() or np.any(field < 0) or np.any(frequency < 0):
        raise ValueError('Raw uncertainty must be finite and nonnegative')
    field[:, interior_mask] = np.maximum(field[:, interior_mask], 1e-6)
    field[:, ~interior_mask] = 0.
    frequency = np.maximum(frequency, 1e-3)
    return dict(prediction, correction_std=field, frequency_std=frequency)


def conformal_state(dataset, rows, prediction):
    rows = np.asarray(rows, dtype=int)
    error = np.abs(prediction['correction']-dataset.correction[rows])
    frequency_error = np.abs(prediction['frequency']-dataset.f_hf[rows])
    if (not len(rows) or not np.isfinite(error).all() or not np.isfinite(frequency_error).all()
            or not np.isfinite(prediction['correction_std']).all()
            or not np.isfinite(prediction['frequency_std']).all()):
        raise ValueError('Calibration requires nonempty finite targets, predictions and scales')
    records = []
    for run in np.unique(dataset.run_ids[rows]):
        local = np.flatnonzero(dataset.run_ids[rows] == run)
        sigma = prediction['correction_std'][local][:, dataset.interior_mask]
        if np.any(sigma <= 0) or np.any(prediction['frequency_std'][local] <= 0):
            raise ValueError('Positive interior/frequency scales required')
        field = float(np.max(error[local][:, dataset.interior_mask]/sigma))
        frequency = float(np.max(frequency_error[local]/prediction['frequency_std'][local]))
        rms = np.sqrt(np.mean(error[local][:, dataset.interior_mask]**2, axis=1))
        rms_scale = np.sqrt(np.mean(sigma**2, axis=1))
        rms_score = float(np.max(rms/rms_scale))
        records.append({'run': int(run), 'field': field, 'frequency': frequency,
                        'joint': max(field, frequency), 'field_rms': rms_score})
    multipliers, unsupported = {}, {}
    for coverage in (90, 95):
        minimum = 9 if coverage == 90 else 19
        if len(records) < minimum:
            unsupported[str(coverage)] = (
                f'Finite {coverage}% intervals require at least {minimum} calibration runs; received {len(records)}')
        else:
            multipliers[str(coverage)] = {key: _finite_quantile([r[key] for r in records], coverage)
                                          for key in ('field', 'frequency', 'joint', 'field_rms')}
    return {'method': 'Split conformal; run maximum over accepted modes/interior pixels',
            'quantile_rule': 'ceil((n+1)*coverage/100), without interpolation',
            'runs': [r['run'] for r in records], 'scores': records,
            'multipliers': multipliers, 'unsupported': unsupported}


def coverage_interval(successes, total):
    if not total:
        return None
    return [float(beta.ppf(.025, successes, total-successes+1)) if successes else 0.,
            float(beta.ppf(.975, successes+1, total-successes)) if successes < total else 1.]


def interval_score(error, half_width, alpha):
    if not 0 < alpha < 1:
        raise ValueError('Interval-score alpha must lie in (0,1)')
    return 2*half_width+(2/alpha)*np.maximum(np.abs(error)-half_width, 0.)


def uncertainty_metrics(dataset, rows, prediction, calibration, coverage=90, joint=False):
    rows = np.asarray(rows, dtype=int)
    q = calibration['multipliers'][str(coverage)]
    alpha = 1-coverage/100
    field_width = (q['joint'] if joint else q['field'])*prediction['correction_std'][:, dataset.interior_mask]
    frequency_width = (q['joint'] if joint else q['frequency'])*prediction['frequency_std']
    field_error = (prediction['correction']-dataset.correction[rows])[:, dataset.interior_mask]
    frequency_error = prediction['frequency']-dataset.f_hf[rows]
    records = []
    for run in np.unique(dataset.run_ids[rows]):
        local = np.flatnonzero(dataset.run_ids[rows] == run)
        covered_field = bool(np.all(np.abs(field_error[local]) <= field_width[local]))
        covered_frequency = bool(np.all(np.abs(frequency_error[local]) <= frequency_width[local]))
        rms_error = np.sqrt(np.mean(field_error[local]**2, axis=1))
        rms_scale = np.sqrt(np.mean(prediction['correction_std'][local][:, dataset.interior_mask]**2, axis=1))
        records.append({'run': int(run), 'field_covered': covered_field,
            'frequency_covered': covered_frequency, 'joint_covered': covered_field and covered_frequency,
            'field_rms_ball_covered': bool(np.all(rms_error <= q['field_rms']*rms_scale)),
            'mean_field_width': float(2*field_width[local].mean()),
            'mean_frequency_width_hz': float(2*frequency_width[local].mean()),
            'mean_frequency_relative_width_pct': float(np.mean(200*frequency_width[local]/dataset.f_hf[rows[local]])),
            'mean_field_interval_score': float(interval_score(field_error[local], field_width[local], alpha).mean()),
            'mean_frequency_interval_score_hz': float(interval_score(frequency_error[local], frequency_width[local], alpha).mean()),
            'field_error_median': float(np.median(rms_error)),
            'frequency_error_median_pct': float(np.median(100*np.abs(frequency_error[local])/dataset.f_hf[rows[local]])),
            'field_uncertainty': float(np.median(rms_scale)),
            'frequency_uncertainty_relative': float(np.median(prediction['frequency_std'][local]/prediction['frequency'][local]))})
    result = {'coverage_level': coverage, 'intervals': 'joint' if joint else 'marginal',
              'geometry_count': len(records), 'per_geometry': records,
              'field_score_units': 'Peak-normalized field', 'frequency_score_units': 'Hz (proper absolute interval score)',
              'relative_width_is_descriptive_not_a_proper_score': True}
    for key in ('field_covered', 'frequency_covered', 'joint_covered', 'field_rms_ball_covered'):
        successes = sum(r[key] for r in records)
        result[key] = {'successes': successes, 'total': len(records),
                       'fraction': successes/len(records), 'clopper_pearson_95': coverage_interval(successes, len(records))}
    for key in ('mean_field_width', 'mean_frequency_width_hz', 'mean_frequency_relative_width_pct',
                'mean_field_interval_score', 'mean_frequency_interval_score_hz'):
        result[key] = float(np.mean([r[key] for r in records]))
    result['discrimination'] = {}
    result['referral_curves'] = {}
    for quantity, error_key, scale_key in (
        ('field', 'field_error_median', 'field_uncertainty'),
        ('frequency', 'frequency_error_median_pct', 'frequency_uncertainty_relative')):
        errors = np.array([r[error_key] for r in records])
        scales = np.array([r[scale_key] for r in records])
        association = spearmanr(scales, errors) if len(records) >= 3 and np.ptp(scales) > 0 and np.ptp(errors) > 0 else None
        result['discrimination'][quantity] = ({'spearman_r': float(association.statistic),
                                               'p_value': float(association.pvalue)} if association is not None else None)
        order = np.argsort(scales, kind='stable')
        result['referral_curves'][quantity] = [
            {'retained': count, 'retained_fraction': count/len(records),
             'mean_retained_error': float(errors[order[:count]].mean()),
             'maximum_retained_error': float(errors[order[:count]].max()),
             'random_referral_expected_mean': float(errors.mean())}
            for count in sorted({max(1, int(np.ceil(f*len(records)))) for f in (.25, .5, .75, 1.)})]
    return result


def uncertainty_selection(output, region=False):
    path = output/('region_uncertainty_selection' if region else 'uncertainty_selection')
    path.mkdir(exist_ok=False)
    selection_path = output/('region_selection' if region else 'selection')
    selection = json.loads((selection_path/'results.json').read_text())
    name = selection['selected']
    config = selection['candidates'][name]['configuration']
    train, _, _ = training_data()
    train = train.select_runs(selection['training_geometries'])
    registration = {'registered_at_utc': utc(), 'mean_configuration': config,
        'recipes': ['raw', 'local_residual'], 'local_neighbors': 5,
        'absolute_scale_floors': {'field': 1e-6, 'frequency_hz': 1e-3},
        'outer_geometry_folds': 4, 'outer_fit_calibration_split': 'Of each36 (or region equivalent), reserve one third for independent calibration; remaining two thirds fit mean/scale.',
        'residual_scale': 'Three-fold OOF predictions on mean-fitting geometries only; calibration and outer-validation labels excluded.',
        'selection': 'Joint90% proper field interval score improves at least5%, frequency proper score degrades no more than5%, joint empirical coverage loses no more than one geometry; otherwise raw retained.',
        'limit': 'Fixed mean selected earlier on train CV; these are post-selection training diagnostics, not unbiased final evidence. Independent prospective calibration/test determines final reported coverage.',
        'source_sha256': file_sha256(__file__)}
    write_json(path/'protocol.json', registration)
    results = {'raw': [], 'local_residual': []}
    for fold, (training, validation, training_runs, validation_runs) in enumerate(grouped_run_splits(train.run_ids, 4, SEED)):
        shuffled = np.random.default_rng(SEED+4000+fold).permutation(training_runs)
        calibration_runs = shuffled[:max(9, int(np.ceil(len(shuffled)/3)))]
        fit_runs = shuffled[len(calibration_runs):]
        fit_rows = np.flatnonzero(np.isin(train.run_ids, fit_runs))
        calibration_rows = np.flatnonzero(np.isin(train.run_ids, calibration_runs))
        if len(fit_runs) < 3 or len(calibration_runs) >= len(training_runs):
            raise ValueError('Insufficient region geometries for nested uncertainty fitting/calibration')
        directory = path/f'fold{fold}'
        directory.mkdir()
        write_json(directory/'geometry_plan.json', {'mean_scale_fit_runs': fit_runs.tolist(),
            'calibration_runs': calibration_runs.tolist(), 'validation_runs': validation_runs.tolist()})
        oof_field = np.full_like(train.correction[fit_rows], np.nan)
        oof_frequency = np.full(len(fit_rows), np.nan)
        for inner, (a, b, _, _) in enumerate(grouped_run_splits(train.run_ids[fit_rows], 3, SEED+5000+fold)):
            inner_model = make_model(config).fit(train, fit_rows[a])
            inner_prediction = inner_model.predict_rows(train, fit_rows[b])
            oof_field[b], oof_frequency[b] = inner_prediction['correction'], inner_prediction['frequency']
        scale = LocalResidualScale().fit(train, fit_rows,
                                        {'correction': oof_field, 'frequency': oof_frequency})
        scale.save(directory/'local_scale')
        model = make_model(config).fit(train, fit_rows)
        save_model(model, directory/'mean', train, config)
        calibration_prediction = model.predict_rows(train, calibration_rows)
        validation_prediction = model.predict_rows(train, validation)
        for recipe in results:
            calibrated = (raw_scale(calibration_prediction, train.interior_mask) if recipe == 'raw'
                          else scale.predict(train, calibration_rows, calibration_prediction))
            predicted = (raw_scale(validation_prediction, train.interior_mask) if recipe == 'raw'
                         else scale.predict(train, validation, validation_prediction))
            state = conformal_state(train, calibration_rows, calibrated)
            report = uncertainty_metrics(train, validation, predicted, state, 90, joint=True)
            write_json(directory/f'{recipe}_calibration.json', state)
            write_json(directory/f'{recipe}_validation.json', report)
            results[recipe].extend(report['per_geometry'])
        print(f'UNCERTAINTY fold{fold}: mean/scale {len(fit_runs)}, independent calibration {len(calibration_runs)}, validation {len(validation_runs)}', flush=True)
    metrics = {}
    for recipe, records in results.items():
        metrics[recipe] = {'geometries': len(records),
            'proper_field_interval_score': float(np.mean([r['mean_field_interval_score'] for r in records])),
            'proper_frequency_interval_score_hz': float(np.mean([r['mean_frequency_interval_score_hz'] for r in records])),
            'mean_field_width': float(np.mean([r['mean_field_width'] for r in records])),
            'mean_frequency_width_hz': float(np.mean([r['mean_frequency_width_hz'] for r in records])),
            'joint_covered': int(sum(r['joint_covered'] for r in records))}
    raw, local = metrics['raw'], metrics['local_residual']
    promote = (local['proper_field_interval_score'] <= .95*raw['proper_field_interval_score']
               and local['proper_frequency_interval_score_hz'] <= 1.05*raw['proper_frequency_interval_score_hz']
               and local['joint_covered'] >= raw['joint_covered']-1)
    selected = 'local_residual' if promote else 'raw'
    with np.load(selection_path/name/'oof.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['run_ids'], train.run_ids)
        np.testing.assert_array_equal(saved['mode_ids'], train.mode_ids)
        scale = LocalResidualScale().fit(train, np.arange(train.n_rows),
            {'correction': saved['correction'], 'frequency': saved['frequency']})
    scale.save(path/'final_local_scale')
    write_json(path/'results.json', {'selected_recipe': selected, 'promoted': bool(promote),
        'recipes': metrics, 'per_geometry': results, 'mean_configuration': config,
        'training_geometries': train.runs.tolist(), 'protocol': registration})
    print(json.dumps({'uncertainty_selected': selected, 'metrics': metrics}), flush=True)


def freeze_models(output):
    destination = output/'frozen_models'
    destination.mkdir(exist_ok=False)
    selection = json.loads((output/'selection/results.json').read_text())
    region = json.loads((output/'region_selection/results.json').read_text())
    uncertainty = json.loads((output/'uncertainty_selection/results.json').read_text())
    regional_uncertainty = json.loads((output/'region_uncertainty_selection/results.json').read_text())
    train, _, _ = training_data()
    names = sorted(set(selection['family_winners'].values()) | {selection['selected']})
    reports = {}
    for name in names+['region_selected']:
        regional = name == 'region_selected'
        entry = region['candidates'][region['selected']] if regional else selection['candidates'][name]
        config = entry['configuration']
        dataset = train.select_runs(region['training_geometries']) if regional else train
        rows = np.arange(dataset.n_rows)
        started = time.monotonic()
        model = make_model(config).fit(dataset, rows)
        fit_seconds = time.monotonic()-started
        directory = destination/name
        save_model(model, directory, dataset, config)
        # Exercise real fitted serialization, not metadata-only equality.
        probe = np.unique(np.r_[np.arange(min(10, dataset.n_rows)),
                                np.linspace(0, dataset.n_rows-1, 20, dtype=int)])
        before = model.predict_rows(dataset, probe)
        after = load_model(directory).predict_rows(dataset, probe)
        replay = {}
        for key in before:
            np.testing.assert_allclose(after[key], before[key], rtol=1e-12, atol=1e-12)
            replay[key] = float(np.max(np.abs(after[key]-before[key])))
        reports[name] = {'configuration': config, 'training_geometries': dataset.runs.tolist(),
                         'fit_seconds': fit_seconds, 'prediction_roundtrip_max_absolute': replay}
        if config['family'] == 'neural_inr':
            from scipy.interpolate import RegularGridInterpolator
            first = probe[0]
            x = np.array([0., dataset.x[-1], .113, .219])
            y = np.array([.147, .147, .087, .201])
            lf = RegularGridInterpolator((dataset.y, dataset.x), dataset.lf[first])(np.column_stack((y, x)))
            points = model.predict_points(np.repeat(dataset.parameters[first:first+1], len(x), axis=0),
                np.repeat(dataset.mode_ids[first], len(x)), x, y, lf,
                np.repeat(dataset.f_lf[first], len(x)))
            if not np.isfinite(points).all() or np.any(points[:2] != 0):
                raise ValueError('Real neural INR failed arbitrary-coordinate/clamped-boundary smoke')
            reports[name]['arbitrary_coordinate_smoke'] = {'x': x.tolist(), 'y': y.tolist(),
                'lf_source': 'Interpolation of actual LF field; no HF input', 'correction': points.tolist()}
        print(f'FREEZE {name}: trained {len(dataset.runs)} geometries; serialized predictions replayed', flush=True)
    from p1_prospective import freeze_reference
    reference = freeze_reference(output, train.parameter_names)
    threshold = float(np.median(train.f_hf[train.mode_ids == 1]))
    hashes = {path.relative_to(output).as_posix(): file_sha256(path)
              for path in sorted(destination.rglob('*')) if path.is_file()}
    for directory in ('uncertainty_selection/final_local_scale', 'region_uncertainty_selection/final_local_scale'):
        hashes.update({path.relative_to(output).as_posix(): file_sha256(path)
                       for path in sorted((output/directory).rglob('*')) if path.is_file()})
    hashes.update(reference['artifact_hashes'])
    frozen = {'frozen_at_utc': utc(), 'selected_mean': selection['selected'],
        'selected_uncertainty': uncertainty['selected_recipe'],
        'region_mean_configuration': region['candidates'][region['selected']]['configuration'],
        'region_uncertainty': regional_uncertainty['selected_recipe'],
        'family_winners': selection['family_winners'], 'models': reports,
        'frequency_screen_threshold_hz': threshold,
        'screening_confidence': 95,
        'screening_interval': 'Independently calibrated mode1-only frequency interval; do not use evaluation labels.',
        'engineering_policy': 'Lower bound>=threshold: accept. Upper bound<threshold: reject. Otherwise HF. LF identity failure: HF. Unverified automatic decisions are not successes.',
        'artifact_hashes': hashes, 'reference': reference,
        'method_source_hashes': {name: file_sha256(Path(__file__).with_name(name))
                                for name in ('p1_improve.py', 'p1_improved_models.py',
                                             'p1_improvement_study.py', 'p1_prospective.py')},
        'native_reference_settings': {'mesh_size': 4, 'n_eigs': 16, 'candidate_eigs': 32,
                                      'eigen_shift_hz': 1000., 'solver_override': None,
                                      'policy': 'Original COMSOL shell builder and default solver; no prospective solver cutover.'},
        'protocol_sha256': file_sha256(output/'protocol.json'),
        'selection_sha256': file_sha256(output/'selection/results.json'),
        'region_selection_sha256': file_sha256(output/'region_selection/results.json'),
        'uncertainty_selection_sha256': file_sha256(output/'uncertainty_selection/results.json'),
        'region_uncertainty_selection_sha256': file_sha256(output/'region_uncertainty_selection/results.json'),
        'source_sha256': file_sha256(__file__),
        'prospective_status': 'No prospective geometries or labels generated before this freeze.'}
    if (output/'prospective').exists():
        raise ValueError('Prospective directory exists before model freeze')
    write_json(output/'model_freeze.json', frozen)
    print(json.dumps({'method_frozen': True, 'threshold_hz': threshold,
                      'selected': selection['selected'], 'uncertainty': uncertainty['selected_recipe']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('learning-curves', 'uncertainty', 'uncertainty-region', 'freeze-models'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    preserve_stage_source(output, args.stage)
    if args.stage == 'learning-curves':
        learning_curves(output)
    elif args.stage.startswith('uncertainty'):
        uncertainty_selection(output, region=args.stage == 'uncertainty-region')
    else:
        freeze_models(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
