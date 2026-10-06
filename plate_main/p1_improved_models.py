#!/usr/bin/env python3
"""Controlled P1 alternatives; the audited surrogate module remains unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from p1_design import file_sha256, write_json
from p1_surrogate import (BoundaryLatentDecoder, ExactMultiOutputGP, LatentGPModel,
                          LinearCoKrigingBaseline, ParameterScaler, randomized_svd)


class OutputMetricDecoder(BoundaryLatentDecoder):
    """PCA minimizing the reported physical interior RMS, not envelope-divided RMS.

    The representation still uses the original boundary envelope at decoding;
    only the training/encoding inner product changes. No boundary is relaxed.
    """

    def fit(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        if not len(rows):
            raise ValueError('Decoder needs training rows')
        self.x, self.y = dataset.x.copy(), dataset.y.copy()
        self.boundary_mask = dataset.boundary_mask.copy()
        self.interior_mask = dataset.interior_mask.copy()
        self.envelope = self.boundary_mask/self.boundary_mask.max()
        target = dataset.correction[rows][:, self.interior_mask]
        mean = target.mean(axis=0)
        self.singular_values, components = randomized_svd(target-mean, self.latent_dim, seed=self.seed)
        envelope = self.envelope[self.interior_mask]
        self.mean, self.components = mean/envelope, components/envelope[None, :]
        reconstructed = mean+((target-mean)@components.T)@components
        self.residual_variance = np.var(target-reconstructed, axis=0)
        return self

    def encode(self, fields):
        self._check_fit()
        fields = np.asarray(fields, dtype=float)
        if fields.ndim != 3 or fields.shape[1:] != self.envelope.shape:
            raise ValueError('Field shape does not match fitted decoder')
        envelope = self.envelope[self.interior_mask]
        return ((fields[:, self.interior_mask]-self.mean*envelope[None, :])
                @(self.components*envelope[None, :]).T)


class KernelGP(ExactMultiOutputGP):
    def __init__(self, kernel='rbf', max_iterations=80):
        if kernel not in ('rbf', 'matern32'):
            raise ValueError(f'Unsupported kernel: {kernel}')
        super().__init__(max_iterations)
        self.kernel = kernel

    def _kernel(self, x1, x2, length_scales, signal):
        if self.kernel == 'rbf':
            return super()._kernel(x1, x2, length_scales, signal)
        distance = np.sqrt(np.sum(((x1[:, None, :]-x2[None, :, :])/length_scales)**2, axis=2))
        scaled = np.sqrt(3.)*distance
        return signal**2*(1+scaled)*np.exp(-scaled)


def restore_gp(arrays, prefix, kernel):
    saved = ExactMultiOutputGP.from_arrays(arrays, prefix)
    gp = KernelGP(kernel)
    for key, value in vars(saved).items():
        setattr(gp, key, value)
    return gp


class ImprovedLatentGP(LatentGPModel):
    def __init__(self, kernel='rbf', latent_dim=48, seed=42, max_iterations=80):
        super().__init__(latent_dim, seed, max_iterations)
        self.kernel = kernel

    def fit(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        self.training_rows = rows.copy()
        self.training_runs = np.unique(dataset.run_ids[rows])
        self.training_modes = np.unique(dataset.mode_ids[rows])
        if len(self.training_runs) < 2:
            raise ValueError('At least two training geometries required')
        self.parameter_names, self.grid_shape = dataset.parameter_names, dataset.grid_shape
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.decoder = OutputMetricDecoder(self.latent_dim, self.seed).fit(dataset, rows)
        latent = self.decoder.encode(dataset.correction[rows])
        x = self.scaler.transform(dataset.parameters[rows])
        relative_frequency = (dataset.f_hf[rows]-dataset.f_lf[rows])/dataset.f_hf[rows]
        for mode in self.training_modes.tolist():
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            self.field_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], latent[local], seed=self.seed+int(mode))
            self.frequency_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], relative_frequency[local], seed=self.seed+1000+int(mode))
        return self

    def save(self, directory, dataset):
        super().save(directory, dataset)
        path = Path(directory)/'model.json'
        metadata = json.loads(path.read_text())
        metadata.update(model_type='output_metric_latent_gp', kernel=self.kernel,
                        representation_metric='physical interior RMS; original boundary envelope retained')
        write_json(path, metadata, overwrite=True)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'model.json').read_text())
        with np.load(directory/'model.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        model = cls(meta['kernel'], meta['latent_dim'], meta['seed'])
        model.decoder = OutputMetricDecoder.from_arrays(arrays, meta['latent_dim'], meta['seed'])
        model.scaler = ParameterScaler.from_state(meta['scaler'])
        model.parameter_names = tuple(meta['parameter_names'])
        model.training_runs = np.asarray(meta['training_runs'])
        model.training_rows = np.asarray(meta['training_row_indices'])
        model.training_modes = np.asarray(meta['modes'])
        model.grid_shape = tuple(meta['grid_shape'])
        for mode in model.training_modes:
            model.field_gps[int(mode)] = restore_gp(arrays, f'field_{mode}_', model.kernel)
            model.frequency_gps[int(mode)] = restore_gp(arrays, f'frequency_{mode}_', model.kernel)
        return model


class DirectHFModel:
    """No LF information in the fitted design-to-HF field/frequency functions."""

    def __init__(self, kernel='rbf', max_iterations=80, seed=42):
        self.kernel, self.max_iterations, self.seed = kernel, max_iterations, seed
        self.field_gps, self.frequency_gps = {}, {}

    def fit(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        self.training_rows = rows.copy()
        self.training_runs = np.unique(dataset.run_ids[rows])
        self.parameter_names = dataset.parameter_names
        self.interior_mask = dataset.interior_mask.copy()
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        x = self.scaler.transform(dataset.parameters[rows])
        for mode in np.unique(dataset.mode_ids[rows]):
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            self.field_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], dataset.hf[rows[local]][:, self.interior_mask],
                optimize_columns=256, seed=self.seed+int(mode))
            self.frequency_gps[int(mode)] = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], np.log(dataset.f_hf[rows[local]]), seed=self.seed+1000+int(mode))
        return self

    def predict_rows(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        field = np.zeros((len(rows), *dataset.grid_shape))
        std = np.zeros_like(field)
        frequency, frequency_std = np.zeros(len(rows)), np.zeros(len(rows))
        mask = np.flatnonzero(self.interior_mask.ravel())
        for mode in np.unique(dataset.mode_ids[rows]):
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            x = self.scaler.transform(dataset.parameters[rows[local]])
            mean, sigma = self.field_gps[int(mode)].predict(x)
            field.reshape(len(rows), -1)[np.ix_(local, mask)] = mean
            std.reshape(len(rows), -1)[np.ix_(local, mask)] = sigma
            mean_log, std_log = self.frequency_gps[int(mode)].predict(x)
            frequency[local] = np.exp(mean_log[:, 0])
            frequency_std[local] = frequency[local]*std_log[:, 0]
        if not np.isfinite(frequency).all():
            raise ValueError('Nonfinite direct-HF frequency prediction')
        # LF is subtracted solely to use the existing correction evaluator.
        # Adding it back recovers the same HF prediction for any LF input.
        return {'correction': field-dataset.lf[rows], 'correction_std': std,
                'frequency': frequency, 'frequency_std': frequency_std}

    def save(self, directory, dataset):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'interior_mask': self.interior_mask}
        for mode in self.field_gps:
            arrays.update(self.field_gps[mode].state_arrays(f'field_{mode}_'))
            arrays.update(self.frequency_gps[mode].state_arrays(f'frequency_{mode}_'))
        np.savez_compressed(directory/'model.npz', **arrays)
        write_json(directory/'model.json', {'model_type': 'direct_hf', 'kernel': self.kernel,
            'seed': self.seed, 'scaler': self.scaler.state(), 'parameter_names': self.parameter_names,
            'training_runs': self.training_runs.tolist(), 'training_rows': self.training_rows.tolist(),
            'modes': sorted(self.field_gps), 'lf_information_in_fit': False})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'model.json').read_text())
        with np.load(directory/'model.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        model = cls(meta['kernel'], seed=meta['seed'])
        model.scaler = ParameterScaler.from_state(meta['scaler'])
        model.parameter_names = tuple(meta['parameter_names'])
        model.training_runs, model.training_rows = np.asarray(meta['training_runs']), np.asarray(meta['training_rows'])
        model.interior_mask = arrays['interior_mask'].astype(bool)
        for mode in meta['modes']:
            model.field_gps[mode] = restore_gp(arrays, f'field_{mode}_', model.kernel)
            model.frequency_gps[mode] = restore_gp(arrays, f'frequency_{mode}_', model.kernel)
        return model


class CoKrigingModel(LinearCoKrigingBaseline):
    def __init__(self, kernel='rbf', max_iterations=80, seed=42):
        super().__init__(max_iterations, seed)
        self.kernel = kernel

    def fit(self, dataset, rows):
        rows = np.asarray(rows, dtype=int)
        self.training_rows = rows.copy()
        self.training_runs = np.unique(dataset.run_ids[rows])
        self.parameter_names = dataset.parameter_names
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.interior_mask, self.boundary_mask = dataset.interior_mask.copy(), dataset.boundary_mask.copy()
        x = self.scaler.transform(dataset.parameters[rows])
        for mode in np.unique(dataset.mode_ids[rows]):
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            lf, hf = dataset.lf[rows[local]][:, self.interior_mask], dataset.hf[rows[local]][:, self.interior_mask]
            rho = float(np.sum(lf*hf)/max(np.sum(lf*lf), 1e-30))
            gp = KernelGP(self.kernel, self.max_iterations).fit(
                x[local], hf-rho*lf, optimize_columns=256, seed=self.seed+int(mode))
            relative = (dataset.f_hf[rows[local]]-dataset.f_lf[rows[local]])/dataset.f_hf[rows[local]]
            fgp = KernelGP(self.kernel, self.max_iterations).fit(x[local], relative, seed=self.seed+1000+int(mode))
            self.mode_models[int(mode)], self.frequency_gps[int(mode)] = (rho, gp), fgp
        return self

    def save(self, directory, dataset):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'interior_mask': self.interior_mask, 'boundary_mask': self.boundary_mask}
        for mode, (rho, gp) in self.mode_models.items():
            arrays[f'rho_{mode}'] = np.asarray(rho)
            arrays.update(gp.state_arrays(f'field_{mode}_'))
            arrays.update(self.frequency_gps[mode].state_arrays(f'frequency_{mode}_'))
        np.savez_compressed(directory/'model.npz', **arrays)
        write_json(directory/'model.json', {'model_type': 'cokriging', 'kernel': self.kernel,
            'seed': self.seed, 'scaler': self.scaler.state(), 'parameter_names': self.parameter_names,
            'training_runs': self.training_runs.tolist(), 'training_rows': self.training_rows.tolist(),
            'modes': sorted(self.mode_models)})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'model.json').read_text())
        with np.load(directory/'model.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        model = cls(meta['kernel'], seed=meta['seed'])
        model.scaler = ParameterScaler.from_state(meta['scaler'])
        model.parameter_names = tuple(meta['parameter_names'])
        model.training_runs, model.training_rows = np.asarray(meta['training_runs']), np.asarray(meta['training_rows'])
        model.interior_mask, model.boundary_mask = arrays['interior_mask'].astype(bool), arrays['boundary_mask']
        for mode in meta['modes']:
            model.mode_models[mode] = (float(arrays[f'rho_{mode}']), restore_gp(arrays, f'field_{mode}_', model.kernel))
            model.frequency_gps[mode] = restore_gp(arrays, f'frequency_{mode}_', model.kernel)
        return model


def neural_networks(point_features, design_features, width):
    import torch.nn as nn
    def network(features, hidden):
        return nn.Sequential(nn.Linear(features, hidden), nn.Tanh(),
                             nn.Linear(hidden, hidden), nn.Tanh(),
                             nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    return network(point_features, width), network(design_features, width)


class NeuralINR:
    """Actual learned MLP residual, conditioned on geometry, mode and LF field.

    Fixed sinusoidal coordinate features help resolve modal oscillations;
    learned nonlinear layers distinguish this from the old ridge proxy.
    Geometry groups, not spatial points, define all validation splits.
    """

    def __init__(self, learning_rate=1e-3, steps=1500, width=64, seed=42):
        self.learning_rate, self.steps, self.width, self.seed = learning_rate, steps, width, seed

    def _design_features(self, parameters, mode_ids, f_lf):
        modes = np.asarray(mode_ids, dtype=int)
        if np.any(modes < 1) or np.any(modes > self.n_modes):
            raise ValueError('Untrained neural mode ID')
        geometry = self.scaler.transform(parameters)
        one_hot = np.eye(self.n_modes)[modes-1]
        log_frequency = (np.log(np.asarray(f_lf))-self.log_f_mean)/self.log_f_scale
        return np.column_stack((geometry, one_hot, log_frequency)).astype(np.float32)

    def _point_features(self, design, coordinates, lf_values):
        coordinates = np.asarray(coordinates, dtype=np.float32)
        features = [design, coordinates, np.asarray(lf_values, dtype=np.float32)[:, None]]
        for harmonic in range(1, 7):
            features += [np.sin(np.pi*harmonic*coordinates), np.cos(np.pi*harmonic*coordinates)]
        return np.column_stack(features).astype(np.float32)

    def fit(self, dataset, rows):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rows = np.asarray(rows, dtype=int)
        self.training_rows, self.training_runs = rows.copy(), np.unique(dataset.run_ids[rows])
        self.parameter_names = dataset.parameter_names
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.n_modes = int(dataset.mode_ids.max())
        self.x, self.y = dataset.x.copy(), dataset.y.copy()
        self.interior_mask, self.boundary_mask = dataset.interior_mask.copy(), dataset.boundary_mask.copy()
        self.envelope_max = float(self.boundary_mask.max())
        xx, yy = np.meshgrid((self.x-self.x[0])/(self.x[-1]-self.x[0]),
                             (self.y-self.y[0])/(self.y[-1]-self.y[0]), indexing='xy')
        coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        envelope = (self.boundary_mask/self.envelope_max).ravel().astype(np.float32)
        lf_matrix = dataset.lf.reshape(dataset.n_rows, -1)
        correction_matrix = dataset.correction.reshape(dataset.n_rows, -1)
        self.log_f_mean = float(np.log(dataset.f_lf[rows]).mean())
        self.log_f_scale = max(float(np.log(dataset.f_lf[rows]).std()), 1e-6)
        self.field_scale = np.ones(self.n_modes)
        self.frequency_mean, self.frequency_scale = np.zeros(self.n_modes), np.ones(self.n_modes)
        ratio = np.log(dataset.f_hf[rows]/dataset.f_lf[rows])
        for mode in np.unique(dataset.mode_ids[rows]):
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            self.field_scale[mode-1] = max(float(np.sqrt(np.mean(dataset.correction[rows[local]][:, self.interior_mask]**2))), 1e-5)
            self.frequency_mean[mode-1] = float(ratio[local].mean())
            self.frequency_scale[mode-1] = max(float(ratio[local].std()), 1e-4)
        design = self._design_features(dataset.parameters[rows], dataset.mode_ids[rows], dataset.f_lf[rows])
        normalized_frequency = ((ratio-self.frequency_mean[dataset.mode_ids[rows]-1])
                                /self.frequency_scale[dataset.mode_ids[rows]-1]).astype(np.float32)
        self.field_network, self.frequency_network = neural_networks(design.shape[1]+27, design.shape[1], self.width)
        optimizer = torch.optim.Adam(list(self.field_network.parameters())+list(self.frequency_network.parameters()), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        groups = [np.flatnonzero(dataset.run_ids[rows] == run) for run in self.training_runs]
        spatial = np.flatnonzero(self.interior_mask.ravel())
        start = time.monotonic()
        self.field_network.train()
        self.frequency_network.train()
        for step in range(self.steps):
            selected = np.array([rng.choice(groups[index]) for index in rng.integers(len(groups), size=8)])
            pixel = rng.choice(spatial, size=(len(selected), 256), replace=True)
            local = np.repeat(selected, pixel.shape[1])
            flat = pixel.ravel()
            inputs = self._point_features(design[local], coordinates[flat], lf_matrix[rows[local], flat])
            target = correction_matrix[rows[local], flat]
            target = (target/self.field_scale[dataset.mode_ids[rows[local]]-1]).astype(np.float32)
            prediction = self.field_network(torch.from_numpy(inputs)).squeeze(1)*torch.from_numpy(envelope[flat])
            frequency_prediction = self.frequency_network(torch.from_numpy(design[selected])).squeeze(1)
            loss = torch.mean((prediction-torch.from_numpy(target))**2)
            loss += torch.mean((frequency_prediction-torch.from_numpy(normalized_frequency[selected]))**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 500 == 0 or step == self.steps-1:
                print(f'NEURAL step{step+1}/{self.steps}: loss={loss.item():.6g}; elapsed={time.monotonic()-start:.1f}s', flush=True)
        self.field_network.eval()
        self.frequency_network.eval()
        self.field_residual_std = np.zeros((self.n_modes, *dataset.grid_shape))
        self.frequency_residual_scale = np.ones(self.n_modes)*1e-4
        # Training residuals are an explicitly labeled scale proxy; they are
        # not an epistemic posterior. Independent calibration follows later.
        prediction = self.predict_rows(dataset, rows)
        for mode in np.unique(dataset.mode_ids[rows]):
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            error = prediction['correction'][local]-dataset.correction[rows[local]]
            self.field_residual_std[mode-1] = np.sqrt(np.mean(error**2, axis=0))
            freq_error = np.log(prediction['frequency'][local]/dataset.f_hf[rows[local]])
            self.frequency_residual_scale[mode-1] = max(float(np.sqrt(np.mean(freq_error**2))), 1e-5)
        return self

    def predict_points(self, parameters, mode_ids, x, y, lf_values, f_lf):
        import torch
        mode_ids = np.asarray(mode_ids, dtype=int)
        x, y = np.asarray(x), np.asarray(y)
        if (np.any(x < self.x[0]) or np.any(x > self.x[-1])
                or np.any(y < self.y[0]) or np.any(y > self.y[-1])):
            raise ValueError('Neural coordinates outside declared physical domain')
        xu, yu = (x-self.x[0])/(self.x[-1]-self.x[0]), (y-self.y[0])/(self.y[-1]-self.y[0])
        design = self._design_features(parameters, mode_ids, f_lf)
        features = self._point_features(design, np.column_stack((xu, yu)), lf_values)
        envelope = xu*(1-xu)*yu*(1-yu)/self.envelope_max
        values = []
        with torch.no_grad():
            for start in range(0, len(features), 8192):
                values.append(self.field_network(torch.from_numpy(features[start:start+8192])).numpy()[:, 0])
        return np.concatenate(values)*envelope*self.field_scale[mode_ids-1]

    def predict_rows(self, dataset, rows):
        import torch
        rows = np.asarray(rows, dtype=int)
        xx, yy = np.meshgrid(dataset.x, dataset.y, indexing='xy')
        prediction = np.zeros((len(rows), *dataset.grid_shape))
        for start in range(0, len(rows), 8):
            local = rows[start:start+8]
            count = xx.size
            values = self.predict_points(np.repeat(dataset.parameters[local], count, axis=0),
                np.repeat(dataset.mode_ids[local], count), np.tile(xx.ravel(), len(local)),
                np.tile(yy.ravel(), len(local)), dataset.lf[local].ravel(), np.repeat(dataset.f_lf[local], count))
            prediction[start:start+len(local)] = values.reshape(len(local), *dataset.grid_shape)
        design = self._design_features(dataset.parameters[rows], dataset.mode_ids[rows], dataset.f_lf[rows])
        with torch.no_grad():
            ratio = self.frequency_network(torch.from_numpy(design)).numpy()[:, 0]
        ratio = ratio*self.frequency_scale[dataset.mode_ids[rows]-1]+self.frequency_mean[dataset.mode_ids[rows]-1]
        frequency = dataset.f_lf[rows]*np.exp(ratio)
        if not np.isfinite(frequency).all() or not np.isfinite(prediction).all():
            raise ValueError('Nonfinite neural prediction')
        return {'correction': prediction, 'correction_std': self.field_residual_std[dataset.mode_ids[rows]-1],
                'frequency': frequency, 'frequency_std': frequency*self.frequency_residual_scale[dataset.mode_ids[rows]-1]}

    def save(self, directory, dataset):
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({'field': self.field_network.state_dict(), 'frequency': self.frequency_network.state_dict()}, directory/'weights.pt')
        np.savez_compressed(directory/'model.npz', x=self.x, y=self.y, interior_mask=self.interior_mask,
            boundary_mask=self.boundary_mask, field_scale=self.field_scale, frequency_mean=self.frequency_mean,
            frequency_scale=self.frequency_scale, field_residual_std=self.field_residual_std,
            frequency_residual_scale=self.frequency_residual_scale)
        write_json(directory/'model.json', {'model_type': 'neural_inr', 'scaler': self.scaler.state(),
            'parameter_names': self.parameter_names, 'seed': self.seed, 'learning_rate': self.learning_rate,
            'steps': self.steps, 'width': self.width, 'n_modes': self.n_modes, 'envelope_max': self.envelope_max,
            'log_f_mean': self.log_f_mean, 'log_f_scale': self.log_f_scale,
            'training_runs': self.training_runs.tolist(), 'training_rows': self.training_rows.tolist(),
            'architecture': 'Three hidden tanh layers; deterministic sinusoidal coordinates; geometry/mode/LF-conditioned MLP',
            'uncertainty': 'Training residual scale proxy, not epistemic posterior'})

    @classmethod
    def load(cls, directory):
        import torch
        directory = Path(directory)
        meta = json.loads((directory/'model.json').read_text())
        model = cls(meta['learning_rate'], meta['steps'], meta['width'], meta['seed'])
        model.scaler = ParameterScaler.from_state(meta['scaler'])
        model.parameter_names = tuple(meta['parameter_names'])
        for name in ('n_modes', 'envelope_max', 'log_f_mean', 'log_f_scale'):
            setattr(model, name, meta[name])
        model.training_runs, model.training_rows = np.asarray(meta['training_runs']), np.asarray(meta['training_rows'])
        with np.load(directory/'model.npz', allow_pickle=False) as archive:
            for name in archive.files:
                setattr(model, name, archive[name])
        design_features = len(model.parameter_names)+model.n_modes+1
        model.field_network, model.frequency_network = neural_networks(design_features+27, design_features, model.width)
        weights = torch.load(directory/'weights.pt', weights_only=True, map_location='cpu')
        model.field_network.load_state_dict(weights['field'])
        model.frequency_network.load_state_dict(weights['frequency'])
        model.field_network.eval()
        model.frequency_network.eval()
        return model


def make_model(config):
    common = {'seed': config['seed'], 'max_iterations': config.get('max_iterations', 80)}
    family = config['family']
    if family == 'current_pca':
        return LatentGPModel(latent_dim=config.get('latent_dim', 48), **common)
    if family == 'output_pca':
        return ImprovedLatentGP(kernel=config['kernel'], latent_dim=config.get('latent_dim', 48), **common)
    if family == 'direct_hf':
        return DirectHFModel(kernel=config['kernel'], **common)
    if family == 'cokriging':
        return CoKrigingModel(kernel=config['kernel'], **common)
    if family == 'neural_inr':
        return NeuralINR(config['learning_rate'], config.get('steps', 1500), config.get('width', 64), config['seed'])
    raise ValueError(f'Unknown model family: {family}')


def save_model(model, directory, dataset, config):
    model.save(directory, dataset)
    write_json(Path(directory)/'configuration.json', {**config, 'implementation_sha256': file_sha256(__file__)})


def load_model(directory):
    directory = Path(directory)
    family = json.loads((directory/'configuration.json').read_text())['family']
    classes = {'current_pca': LatentGPModel, 'output_pca': ImprovedLatentGP,
               'direct_hf': DirectHFModel, 'cokriging': CoKrigingModel, 'neural_inr': NeuralINR}
    return classes[family].load(directory)
