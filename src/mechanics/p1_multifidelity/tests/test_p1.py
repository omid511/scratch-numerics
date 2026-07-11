"""Tests for Proposal 1: Multi-Fidelity Correction Field."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from mechanics.p1_multifidelity.data import boundary_envelope, extract_correction_fields, create_synthetic_hf
from mechanics.p1_multifidelity.encoder import CorrectionEncoder
from mechanics.p1_multifidelity.decoder import CorrectionDecoder
from mechanics.p1_multifidelity.gp_model import LatentGP, RBFKernel
from mechanics.p1_multifidelity.inr_baseline import CorrectionINR
from mechanics.p1_multifidelity.train import train_autoencoder, train_gp


class TestBoundaryEnvelope:
    def test_vanishes_at_edges(self):
        """b(x,y) must be 0 at all domain boundaries."""
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        X, Y = np.meshgrid(x, y)
        b = boundary_envelope(X, Y)
        # Edges
        assert np.allclose(b[0, :], 0.0, atol=1e-14)
        assert np.allclose(b[-1, :], 0.0, atol=1e-14)
        assert np.allclose(b[:, 0], 0.0, atol=1e-14)
        assert np.allclose(b[:, -1], 0.0, atol=1e-14)

    def test_positive_inside(self):
        """b(x,y) > 0 strictly inside domain."""
        x = np.linspace(0.1, 0.9, 10)
        y = np.linspace(0.1, 0.9, 10)
        X, Y = np.meshgrid(x, y)
        b = boundary_envelope(X, Y)
        assert np.all(b > 0.0)

    def test_peak_at_center(self):
        """Maximum near domain center."""
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y)
        b = boundary_envelope(X, Y)
        peak_idx = np.unravel_index(np.argmax(b), b.shape)
        assert abs(peak_idx[0] - 50) <= 1
        assert abs(peak_idx[1] - 50) <= 1


class TestEncoder:
    def test_encode_decode_shape(self):
        """Encode then decode preserves shape."""
        rng = np.random.default_rng(0)
        fields = rng.standard_normal((20, 64, 64))
        enc = CorrectionEncoder(d_z=8, grid_size=(64, 64))
        enc.fit(fields)
        z = enc.encode(fields)
        assert z.shape == (20, 8)
        recon = enc.decode(z)
        assert recon.shape == (20, 64, 64)

    def test_single_field(self):
        """Encode/decode a single field (needs 2+ training samples for PCA)."""
        rng = np.random.default_rng(1)
        # PCA needs n >= 2, so use 2 very similar fields
        field = rng.standard_normal((64, 64))
        fields = np.stack([field, field * 1.01])
        enc = CorrectionEncoder(d_z=1, grid_size=(64, 64))
        enc.fit(fields)
        z = enc.encode(field)
        assert z.shape == (1,)
        recon = enc.decode(z)
        assert recon.shape == (64, 64)

    def test_reconstruction_quality(self):
        """PCA should reconstruct training data well with enough components."""
        rng = np.random.default_rng(2)
        # Low-rank data
        base = rng.standard_normal((5, 64 * 64))
        fields = np.stack([base[i % 5] + rng.standard_normal(64 * 64) * 0.01 for i in range(30)])
        fields = fields.reshape(30, 64, 64)
        enc = CorrectionEncoder(d_z=5, grid_size=(64, 64))
        enc.fit(fields)
        z = enc.encode(fields)
        recon = enc.decode(z)
        mse = np.mean((recon - fields) ** 2)
        assert mse < 0.1  # reasonable for random data


class TestDecoder:
    def test_boundary_enforcement(self):
        """Decoder output must vanish at boundaries."""
        rng = np.random.default_rng(3)
        fields = rng.standard_normal((10, 64, 64))
        enc = CorrectionEncoder(d_z=4, grid_size=(64, 64))
        enc.fit(fields)
        dec = CorrectionDecoder(enc)
        z = enc.encode(fields)
        recon = dec.decode(z, apply_boundary=True)
        # Edges should be zero
        assert np.allclose(recon[:, 0, :], 0.0, atol=1e-10)
        assert np.allclose(recon[:, -1, :], 0.0, atol=1e-10)
        assert np.allclose(recon[:, :, 0], 0.0, atol=1e-10)
        assert np.allclose(recon[:, :, -1], 0.0, atol=1e-10)

    def test_no_boundary_option(self):
        """Can decode without boundary enforcement."""
        rng = np.random.default_rng(4)
        fields = rng.standard_normal((10, 64, 64))
        enc = CorrectionEncoder(d_z=4, grid_size=(64, 64))
        enc.fit(fields)
        dec = CorrectionDecoder(enc)
        z = enc.encode(fields)
        recon = dec.decode(z, apply_boundary=False)
        # Without boundary, edges not necessarily zero
        assert recon.shape == (10, 64, 64)


class TestGP:
    def test_fit_predict_shape(self):
        """GP fits and predicts with correct shapes."""
        rng = np.random.default_rng(5)
        theta = rng.uniform(0, 1, (20, 3))
        z = rng.standard_normal((20, 4))
        gp = LatentGP(d_z=4)
        gp.fit(theta, z)
        mean, var = gp.predict(theta[:5])
        assert mean.shape == (5, 4)
        assert var.shape == (5, 4)
        assert np.all(var >= 0)

    def test_uncertainty_increases_away(self):
        """Variance should increase away from training data."""
        rng = np.random.default_rng(6)
        theta_train = rng.uniform(0.3, 0.7, (30, 2))
        z_train = rng.standard_normal((30, 2))
        gp = LatentGP(d_z=2, kernel=RBFKernel(length_scale=0.3))
        gp.fit(theta_train, z_train)

        # Predict at training points (low variance)
        mean_in, var_in = gp.predict(theta_train[:5])
        # Predict far from training (high variance)
        theta_out = rng.uniform(2.0, 3.0, (5, 2))
        mean_out, var_out = gp.predict(theta_out)
        # Average variance outside should be larger
        assert var_out.mean() > var_in.mean()

    def test_sample_shape(self):
        """Sampling produces correct shapes."""
        rng = np.random.default_rng(7)
        theta = rng.uniform(0, 1, (15, 3))
        z = rng.standard_normal((15, 4))
        gp = LatentGP(d_z=4)
        gp.fit(theta, z)
        samples = gp.sample(theta[:3], n_samples=10)
        assert samples.shape == (10, 3, 4)


class TestINR:
    def test_predict_grid_shape(self):
        """INR produces correct output shape."""
        rng = np.random.default_rng(8)
        theta = rng.uniform(0, 1, (3,))
        inr = CorrectionINR(d_theta=3, grid_size=(32, 32))
        delta = inr.predict_grid(theta)
        assert delta.shape == (32, 32)

    def test_boundary_enforcement(self):
        """INR output vanishes at boundaries."""
        inr = CorrectionINR(d_theta=2, grid_size=(64, 64))
        theta = np.array([0.5, 0.5])
        delta = inr.predict_grid(theta)
        assert np.allclose(delta[0, :], 0.0, atol=1e-10)
        assert np.allclose(delta[-1, :], 0.0, atol=1e-10)
        assert np.allclose(delta[:, 0], 0.0, atol=1e-10)
        assert np.allclose(delta[:, -1], 0.0, atol=1e-10)

    def test_predict_batch(self):
        """Batch prediction works."""
        inr = CorrectionINR(d_theta=2, grid_size=(32, 32))
        thetas = np.random.default_rng(9).uniform(0, 1, (5, 2))
        deltas = inr.predict_batch(thetas)
        assert deltas.shape == (5, 32, 32)


class TestTrain:
    def test_train_autoencoder(self):
        """Autoencoder training returns valid results."""
        rng = np.random.default_rng(10)
        fields = rng.standard_normal((20, 32, 32))
        enc, dec, metrics = train_autoencoder(fields, d_z=4, grid_size=(32, 32))
        assert "reconstruction_mse" in metrics
        assert "boundary_violation" in metrics
        assert metrics["d_z"] == 4

    def test_train_gp(self):
        """GP training returns fitted model."""
        rng = np.random.default_rng(11)
        theta = rng.uniform(0, 1, (15, 3))
        z = rng.standard_normal((15, 4))
        gp = train_gp(theta, z, d_z=4)
        mean, var = gp.predict(theta[:3])
        assert mean.shape == (3, 4)


class TestEncoderBehavioral:
    def test_encoder_reconstructs_signal_better_than_noise(self):
        """PCA encoder reconstructs structured fields better than random noise."""
        rng = np.random.default_rng(100)
        rank = 5
        base = rng.standard_normal((rank, 64, 64))
        n = 30
        idx = rng.integers(0, rank, size=n)
        fields = base[idx] + rng.standard_normal((n, 64, 64)) * 0.01
        enc = CorrectionEncoder(d_z=rank, grid_size=(64, 64))
        enc.fit(fields)
        recon = enc.decode(enc.encode(fields))
        mse_signal = np.mean((recon - fields) ** 2)
        noise = rng.standard_normal((n, 64, 64))
        recon_noise = enc.decode(enc.encode(noise))
        mse_noise = np.mean((recon_noise - noise) ** 2)
        assert mse_noise / mse_signal > 10, f"noise/signal MSE ratio {mse_noise/mse_signal:.1f} < 10"

    def test_encoder_reconstruction_mse_near_zero_on_training_data(self):
        """PCA with d_z >= rank reconstructs training data exactly (up to numerical precision)."""
        rng = np.random.default_rng(101)
        rank = 3
        base = rng.standard_normal((rank, 32, 32))
        n = 20
        idx = rng.integers(0, rank, size=n)
        fields = base[idx]
        enc = CorrectionEncoder(d_z=rank, grid_size=(32, 32))
        enc.fit(fields)
        recon = enc.decode(enc.encode(fields))
        mse = np.mean((recon - fields) ** 2)
        assert mse < 1e-10, f"MSE {mse} >= 1e-10"


class TestGPBehavioral:
    def test_gp_interpolates_smooth_function(self):
        """GP fits a simple linear function and interpolates accurately."""
        rng = np.random.default_rng(102)
        n_train = 40
        n_test = 20
        theta_train = rng.uniform(0, 1, (n_train, 2))
        z_train = (theta_train[:, 0:1] + theta_train[:, 1:2])  # f = θ0 + θ1, shape (n,1)
        gp = LatentGP(d_z=1, kernel=RBFKernel(length_scale=0.5), noise=1e-6)
        gp.fit(theta_train, z_train)
        theta_test = rng.uniform(0, 1, (n_test, 2))
        z_test = theta_test[:, 0:1] + theta_test[:, 1:2]
        z_pred, _ = gp.predict(theta_test)
        mae = np.mean(np.abs(z_pred[:, 0] - z_test.ravel()))
        assert mae < 0.05, f"MAE {mae} >= 0.05"

    def test_gp_leave_one_out_accuracy(self):
        """GP LOO prediction has correct uncertainty calibration."""
        rng = np.random.default_rng(103)
        n = 40
        theta = rng.uniform(0, 1, (n, 2))
        z = np.column_stack([theta[:, 0] * 2, theta[:, 1] * 3])
        coverage_count = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            gp_loo = LatentGP(d_z=2, kernel=RBFKernel(length_scale=0.5), noise=0.1)
            gp_loo.fit(theta[mask], z[mask])
            mean, var = gp_loo.predict(theta[i : i + 1])
            std = np.sqrt(var[0])
            if np.all(np.abs(mean[0] - z[i]) < 3 * std):
                coverage_count += 1
        frac = coverage_count / n
        assert frac > 0.80, f"LOO coverage {frac:.2f} < 0.80"


class TestINRBehavioral:
    def test_inr_loss_decreases_during_training(self):
        """INR loss drops substantially over 200 training steps."""
        rng = np.random.default_rng(104)
        n = 20
        theta = rng.uniform(0, 1, (n, 2))
        gx, gy = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        fields = np.stack([
            10.0 * (theta[i, 0] + theta[i, 1]) * gx * gy * (1 - gx) * (1 - gy)
            for i in range(n)
        ])
        inr = CorrectionINR(d_theta=2, grid_size=(32, 32), seed=42)
        # initial loss: forward pass before any weight update
        inr._mlp.eval()
        n_pts = 32 * 32
        pos_enc = inr._pos_enc_t
        theta_t = torch.as_tensor(theta[:1], dtype=torch.float32)
        theta_tiled = theta_t[0].unsqueeze(0).expand(n_pts, -1)
        full_inp = torch.cat([pos_enc, theta_tiled], dim=-1)
        tgt = torch.as_tensor(fields[0].ravel(), dtype=torch.float32)
        with torch.no_grad():
            out = inr._mlp(full_inp)
            initial_loss = float(nn.functional.mse_loss(out, tgt))
        # train 200 steps with aggressive lr
        lr = 5e-3
        rng2 = np.random.default_rng(42)
        for _ in range(200):
            idx = rng2.choice(n, size=min(8, n), replace=False)
            inr._train_step(theta[idx], fields[idx], lr=lr)
        final_loss = inr._train_step(theta[:1], fields[:1], lr=lr)
        assert final_loss < 0.5 * initial_loss, f"final {final_loss} not < 0.5 * initial {initial_loss}"

    def test_inr_different_theta_different_output(self):
        """Different θ produce different correction fields."""
        rng = np.random.default_rng(105)
        n = 30
        theta = rng.uniform(0, 1, (n, 2))
        gx, gy = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        fields = np.stack([theta[i, 0] * gx + theta[i, 1] * gy for i in range(n)])
        inr = CorrectionINR(d_theta=2, grid_size=(32, 32), seed=42)
        for _ in range(50):
            idx = rng.choice(n, size=min(8, n), replace=False)
            inr._train_step(theta[idx], fields[idx], lr=1e-3)
        da = inr.predict_grid(np.array([0.2, 0.3]))
        db = inr.predict_grid(np.array([0.8, 0.7]))
        diff = np.linalg.norm(da - db)
        assert diff > 1e-3, f"L2 diff {diff} <= 1e-3"

    def test_inr_generalizes_to_unseen_theta(self):
        """INR trained on [0.1,0.4] produces non-trivial output at θ=0.7 with zero boundaries."""
        rng = np.random.default_rng(106)
        n = 40
        theta = rng.uniform(0.1, 0.4, (n, 2))
        gx, gy = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        fields = np.stack([
            theta[i, 0] * np.sin(np.pi * gx) * np.sin(np.pi * gy)
            for i in range(n)
        ])
        inr = CorrectionINR(d_theta=2, grid_size=(32, 32), seed=42)
        for _ in range(200):
            idx = rng.choice(n, size=min(8, n), replace=False)
            inr._train_step(theta[idx], fields[idx], lr=1e-3)
        pred = inr.predict_grid(np.array([0.7, 0.7]))
        assert np.linalg.norm(pred) > 1e-6, "Prediction is trivially zero"
        assert np.allclose(pred[0, :], 0.0, atol=1e-10), "Top edge not zero"
        assert np.allclose(pred[-1, :], 0.0, atol=1e-10), "Bottom edge not zero"
        assert np.allclose(pred[:, 0], 0.0, atol=1e-10), "Left edge not zero"
        assert np.allclose(pred[:, -1], 0.0, atol=1e-10), "Right edge not zero"


class TestPipelineBehavioral:
    def test_pipeline_corrected_lf_closer_to_hf(self):
        """Applying predicted correction brings LF closer to HF."""
        rng = np.random.default_rng(107)
        n = 20
        grid_nx, grid_ny = 32, 32
        lf_fields = rng.standard_normal((n, grid_ny, grid_nx))
        gx, gy = np.meshgrid(np.linspace(0, 1, grid_nx), np.linspace(0, 1, grid_ny))
        correction = 0.04 * np.sin(np.pi * gx) * np.sin(np.pi * gy)
        hf_fields = lf_fields * (1 + correction)
        enc = CorrectionEncoder(d_z=4, grid_size=(grid_nx, grid_ny))
        corrections = hf_fields / lf_fields - 1
        enc.fit(corrections)
        dec = CorrectionDecoder(enc)
        z = enc.encode(corrections)
        theta = rng.uniform(0, 1, (n, 3))
        gp = LatentGP(d_z=z.shape[1], kernel=RBFKernel(length_scale=0.5), noise=1e-4)
        gp.fit(theta, z)
        z_pred, _ = gp.predict(theta)
        predicted_correction = dec.decode(z_pred, apply_boundary=True)
        corrected_lf = lf_fields * (1 + predicted_correction)
        err_before = np.linalg.norm(lf_fields - hf_fields)
        err_after = np.linalg.norm(corrected_lf - hf_fields)
        assert err_after < err_before, f"corrected {err_after:.4f} not < uncorrected {err_before:.4f}"


class TestEndToEnd:
    def test_synthetic_hf_correction_fields(self):
        """Synthetic HF produces correction fields with expected structure."""
        rng = np.random.default_rng(12)
        n = 5
        grid_nx, grid_ny = 16, 16
        lf_fields = rng.standard_normal((n, grid_ny, grid_nx))
        hf_fields = lf_fields * 1.04  # uniform 4% perturbation

        lf = {"mode_shapes": lf_fields[:, np.newaxis], "frequencies": np.ones((n, 1)), "params": rng.uniform(0, 1, (n, 3)), "grid_x": np.linspace(0, 1, grid_nx), "grid_y": np.linspace(0, 1, grid_ny)}
        hf = {"mode_shapes": hf_fields[:, np.newaxis], "frequencies": np.ones((n, 1)), "params": rng.uniform(0, 1, (n, 3)), "grid_x": np.linspace(0, 1, grid_nx), "grid_y": np.linspace(0, 1, grid_ny)}

        corrections = extract_correction_fields(lf, hf, mode_idx=0)
        assert corrections.shape == (n, grid_ny, grid_nx)
        # Should be approximately 0.04 everywhere
        assert np.allclose(corrections, 0.04, atol=0.01)

    def test_full_decode_pipeline(self):
        """Encode → GP predict → decode pipeline works end to end."""
        rng = np.random.default_rng(13)
        n = 30
        fields = rng.standard_normal((n, 32, 32))
        theta = rng.uniform(0, 1, (n, 3))

        # Train encoder
        enc = CorrectionEncoder(d_z=4, grid_size=(32, 32))
        enc.fit(fields)
        dec = CorrectionDecoder(enc)

        # Encode
        z = enc.encode(fields)

        # Train GP
        gp = train_gp(theta, z, d_z=4)

        # Predict at new points
        theta_new = rng.uniform(0, 1, (5, 3))
        z_mean, z_var = gp.predict(theta_new)

        # Decode
        recon = dec.decode(z_mean, apply_boundary=True)
        assert recon.shape == (5, 32, 32)
        # Boundaries enforced
        assert np.allclose(recon[:, 0, :], 0.0, atol=1e-10)
        assert np.allclose(recon[:, -1, :], 0.0, atol=1e-10)
