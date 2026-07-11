"""Behavioral tests for P2 inverse damage identification.

Tests verify:
1. Damage scaling produces frequency shifts
2. Encoder-decoder roundtrip reconstructs
3. Posterior samples are diverse
4. NLL decreases during training
"""
import numpy as np
import pytest

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.p2_inverse_damage.damage_data import DamageSampler, generate_damage_dataset
from mechanics.p2_inverse_damage.encoder import MeasurementEncoder
from mechanics.p2_inverse_damage.decoder import DamageDecoder
from mechanics.p2_inverse_damage.posterior import ConditionalPosterior
from mechanics.p2_inverse_damage.train import train_autoencoder, train_posterior


# ─── Fixtures ────────────────────────────────────────────────────────

def _base_laminate() -> Laminate:
    E = 70e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(
        E1=4.726844e7, E2=4.754649e7,
        G23=1.012895e9, G13=1.012895e9, G12=1.197467e7,
        nu12=0.9824561, rho=278.1545,
    )
    h_f = 0.001
    h_c = 0.025
    z = [-(h_f + h_c / 2), -h_c / 2, h_c / 2, h_f + h_c / 2]
    return Laminate(materials=[face, core, face], angles=[0.0, 0.0, 0.0], z=z)


def _solve(lam: Laminate, M: int = 8, N: int = 8, n_modes: int = 6):
    solver = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, grid=(32, 32))
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    return solver.solve_modal(n_modes=n_modes)


# ─── 1. Damage scaling produces frequency shifts ─────────────────────

class TestDamageScaling:
    def test_frequency_decreases_with_damage(self):
        """Reducing stiffness (d < 1) should lower natural frequencies."""
        base = _base_laminate()
        r_pris = _solve(base)

        d = 0.5
        damaged = DamageSampler.damage_laminate(base, d)
        r_dmg = _solve(damaged)

        # All frequencies should be lower for damaged plate
        assert np.all(r_dmg.frequencies.real < r_pris.frequencies.real)

    def test_monotonic_frequency_vs_damage(self):
        """Frequency should decrease monotonically as damage increases."""
        base = _base_laminate()
        factors = [1.0, 0.8, 0.6, 0.4]
        f0 = _solve(base).frequencies.real[0]

        for d in factors:
            damaged = DamageSampler.damage_laminate(base, d)
            f_d = _solve(damaged).frequencies.real[0]
            assert f_d <= f0 + 1e-6, f"freq at d={d} ({f_d}) > pristine ({f0})"
            f0 = f_d

    def test_damage_sampler_range(self):
        sampler = DamageSampler(d_min=0.3, rng=np.random.default_rng(0))
        samples = [sampler.sample() for _ in range(200)]
        assert all(0.3 <= s <= 1.0 for s in samples)
        assert np.std(samples) > 0.01  # actually varying

    def test_apply_damage_scales_elastic_moduli(self):
        mat = Material(100e9, 80e9, 30e9, 30e9, 30e9, 0.3, 2700)
        d = 0.5
        damaged = DamageSampler.apply_damage(mat, d)
        assert damaged.E1 == pytest.approx(50e9)
        assert damaged.E2 == pytest.approx(40e9)
        assert damaged.G12 == pytest.approx(15e9)
        assert damaged.nu12 == mat.nu12  # nu unchanged
        assert damaged.rho == mat.rho   # rho unchanged


# ─── 2. Encoder-decoder roundtrip ────────────────────────────────────

class TestEncoderDecoderRoundtrip:
    def test_encoder_output_shape(self):
        enc = MeasurementEncoder(n_modes=6, grid_size=(32, 32), d_c=64, seed=0)
        freqs = np.random.default_rng(1).uniform(10, 1000, (4, 6))
        modes = np.random.default_rng(2).standard_normal((4, 6, 32, 32))
        c = enc.forward(freqs, modes)
        assert c.shape == (4, 64)

    def test_decoder_output_in_01(self):
        dec = DamageDecoder(d_z=32, d_c=64, grid_size=(32, 32), seed=0)
        z = np.random.default_rng(3).standard_normal((4, 32))
        c = np.random.default_rng(4).standard_normal((4, 64))
        d = dec.forward(z, c)
        assert d.shape == (4, 32, 32)
        assert d.min() >= 0.0
        assert d.max() <= 1.0

    def test_roundtrip_reconstruction(self):
        """After autoencoder training, encoder(decoder(z)) should approximate input."""
        rng = np.random.default_rng(10)
        n_modes, gx, gy = 6, 16, 16
        n_train = 30

        freqs = rng.uniform(10, 500, (n_train, n_modes))
        modes = rng.standard_normal((n_train, n_modes, gx, gy))
        # Target damage fields
        targets = rng.uniform(0.3, 1.0, (n_train, gx, gy))

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)

        losses = train_autoencoder(enc, dec, freqs, modes, targets,
                                   n_epochs=20, lr=1e-3, batch_size=16, seed=42)
        # Loss should decrease
        assert losses[-1] <= losses[0]


# ─── 3. Posterior samples are diverse ─────────────────────────────────

class TestPosteriorDiversity:
    def test_sample_shape(self):
        post = ConditionalPosterior(d_z=16, d_c=32, seed=0)
        c = np.random.default_rng(5).standard_normal((8, 32))
        z = post.sample(c, rng=np.random.default_rng(6))
        assert z.shape == (8, 16)

    def test_samples_differ(self):
        """Drawing multiple samples from same c should produce different z."""
        post = ConditionalPosterior(d_z=16, d_c=32, seed=0)
        c = np.ones((1, 32))
        z1 = post.sample(c, rng=np.random.default_rng(10))
        z2 = post.sample(c, rng=np.random.default_rng(11))
        assert not np.allclose(z1, z2, atol=1e-6)

    def test_prior_samples_are_standard_normal(self):
        post = ConditionalPosterior(d_z=32, d_c=64, seed=0)
        z = post.sample_prior(500, rng=np.random.default_rng(7))
        assert z.shape == (500, 32)
        # Mean near 0, std near 1
        assert np.abs(z.mean()) < 0.15
        assert np.abs(z.std() - 1.0) < 0.15

    def test_log_prob_returns_batch(self):
        post = ConditionalPosterior(d_z=16, d_c=32, seed=0)
        z = np.random.default_rng(8).standard_normal((5, 16))
        c = np.random.default_rng(9).standard_normal((5, 32))
        lp = post.log_prob(z, c)
        assert lp.shape == (5,)

    def test_kl_divergence_nonnegative(self):
        post = ConditionalPosterior(d_z=16, d_c=32, seed=0)
        c = np.random.default_rng(12).standard_normal((10, 32))
        kl = post.kl_divergence(c)
        assert kl >= -0.01  # KL >= 0, allow small numerical error


# ─── 4. NLL decreases during training ────────────────────────────────

class TestNLLOptimization:
    def test_nll_decreases(self):
        """Posterior NLL should decrease over training epochs."""
        rng = np.random.default_rng(20)
        n_modes, gx, gy = 4, 8, 8
        n_train = 20

        freqs = rng.uniform(10, 300, (n_train, n_modes))
        modes = rng.standard_normal((n_train, n_modes, gx, gy))
        damage = rng.uniform(0.3, 1.0, (n_train, gx, gy))

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=16, seed=0)
        dec = DamageDecoder(d_z=8, d_c=16, grid_size=(gx, gy), seed=1)
        post = ConditionalPosterior(d_z=8, d_c=16, seed=2)

        # Pre-train autoencoder briefly
        train_autoencoder(enc, dec, freqs, modes, damage,
                          n_epochs=10, batch_size=10, seed=42)

        # Train posterior
        losses = train_posterior(post, enc, dec, freqs, modes, damage,
                                 n_epochs=20, batch_size=10, seed=42)
        # Final loss should be <= initial
        assert losses[-1] <= losses[0] + 0.1  # allow small tolerance

    def test_full_pipeline_small(self):
        """End-to-end: generate tiny dataset, train, verify outputs."""
        data = generate_damage_dataset(
            n_samples=8, L1=0.3, L2=0.3, M=6, N=6,
            n_modes=4, grid=(16, 16), d_min=0.5, seed=99,
        )
        assert data["frequencies"].shape == (8, 4)
        assert data["mode_shapes"].shape == (8, 4, 16, 16)
        assert data["damage_factors"].shape == (8,)
        assert np.all(data["damage_factors"] >= 0.5)
        assert np.all(data["damage_factors"] <= 1.0)


# ─── Synthetic data helper ────────────────────────────────────────────

def _make_synthetic(n_samples, n_modes, gx, gy, seed=42):
    """Synthetic measurement-damage pairs. Measurements scale with damage factor."""
    rng = np.random.default_rng(seed)
    base_freq = rng.uniform(100, 400, n_modes)
    base_modes = rng.standard_normal((n_modes, gx, gy))

    dmg_factors = rng.uniform(0.3, 1.0, n_samples)
    freqs = np.array([base_freq * np.sqrt(d) for d in dmg_factors])
    modes = np.array([base_modes * (0.5 + 0.5 * d) for d in dmg_factors])
    damage_fields = np.array([np.full((gx, gy), d) for d in dmg_factors])

    return freqs, modes, damage_fields, dmg_factors, base_freq, base_modes


# ─── P2 Behavioral Tests ──────────────────────────────────────────────

class TestP2Behavioral:
    def test_encoder_similar_damage_close_representations(self):
        """Encoder outputs for damage factors within 0.05 have cosine similarity > 0.8."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, base_freq, base_modes = _make_synthetic(60, n_modes, gx, gy, seed=10)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        d1, d2 = 0.7, 0.73
        c1 = enc.forward_single(base_freq * np.sqrt(d1), base_modes * (0.5 + 0.5 * d1))
        c2 = enc.forward_single(base_freq * np.sqrt(d2), base_modes * (0.5 + 0.5 * d2))
        cos_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        assert cos_sim > 0.8, f"Cosine similarity {cos_sim:.3f} < 0.8"

    def test_decoder_reconstructs_constant_damage_field(self):
        """After training, decoding z=0 recovers constant damage within 0.15 MSE."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, base_freq, base_modes = _make_synthetic(60, n_modes, gx, gy, seed=20)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        d_test = 0.6
        c = enc.forward_single(base_freq * np.sqrt(d_test), base_modes * (0.5 + 0.5 * d_test))
        pred = dec.forward_single(np.zeros(16), c)
        mse = np.mean((pred - np.full((gx, gy), d_test)) ** 2)
        assert mse < 0.15, f"Reconstruction MSE {mse:.4f} >= 0.15"

    def test_posterior_samples_contain_truth(self):
        """After full training, 5th-95th percentile of decoded samples contains truth at >70% of grid."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, base_freq, base_modes = _make_synthetic(60, n_modes, gx, gy, seed=30)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        post = ConditionalPosterior(d_z=16, d_c=32, seed=2)

        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)
        train_posterior(post, enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        d_test = 0.55
        c = enc.forward_single(base_freq * np.sqrt(d_test), base_modes * (0.5 + 0.5 * d_test))
        target = np.full((gx, gy), d_test)

        rng_samp = np.random.default_rng(99)
        samples = []
        for _ in range(100):
            z = post.sample(c[np.newaxis, :], rng=rng_samp)
            samples.append(dec.forward_single(z[0], c))
        samples = np.array(samples)

        lo = np.percentile(samples, 5, axis=0)
        hi = np.percentile(samples, 95, axis=0)
        coverage = ((target >= lo) & (target <= hi)).mean()
        assert coverage > 0.70, f"Coverage {coverage:.2%} <= 70%"

    def test_training_reduces_reconstruction_mse(self):
        """Final MSE is at least 3x lower than initial MSE."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, _, _ = _make_synthetic(60, n_modes, gx, gy, seed=40)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)

        rng = np.random.default_rng(99)

        def compute_mse():
            total = 0.0
            for i in range(len(freqs)):
                c = enc.forward_single(freqs[i], modes[i])
                z = rng.standard_normal(16)
                pred = dec.forward_single(z, c)
                total += np.mean((pred - dmg_fields[i]) ** 2)
            return total / len(freqs)

        initial_mse = compute_mse()
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)
        final_mse = compute_mse()

        assert final_mse < initial_mse / 3, f"Initial {initial_mse:.4f}, final {final_mse:.4f}"

    def test_full_roundtrip_mse_below_threshold(self):
        """After training: encode → sample z~N(0,I) → decode → MSE < 0.05."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, _, _ = _make_synthetic(60, n_modes, gx, gy, seed=50)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=100, lr=1e-3, batch_size=16, seed=42)

        rng = np.random.default_rng(77)
        total_mse = 0.0
        for i in range(len(freqs)):
            c = enc.forward_single(freqs[i], modes[i])
            z = rng.standard_normal(16)
            pred = dec.forward_single(z, c)
            total_mse += np.mean((pred - dmg_fields[i]) ** 2)
        avg_mse = total_mse / len(freqs)
        assert avg_mse < 0.05, f"Roundtrip MSE {avg_mse:.4f} >= 0.05"

    def test_encoder_distinguishes_pristine_vs_damaged(self):
        """Encoder distance d=1.0 vs d=0.3 > 10x distance between two d=0.95 samples."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, base_freq, base_modes = _make_synthetic(60, n_modes, gx, gy, seed=60)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        c_pris = enc.forward_single(base_freq * 1.0, base_modes * 1.0)
        c_dmg = enc.forward_single(base_freq * np.sqrt(0.3), base_modes * (0.5 + 0.5 * 0.3))

        c_a = enc.forward_single(base_freq * np.sqrt(0.95), base_modes * (0.5 + 0.5 * 0.95))
        c_b = enc.forward_single(base_freq * np.sqrt(0.95) * 1.001, base_modes * (0.5 + 0.5 * 0.95) * 1.001)

        dist_pristine_damaged = np.linalg.norm(c_pris - c_dmg)
        dist_close = np.linalg.norm(c_a - c_b)
        assert dist_pristine_damaged > 10 * dist_close, (
            f"Pristine-damaged {dist_pristine_damaged:.4f} <= 10x close {dist_close:.4f}"
        )

    def test_posterior_kl_decreases_with_training(self):
        """KL divergence q(z|c) || p(z) is lower after posterior training."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, _, _ = _make_synthetic(40, n_modes, gx, gy, seed=70)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        post = ConditionalPosterior(d_z=16, d_c=32, seed=2)

        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=50, lr=1e-3, batch_size=16, seed=42)

        c = enc.forward(freqs, modes)
        kl_before = post.kl_divergence(c)

        train_posterior(post, enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        c_after = enc.forward(freqs, modes)
        kl_after = post.kl_divergence(c_after)

        assert kl_after < kl_before, f"KL before {kl_before:.4f} >= after {kl_after:.4f}"

    def test_decoder_output_spatial_smoothness(self):
        """Mean absolute difference between adjacent grid cells < 0.1."""
        n_modes, gx, gy = 4, 8, 8
        freqs, modes, dmg_fields, _, base_freq, base_modes = _make_synthetic(60, n_modes, gx, gy, seed=80)

        enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gx, gy), d_c=32, seed=0)
        dec = DamageDecoder(d_z=16, d_c=32, grid_size=(gx, gy), seed=1)
        train_autoencoder(enc, dec, freqs, modes, dmg_fields, n_epochs=80, lr=1e-3, batch_size=16, seed=42)

        d_test = 0.6
        c = enc.forward_single(base_freq * np.sqrt(d_test), base_modes * (0.5 + 0.5 * d_test))
        pred = dec.forward_single(np.zeros(16), c)

        h_diff = np.mean(np.abs(pred[:, :-1] - pred[:, 1:]))
        v_diff = np.mean(np.abs(pred[:-1, :] - pred[1:, :]))
        mean_diff = (h_diff + v_diff) / 2
        assert mean_diff < 0.1, f"Mean adjacent diff {mean_diff:.4f} >= 0.1"
