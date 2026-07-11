"""Configuration for inverse damage identification experiments."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DamageGenConfig:
    grid: tuple[int, int] = (16, 16)
    damage_min: float = 0.1
    damage_max: float = 1.0
    max_patches: int = 3
    patch_size_range: tuple[float, float] = (0.05, 0.25)  # fraction of plate
    seed: int = 42


@dataclass
class SolverConfig:
    L1: float = 0.3
    L2: float = 0.3
    M: int = 15
    N: int = 15
    basis: str = "legendre"
    grid: tuple[int, int] = (64, 64)
    n_modes: int = 10
    k_stiffness: float = 1e12


@dataclass
class EncoderConfig:
    n_modes: int = 10
    grid_size: tuple[int, int] = (64, 64)
    d_conditioning: int = 128
    mode: str = "mlp"  # "mlp" or "cnn"


@dataclass
class DecoderConfig:
    d_latent: int = 64
    d_conditioning: int = 128
    grid_size: tuple[int, int] = (16, 16)
    hidden_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])


@dataclass
class PosteriorConfig:
    d_latent: int = 64
    d_conditioning: int = 128
    n_coupling_layers: int = 8
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 200


@dataclass
class DataConfig:
    n_samples: int = 10000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    noise_std: float = 0.0  # measurement noise
    output_path: str = "./data/damage_dataset.zarr"


@dataclass
class ExperimentConfig:
    damage: DamageGenConfig = field(default_factory=DamageGenConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    data: DataConfig = field(default_factory=DataConfig)
