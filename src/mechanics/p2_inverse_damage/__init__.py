"""Inverse damage identification via conditional VAE posterior inference."""
from .damage_data import DamageSampler, generate_damage_dataset
from .encoder import MeasurementEncoder
from .decoder import DamageDecoder
from .posterior import ConditionalPosterior
from .train import train_autoencoder, train_posterior

__all__ = [
    "DamageSampler",
    "generate_damage_dataset",
    "MeasurementEncoder",
    "DamageDecoder",
    "ConditionalPosterior",
    "train_autoencoder",
    "train_posterior",
]
