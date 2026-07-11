"""Multi-fidelity correction field via latent-space Gaussian Process."""
from .data import generate_lf_dataset, extract_correction_fields, boundary_envelope
from .encoder import CorrectionEncoder
from .decoder import CorrectionDecoder
from .gp_model import LatentGP
from .inr_baseline import CorrectionINR
