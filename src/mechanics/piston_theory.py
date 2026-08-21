"""Supersonic piston theory for aerodynamic pressure."""
from __future__ import annotations
import warnings
import numpy as np

# Physical constants
AIR_DENSITY = 1.2  # kg/m^3
SOUND_SPEED = 340.0  # m/s

# Piston theory practical validity range
DEFAULT_MIN_MACH = 2.0  # below this, piston theory is unreliable


def validate_mach(M_inf: float, *, strict_high_mach: bool = True, min_mach: float = DEFAULT_MIN_MACH):
    """Validate Mach number for piston theory.

    Separates mathematical validity (M>1) from modeling validity (M>=min_mach).
    """
    if M_inf <= 1.0 + 1e-6:
        raise ValueError(f"Piston theory is singular at M=1, got M={M_inf:.4f}.")
    if strict_high_mach and M_inf < min_mach:
        raise ValueError(
            f"Mach {M_inf:.3f} is outside the configured "
            f"high-Mach piston-theory domain M >= {min_mach}."
        )
    if M_inf < min_mach:
        warnings.warn(
            f"Piston-theory result used outside recommended range M >= {min_mach} "
            f"(got M={M_inf:.3f}).",
            RuntimeWarning,
            stacklevel=2,
        )


def piston_pressure(
    velocity: float,
    flow_angle: float,
    dw_dx: float,
    dw_dy: float,
    dw_dt: float,
    rho_inf: float = AIR_DENSITY,
    sound_speed: float = SOUND_SPEED,
    *,
    minimum_mach: float = 2.0,
    pressure_sign: float = -1.0,
) -> float:
    """First-order supersonic piston theory pressure.

    Args:
        velocity: Free stream velocity (m/s)
        flow_angle: Flow angle (radians)
        dw_dx: Partial derivative of w with respect to x
        dw_dy: Partial derivative of w with respect to y
        dw_dt: Partial derivative of w with respect to t
        rho_inf: Air density (kg/m^3)
        sound_speed: Speed of sound (m/s)
        minimum_mach: Minimum Mach number for domain guard
        pressure_sign: Sign convention for pressure (+1 or -1)

    Returns:
        Aerodynamic pressure Delta_p (Pa)

    Raises:
        ValueError: If inputs are non-positive or M < minimum_mach
    """
    if velocity <= 0.0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if sound_speed <= 0.0:
        raise ValueError(f"sound_speed must be positive, got {sound_speed}")
    if rho_inf <= 0.0:
        raise ValueError(f"rho_inf must be positive, got {rho_inf}")

    M_inf = velocity / sound_speed
    if M_inf < minimum_mach:
        raise ValueError(
            f"First-order piston model configured for M >= {minimum_mach}; received M={M_inf:.6f}"
        )
    cos_a = np.cos(flow_angle)
    sin_a = np.sin(flow_angle)

    # Eq (13) from paper
    coeff = rho_inf * velocity**2 / (np.sqrt(M_inf**2 - 1))
    Delta_p = pressure_sign * coeff * (
        cos_a * dw_dx
        + sin_a * dw_dy
        + (M_inf**2 - 2) / (M_inf**2 - 1) * dw_dt / velocity
    )

    return Delta_p


def non_dimensional_lambda(
    velocity: float,
    rho_inf: float,
    L1: float,
    D11: float,
    sound_speed: float = SOUND_SPEED,
) -> float:
    """Non-dimensional aerodynamic pressure parameter lambda.

    Args:
        velocity: Free stream velocity (m/s)
        rho_inf: Air density (kg/m^3)
        L1: Plate length (m)
        D11: Bending stiffness (Pa*m^3)
        sound_speed: Speed of sound (m/s)

    Returns:
        Non-dimensional lambda

    Raises:
        ValueError: If velocity <= sound_speed (subsonic flow)
    """
    M_inf = velocity / sound_speed
    if M_inf <= 1.0:
        raise ValueError(f"Supersonic flow required (M>1), got M={M_inf:.4f}")
    # Eq (14) from paper (matches plate-main/plate/plate.py:760)
    return (rho_inf * velocity**2 * L1**3) / (D11 * np.sqrt(M_inf**2 - 1))


def velocity_from_lambda(
    lambda_: float,
    rho_inf: float,
    L1: float,
    D11: float,
    sound_speed: float = SOUND_SPEED,
) -> float:
    """Convert non-dimensional lambda back to velocity (m/s).

    Solves the quadratic in V^2 analytically.
    Returns the larger root (matches plate-main/plate/plate.py:763-779).
    """
    # lambda = rho * V^2 * L^3 / (D * sqrt((V/c)^2 - 1))
    # Squared: lambda^2 * D^2 * (V^2/c^2 - 1) = rho^2 * V^4 * L^6
    # Let x = V^2: a2*x^2 + a1*x + a0 = 0
    a2 = (rho_inf * L1**3)**2
    a1 = -(lambda_ * D11 / sound_speed)**2
    a0 = (lambda_ * D11)**2
    discriminant = a1**2 - 4 * a2 * a0
    if discriminant < 0:
        raise ValueError(f"No real solution for lambda={lambda_}")
    velocity2 = (-a1 + np.sqrt(discriminant)) / (2 * a2)
    return np.sqrt(velocity2)
