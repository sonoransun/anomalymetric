"""Domain helpers for the gravitational-differential channel.

These convert headline physics parameters (Eotvos parameter eta, Yukawa range
lambda, scalar-DM mass) into the frequency / amplitude of the line they would
produce in a differential-acceleration PSD. Numbers are illustrative.
"""

from __future__ import annotations

from anomalymetric.units import C_LIGHT_M_S, H_PLANCK_EV_S

G_EARTH_M_S2 = 9.80665


def oscillating_dm_freq_hz(m_phi_eV: float) -> float:
    """Scalar/dilaton dark matter oscillates at its Compton frequency nu = m c^2/h."""
    return float(m_phi_eV) / H_PLANCK_EV_S


def yukawa_range_to_freq_hz(lambda_m: float) -> float:
    """Compton frequency of an ultralight Yukawa mediator of range `lambda_m`.

    The mediator mass is m = hbar / (lambda c), so nu = m c^2 / h = c / (2 pi lambda).
    """
    return C_LIGHT_M_S / (2.0 * 3.141592653589793 * float(lambda_m))


def ep_eta_to_amplitude(eta: float, g_local: float = G_EARTH_M_S2) -> float:
    """Differential acceleration from an Eotvos-parameter `eta`: a_diff = eta * g."""
    return float(eta) * g_local


def microscope_modulation_freq_hz(
    orbital_period_s: float = 5946.0, spin_freq_hz: float = 0.0
) -> float:
    """MICROSCOPE-style EP modulation frequency f_EP = f_orbit + f_spin.

    Defaults to the MICROSCOPE ~5946 s orbit (inertial-pointing mode, no spin).
    """
    return 1.0 / float(orbital_period_s) + float(spin_freq_hz)


def gw_response_stub(*args, **kwargs):  # pragma: no cover - optional dependency stub
    """Placeholder for a strain/gravimeter backend (the `[grav]` extra)."""
    raise NotImplementedError(
        "Real gravimeter / strain backends are not bundled; install the optional "
        "`[grav]` extra and wire a device driver here."
    )
