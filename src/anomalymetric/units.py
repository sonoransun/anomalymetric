"""Energy-axis helpers for the unified log10(E/eV) coordinate.

Photons span ~1e-6 eV (radio) to ~1e15 eV (UHE gamma). Cosmic rays extend to
~1e21 eV (ZeV). Everything in this package is internally indexed in eV; user
code converts to/from natural instrument units (Hz, keV, GeV, ...) at the
boundary via the helpers below.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

H_PLANCK_EV_S = 4.135667696e-15  # eV * s
C_LIGHT_M_S = 2.99792458e8
EV_PER_JOULE = 6.241509074e18
EV_PER_KEV = 1e3
EV_PER_MEV = 1e6
EV_PER_GEV = 1e9
EV_PER_TEV = 1e12
EV_PER_PEV = 1e15
EV_PER_EEV = 1e18
EV_PER_ZEV = 1e21


def hz_to_ev(nu_hz: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(nu_hz, dtype=float) * H_PLANCK_EV_S


def ev_to_hz(e_ev: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(e_ev, dtype=float) / H_PLANCK_EV_S


def wavelength_m_to_ev(lam_m: ArrayLike) -> NDArray[np.float64]:
    lam = np.asarray(lam_m, dtype=float)
    return H_PLANCK_EV_S * C_LIGHT_M_S / lam


def wavelength_nm_to_ev(lam_nm: ArrayLike) -> NDArray[np.float64]:
    return wavelength_m_to_ev(np.asarray(lam_nm, dtype=float) * 1e-9)


def log_energy_grid(
    log_e_min: float, log_e_max: float, bins_per_decade: int = 20
) -> NDArray[np.float64]:
    """Return bin edges uniformly spaced in log10(E/eV)."""
    n_bins = int(np.ceil((log_e_max - log_e_min) * bins_per_decade))
    return np.linspace(log_e_min, log_e_max, n_bins + 1)


def bin_widths_eV(log_edges: ArrayLike) -> NDArray[np.float64]:
    """Linear-eV widths from log10(E/eV) edges."""
    edges = 10.0 ** np.asarray(log_edges, dtype=float)
    return np.diff(edges)


def bin_centers_eV(log_edges: ArrayLike) -> NDArray[np.float64]:
    """Geometric centers in eV from log10(E/eV) edges."""
    edges_log = np.asarray(log_edges, dtype=float)
    return 10.0 ** (0.5 * (edges_log[:-1] + edges_log[1:]))


def log_frequency_grid(
    log_f_min: float, log_f_max: float, bins_per_decade: int = 20
) -> NDArray[np.float64]:
    """Bin edges uniform in log10(nu/Hz), returned as canonical log10(E/eV) edges.

    Sensor channels (magnetometer/SQUID, gravimeter) are naturally binned in
    frequency; this maps them onto the package's shared log-energy axis via
    E = h*nu. `hz_to_ev` is monotonic, so the returned edges stay sorted
    ascending and behave identically to `log_energy_grid` downstream.
    """
    n_bins = int(np.ceil((log_f_max - log_f_min) * bins_per_decade))
    log_nu_edges = np.linspace(log_f_min, log_f_max, n_bins + 1)
    return np.log10(hz_to_ev(10.0**log_nu_edges))


def bin_bandwidth_hz(log_edges: ArrayLike) -> NDArray[np.float64]:
    """Per-bin resolution bandwidth in Hz from log10(E/eV) edges.

    Used to turn a white-noise amplitude spectral density into a per-bin sigma
    (sigma ~ ASD * sqrt(bandwidth)).
    """
    e_edges = 10.0 ** np.asarray(log_edges, dtype=float)
    return np.abs(np.diff(ev_to_hz(e_edges)))


def asd_to_psd(asd: ArrayLike) -> NDArray[np.float64]:
    """Amplitude spectral density -> power spectral density (square)."""
    return np.asarray(asd, dtype=float) ** 2


def frequencies_hz(log_energy_centers_eV: ArrayLike) -> NDArray[np.float64]:
    """Frequencies (Hz) for log10(E/eV) centers on a sensor channel (nu = E / h)."""
    return ev_to_hz(10.0 ** np.asarray(log_energy_centers_eV, dtype=float))
