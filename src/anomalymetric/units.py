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
