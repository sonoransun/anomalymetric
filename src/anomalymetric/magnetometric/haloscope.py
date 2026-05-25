"""Haloscope domain helpers: mass <-> frequency, canned scan bands, backend stub.

A virialized dark-matter axion (or dark photon) of rest mass m appears as a
near-monochromatic signal at nu = m c^2 / h. On the package's shared log-energy
axis the bin energy *is* h*nu, so the line sits at E_center = m (in eV) directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anomalymetric.units import H_PLANCK_EV_S


def axion_mass_to_freq_hz(m_a_eV: float) -> float:
    """Compton frequency of an axion of mass `m_a_eV`: nu = m_a c^2 / h."""
    return float(m_a_eV) / H_PLANCK_EV_S


def freq_hz_to_axion_mass(nu_hz: float) -> float:
    """Inverse of `axion_mass_to_freq_hz`."""
    return float(nu_hz) * H_PLANCK_EV_S


def dark_photon_mass_to_freq_hz(m_Aprime_eV: float) -> float:
    """Dark-photon mass -> kinetic-mixing signal frequency (same relation)."""
    return float(m_Aprime_eV) / H_PLANCK_EV_S


@dataclass
class HaloscopeBand:
    """A scan band defined by its frequency edges (Hz)."""

    name: str
    f_lo_hz: float
    f_hi_hz: float

    def mass_grid_eV(self, n: int = 5) -> NDArray[np.float64]:
        """`n` log-spaced candidate masses (eV) across the band."""
        f = np.logspace(np.log10(self.f_lo_hz), np.log10(self.f_hi_hz), n)
        return f * H_PLANCK_EV_S


# Illustrative bands for the major broadband / resonant searches.
ABRACADABRA = HaloscopeBand("abracadabra", 1.0e3, 1.0e6)
ADMX = HaloscopeBand("admx", 5.0e8, 1.0e9)
SHAFT = HaloscopeBand("shaft", 1.0e4, 1.0e7)


def squid_backend(*args, **kwargs):  # pragma: no cover - optional dependency stub
    """Placeholder for a real SQUID-DAQ / cavity backend (the `[squid]` extra)."""
    raise NotImplementedError(
        "Real SQUID acquisition backends are not bundled; install the optional "
        "`[squid]` extra and wire a device driver here."
    )
