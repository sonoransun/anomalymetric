"""Cosmic-ray spectrum convenience constructors.

A CR spectrum is just a `Spectrum` with `kind=cr_*`. The factories here set
sensible defaults: `per_steradian=True`, exposure attached, and a log-energy
grid that spans the all-particle range (~1e9 to ~1e21 eV).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import log_energy_grid


def cr_allparticle_grid(
    log_e_min: float = 9.0, log_e_max: float = 21.0, bins_per_decade: int = 10
) -> NDArray[np.float64]:
    return log_energy_grid(log_e_min, log_e_max, bins_per_decade)


def make_cr_spectrum(
    log_energy_edges_eV: NDArray[np.float64],
    counts: NDArray[np.float64],
    exposure_cm2_s_sr: NDArray[np.float64],
    *,
    kind: SpectrumKind = SpectrumKind.CR_ALLPARTICLE,
) -> Spectrum:
    """Build a Spectrum for binned CR event counts with a per-bin exposure."""
    return Spectrum(
        log_energy_edges_eV=np.asarray(log_energy_edges_eV, dtype=float),
        value=np.asarray(counts, dtype=float),
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=kind,
        per_steradian=True,
        exposure_cm2_s=np.asarray(exposure_cm2_s_sr, dtype=float),
    )


def cr_reference_dnde(
    log_energy_centers_eV: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A textbook all-particle reference: broken power law tuned to PDG values.

    Indices and break energies follow the Particle Data Group cosmic-ray review
    (Workman+ 2022). The normalization is approximate and meant for tests, not
    publication-quality physics.
    """
    log_E = np.asarray(log_energy_centers_eV, dtype=float)
    E = 10.0**log_E
    # Below knee: index ~2.7. Between knee and ankle: ~3.1. Above ankle: ~2.7.
    a_lo, a_mid, a_hi = 2.7, 3.1, 2.7
    E_knee = 3.0e15
    E_ankle = 5.0e18
    # Approximate normalization at 1 TeV: dN/dE ~ 1.8e-13 1/(cm^2 s sr eV)
    E_norm = 1.0e12
    N_norm = 1.8e-13
    dnde = N_norm * (E / E_norm) ** (-a_lo)
    above_knee = E > E_knee
    above_ankle = E > E_ankle
    dnde[above_knee] = (
        N_norm
        * (E_knee / E_norm) ** (-a_lo)
        * (E[above_knee] / E_knee) ** (-a_mid)
    )
    dnde[above_ankle] = (
        N_norm
        * (E_knee / E_norm) ** (-a_lo)
        * (E_ankle / E_knee) ** (-a_mid)
        * (E[above_ankle] / E_ankle) ** (-a_hi)
    )
    return dnde
