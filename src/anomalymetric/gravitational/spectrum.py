"""Gravitational differential (gradiometer / torsion-balance) spectrum builders.

A gravitational spectrum is a `Spectrum` with `kind=GRAVITATIONAL` whose value is
the PSD of a *differential* acceleration measurement S(f) in (m/s^2)^2/Hz, on a
frequency axis mapped to log10(E/eV) via E = h*nu. The natural baseline mixes
thermal (white), seismic (steep red), and Newtonian gravity-gradient noise. An
anomaly is a fifth-force / equivalence-principle / oscillating-dark-matter line
or a broadband bump above that floor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import ev_to_hz, log_frequency_grid


def gravimeter_band_grid(
    log_f_min: float = -6.0, log_f_max: float = 3.0, bins_per_decade: int = 20
) -> NDArray[np.float64]:
    """Log-frequency grid (~1 uHz to ~1 kHz) as log10(E/eV) edges."""
    return log_frequency_grid(log_f_min, log_f_max, bins_per_decade)


def make_grav_spectrum(
    log_energy_edges_eV: NDArray[np.float64],
    psd: NDArray[np.float64],
    sigma_psd: NDArray[np.float64],
    *,
    kind: SpectrumKind = SpectrumKind.GRAVITATIONAL,
    upper_limit_mask: Optional[NDArray[np.bool_]] = None,
) -> Spectrum:
    """Build a gravitational PSD `Spectrum` with its per-bin Gaussian sigma."""
    return Spectrum(
        log_energy_edges_eV=np.asarray(log_energy_edges_eV, dtype=float),
        value=np.asarray(psd, dtype=float),
        value_kind=ValueKind.PSD_PER_BIN,
        kind=kind,
        uncertainty=np.asarray(sigma_psd, dtype=float),
        upper_limit_mask=upper_limit_mask,
    )


def grav_reference_psd(
    log_energy_centers_eV: NDArray[np.float64],
    *,
    S_thermal: float = 1.0e-20,
    S_seismic: float = 1.0e-18,
    f_seismic_hz: float = 0.1,
    S_newtonian: float = 1.0e-19,
    f_nn_hz: float = 1.0e-2,
) -> NDArray[np.float64]:
    """Fixed-shape differential-acceleration noise floor, in (m/s^2)^2/Hz.

        S(f) = S_thermal + S_seismic * (f_seismic / f)^4 + S_newtonian * (f_nn / f)^2

    thermal (white suspension loss) + seismic (steep red below `f_seismic`) +
    Newtonian gravity-gradient ("gravity gradient noise"). Defaults illustrative.
    """
    f = ev_to_hz(10.0 ** np.asarray(log_energy_centers_eV, dtype=float))
    return (
        S_thermal
        + S_seismic * (f_seismic_hz / f) ** 4
        + S_newtonian * (f_nn_hz / f) ** 2
    )
