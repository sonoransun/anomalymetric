"""Magnetometric (SQUID) spectrum constructors.

A magnetometric spectrum is a `Spectrum` with `kind=MAGNETOMETRIC` whose value is
a power spectral density of the measured flux noise, S(f), on a frequency axis
mapped to the shared log10(E/eV) axis via E = h*nu. The natural baseline is the
instrument flux-noise floor (white + 1/f); an anomaly is a narrow line (axion /
dark-photon haloscope signal) or a broadband bump above that floor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import ev_to_hz, log_frequency_grid


def squid_band_grid(
    log_f_min: float = -1.0, log_f_max: float = 9.0, bins_per_decade: int = 20
) -> NDArray[np.float64]:
    """Log-frequency grid (~0.1 Hz to ~1 GHz) as log10(E/eV) edges."""
    return log_frequency_grid(log_f_min, log_f_max, bins_per_decade)


def make_squid_spectrum(
    log_energy_edges_eV: NDArray[np.float64],
    psd: NDArray[np.float64],
    sigma_psd: NDArray[np.float64],
    *,
    kind: SpectrumKind = SpectrumKind.MAGNETOMETRIC,
    upper_limit_mask: Optional[NDArray[np.bool_]] = None,
) -> Spectrum:
    """Build a magnetometric PSD `Spectrum` with its per-bin Gaussian sigma."""
    return Spectrum(
        log_energy_edges_eV=np.asarray(log_energy_edges_eV, dtype=float),
        value=np.asarray(psd, dtype=float),
        value_kind=ValueKind.PSD_PER_BIN,
        kind=kind,
        uncertainty=np.asarray(sigma_psd, dtype=float),
        upper_limit_mask=upper_limit_mask,
    )


def squid_reference_psd(
    log_energy_centers_eV: NDArray[np.float64],
    *,
    S_white: float = 1.0e-12,
    f_knee_hz: float = 1.0,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Fixed-shape SQUID flux-noise floor: white plus a 1/f^alpha corner.

        S(f) = S_white * (1 + (f_knee / f)^alpha)

    `S_white` is in flux-noise PSD units, e.g. (Phi0/sqrt(Hz))^2. Defaults are
    illustrative (a ~uPhi0/sqrt(Hz) device with a 1 Hz 1/f corner), tuned for
    tests rather than publication-quality numbers.
    """
    f = ev_to_hz(10.0 ** np.asarray(log_energy_centers_eV, dtype=float))
    return S_white * (1.0 + (f_knee_hz / f) ** alpha)
