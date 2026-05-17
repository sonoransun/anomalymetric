"""Distance and K-correction helpers.

Scores live in rest-frame luminosity, not observed flux, so multi-object
catalog ranking needs at minimum a redshift correction. Heavy cosmology is
deferred to astropy; we expose two small helpers used by the pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shift_to_rest_frame(
    log_energy_edges_eV: NDArray[np.float64], z: float
) -> NDArray[np.float64]:
    """Return log-energy edges shifted to the rest frame (E_rest = E_obs * (1+z))."""
    if z < 0:
        raise ValueError("redshift z must be non-negative")
    return np.asarray(log_energy_edges_eV, dtype=float) + np.log10(1.0 + z)


def k_correction_factor(dnde_rest_alpha: float, z: float) -> float:
    """Power-law K-correction factor (1+z)^(alpha-1) for spectral index alpha.

    Equivalent to the standard SED K-correction for `dN/dE \\propto E^{-alpha}`.
    """
    return float((1.0 + z) ** (dnde_rest_alpha - 1.0))


def luminosity_distance_cm(z: float, H0_km_s_Mpc: float = 70.0, Omega_m: float = 0.3) -> float:
    """Flat-LCDM luminosity distance via astropy (lazy import)."""
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=H0_km_s_Mpc, Om0=Omega_m)
    return float(cosmo.luminosity_distance(z).to(u.cm).value)
