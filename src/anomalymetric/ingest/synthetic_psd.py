"""Synthetic PSD spectra for the continuous Gaussian channels (SQUID, gravimeter).

Unlike `ingest.synthetic` (Poisson counts), these draw Gaussian noise around a
noise-floor PSD and optionally add a narrow line / broadband bump. An injected
line's strength is expressed as a multiple of the *local* noise floor at the line
center, so `line_amplitude=6` means "a peak six times the floor there".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from anomalymetric.gravitational.spectrum import (
    gravimeter_band_grid,
    grav_reference_psd,
    make_grav_spectrum,
)
from anomalymetric.magnetometric.spectrum import (
    make_squid_spectrum,
    squid_band_grid,
    squid_reference_psd,
)
from anomalymetric.models.exotic import BroadbandBump
from anomalymetric.spectrum import Spectrum, SpectrumKind
from anomalymetric.units import bin_centers_eV


def _draw_psd(reference: np.ndarray, sigma_frac: float, seed: Optional[int], noise: bool):
    sigma = np.clip(sigma_frac * reference, 1e-300, np.inf)
    if not noise:
        return reference.copy(), sigma
    rng = np.random.default_rng(seed)
    drawn = rng.normal(reference, sigma)
    return np.clip(drawn, 1e-300, np.inf), sigma


def _psd_upper_limit_mask(reference: np.ndarray, ul_fraction: float):
    """Flag the lowest-PSD bins as upper limits (weakest-floor channels)."""
    if ul_fraction <= 0.0:
        return None
    n = reference.shape[0]
    n_ul = int(np.floor(ul_fraction * n))
    if n_ul <= 0:
        return None
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(reference)[:n_ul]] = True
    return mask


def _add_line(reference: np.ndarray, log_centers: np.ndarray, E_center_eV: float,
              line_amplitude: float, width_dex: float) -> np.ndarray:
    """Add a peak-height line scaled to `line_amplitude` x the local floor."""
    bump = BroadbandBump(amplitude=1.0, E_center_eV=E_center_eV, width_dex=width_dex)
    shape = bump.dnde(log_centers)  # peaks at 1.0
    peak_bin = int(np.argmax(shape))
    return reference + line_amplitude * reference[peak_bin] * shape


def synthetic_squid_natural(
    log_f_min: float = 2.0,
    log_f_max: float = 9.0,
    bins_per_decade: int = 20,
    S_white: float = 1.0e-12,
    f_knee_hz: float = 1.0,
    sigma_frac: float = 0.05,
    seed: Optional[int] = None,
    noise: bool = True,
    ul_fraction: float = 0.0,
) -> Spectrum:
    """SQUID flux-noise floor sampled with Gaussian noise."""
    edges = squid_band_grid(log_f_min, log_f_max, bins_per_decade)
    log_centers = np.log10(bin_centers_eV(edges))
    reference = squid_reference_psd(log_centers, S_white=S_white, f_knee_hz=f_knee_hz)
    value, sigma = _draw_psd(reference, sigma_frac, seed, noise)
    ul_mask = _psd_upper_limit_mask(reference, ul_fraction)
    spec = make_squid_spectrum(edges, value, sigma, upper_limit_mask=ul_mask)
    spec.meta.update({"synthetic": True, "S_white": S_white, "f_knee_hz": f_knee_hz, "seed": seed})
    return spec


def synthetic_squid_with_exotic(
    base: Spectrum,
    line_E_eV: float,
    line_amplitude: float,
    width_dex: float = 0.02,
    seed: Optional[int] = None,
) -> Spectrum:
    """Add a narrow PSD line to a SQUID base spectrum and redraw Gaussian noise.

    `base` should be a *noiseless* floor (built with `noise=False`); its `value`
    is treated as the noise-floor PSD onto which the line is added before a single
    Gaussian draw.
    """
    log_centers = np.log10(base.energy_centers_eV)
    floor = base.value
    reference = _add_line(floor, log_centers, line_E_eV, line_amplitude, width_dex)
    sigma = base.uncertainty
    rng = np.random.default_rng(seed)
    value = np.clip(rng.normal(reference, sigma), 1e-300, np.inf)
    spec = make_squid_spectrum(base.log_energy_edges_eV, value, sigma, kind=base.kind)
    meta = dict(base.meta)
    meta.update({"injected_line_eV": line_E_eV, "injected_amplitude": line_amplitude})
    spec.meta = meta
    return spec


def synthetic_grav_natural(
    log_f_min: float = -5.0,
    log_f_max: float = 1.0,
    bins_per_decade: int = 20,
    sigma_frac: float = 0.05,
    seed: Optional[int] = None,
    noise: bool = True,
    ul_fraction: float = 0.0,
) -> Spectrum:
    """Differential-acceleration noise floor sampled with Gaussian noise."""
    edges = gravimeter_band_grid(log_f_min, log_f_max, bins_per_decade)
    log_centers = np.log10(bin_centers_eV(edges))
    reference = grav_reference_psd(log_centers)
    value, sigma = _draw_psd(reference, sigma_frac, seed, noise)
    ul_mask = _psd_upper_limit_mask(reference, ul_fraction)
    spec = make_grav_spectrum(edges, value, sigma, upper_limit_mask=ul_mask)
    spec.meta.update({"synthetic": True, "seed": seed})
    return spec


def synthetic_grav_with_exotic(
    base: Spectrum,
    line_E_eV: float,
    line_amplitude: float,
    width_dex: float = 0.02,
    seed: Optional[int] = None,
) -> Spectrum:
    """Add a narrow PSD line to a gravimeter base spectrum and redraw noise.

    `base` should be a *noiseless* floor (built with `noise=False`).
    """
    log_centers = np.log10(base.energy_centers_eV)
    reference = _add_line(base.value, log_centers, line_E_eV, line_amplitude, width_dex)
    sigma = base.uncertainty
    rng = np.random.default_rng(seed)
    value = np.clip(rng.normal(reference, sigma), 1e-300, np.inf)
    spec = make_grav_spectrum(base.log_energy_edges_eV, value, sigma, kind=base.kind)
    meta = dict(base.meta)
    meta.update({"injected_line_eV": line_E_eV, "injected_amplitude": line_amplitude})
    spec.meta = meta
    return spec


_SQUID_BASE_KEYS = {"log_f_min", "log_f_max", "bins_per_decade", "S_white", "f_knee_hz", "sigma_frac", "seed", "noise", "ul_fraction"}
_GRAV_BASE_KEYS = {"log_f_min", "log_f_max", "bins_per_decade", "sigma_frac", "seed", "noise", "ul_fraction"}


@dataclass
class SQUIDSyntheticLoader:
    """Entry-point loader: `source` in {'noise_floor', 'axion_line', 'dark_photon', 'bump'}."""

    name: str = "squid"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        if source == "noise_floor":
            return synthetic_squid_natural(**{k: v for k, v in kwargs.items() if k in _SQUID_BASE_KEYS})
        if source in {"axion_line", "dark_photon", "bump"}:
            base_kwargs = {k: v for k, v in kwargs.items() if k in _SQUID_BASE_KEYS}
            base_kwargs["noise"] = False  # add the line to a clean floor, draw noise once
            base = synthetic_squid_natural(**base_kwargs)
            mass_eV = kwargs.get("mass_eV")
            line_E_eV = float(mass_eV) if mass_eV is not None else float(kwargs.get("line_E_eV", 4.0e-6))
            width = 0.3 if source == "bump" else float(kwargs.get("width_dex", 0.02))
            return synthetic_squid_with_exotic(
                base,
                line_E_eV=line_E_eV,
                line_amplitude=float(kwargs.get("line_amplitude", 6.0)),
                width_dex=width,
                seed=kwargs.get("seed"),
            )
        raise ValueError(f"Unknown squid source '{source}'")


@dataclass
class GravSyntheticLoader:
    """Entry-point loader: `source` in {'noise_floor', 'yukawa', 'ep_violation', 'osc_dm', 'bump'}."""

    name: str = "gravimeter"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        if source == "noise_floor":
            return synthetic_grav_natural(**{k: v for k, v in kwargs.items() if k in _GRAV_BASE_KEYS})
        if source in {"yukawa", "ep_violation", "osc_dm", "bump"}:
            from anomalymetric.gravitational.fifth_force import (
                microscope_modulation_freq_hz,
                oscillating_dm_freq_hz,
            )
            from anomalymetric.units import H_PLANCK_EV_S

            base_kwargs = {k: v for k, v in kwargs.items() if k in _GRAV_BASE_KEYS}
            base_kwargs["noise"] = False
            base = synthetic_grav_natural(**base_kwargs)
            if "line_E_eV" in kwargs:
                line_E_eV = float(kwargs["line_E_eV"])
            elif source == "yukawa":
                line_E_eV = H_PLANCK_EV_S * float(kwargs.get("mod_freq_hz", 1.0e-2))
            elif source == "ep_violation":
                line_E_eV = H_PLANCK_EV_S * microscope_modulation_freq_hz(
                    float(kwargs.get("orbital_period_s", 5946.0))
                )
            elif source == "osc_dm":
                line_E_eV = H_PLANCK_EV_S * oscillating_dm_freq_hz(float(kwargs.get("mass_eV", 1.0e-15)))
            else:  # bump
                line_E_eV = float(kwargs.get("line_E_eV", H_PLANCK_EV_S * 1.0e-3))
            width = 0.3 if source == "bump" else float(kwargs.get("width_dex", 0.02))
            return synthetic_grav_with_exotic(
                base,
                line_E_eV=line_E_eV,
                line_amplitude=float(kwargs.get("line_amplitude", 6.0)),
                width_dex=width,
                seed=kwargs.get("seed"),
            )
        raise ValueError(f"Unknown gravimeter source '{source}'")
