"""Synthetic-spectrum generators (used by tests and the CLI demo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.lines import GaussianLine
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import log_energy_grid


def synthetic_natural(
    log_e_min: float = -3,
    log_e_max: float = 6,
    bins_per_decade: int = 20,
    T_K: float = 300.0,
    bb_amplitude: float = 1.0,
    pl_amplitude: float = 1e-3,
    pl_index: float = 2.0,
    exposure_cm2_s: float = 1.0,
    seed: Optional[int] = None,
    poisson: bool = True,
) -> Spectrum:
    """Blackbody + power-law background sampled with Poisson noise."""
    edges = log_energy_grid(log_e_min, log_e_max, bins_per_decade)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bb = BlackBody(T_K=T_K, amplitude=bb_amplitude)
    pl = PowerLaw(amplitude=pl_amplitude, index=pl_index, reference_eV=1.0)
    dnde = bb.dnde(centers) + pl.dnde(centers)
    widths = np.diff(10.0**edges)
    expected = dnde * widths * exposure_cm2_s
    rng = np.random.default_rng(seed)
    if poisson:
        observed = rng.poisson(np.clip(expected, 0, None)).astype(float)
    else:
        observed = expected.copy()
    return Spectrum(
        log_energy_edges_eV=edges,
        value=observed,
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=SpectrumKind.PHOTON,
        exposure_cm2_s=np.full_like(observed, exposure_cm2_s),
        meta={
            "synthetic": True,
            "T_K": T_K,
            "bb_amplitude": bb_amplitude,
            "pl_amplitude": pl_amplitude,
            "pl_index": pl_index,
            "seed": seed,
        },
    )


def synthetic_with_exotic(
    base: Spectrum,
    line_E_eV: float,
    line_amplitude: float,
    sigma_dex: float = 0.005,
    seed: Optional[int] = None,
) -> Spectrum:
    """Add a Gaussian line on top of an existing synthetic spectrum (re-Poissonize)."""
    centers = base.energy_centers_eV
    log_centers = np.log10(centers)
    widths = base.bin_widths_eV
    line = GaussianLine(amplitude=line_amplitude, E_center_eV=line_E_eV, sigma_dex=sigma_dex)
    exposure = base.exposure_cm2_s if base.exposure_cm2_s is not None else np.ones_like(widths)
    added = line.dnde(log_centers) * widths * exposure
    expected_total = base.value + added
    rng = np.random.default_rng(seed)
    observed = rng.poisson(np.clip(expected_total, 0, None)).astype(float)
    meta = dict(base.meta)
    meta.update({"injected_line_eV": line_E_eV, "injected_amplitude": line_amplitude})
    return Spectrum(
        log_energy_edges_eV=base.log_energy_edges_eV,
        value=observed,
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=base.kind,
        exposure_cm2_s=exposure.copy(),
        meta=meta,
    )


@dataclass
class SyntheticLoader:
    """Entry-point loader: `source` is one of {'blackbody', 'exotic_line'}."""

    name: str = "synthetic"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        if source == "blackbody":
            return synthetic_natural(**kwargs)
        if source == "exotic_line":
            base = synthetic_natural(**{k: v for k, v in kwargs.items() if k in {
                "log_e_min", "log_e_max", "bins_per_decade", "T_K", "bb_amplitude",
                "pl_amplitude", "pl_index", "exposure_cm2_s", "seed", "poisson",
            }})
            return synthetic_with_exotic(
                base,
                line_E_eV=float(kwargs.get("line_E_eV", 2.33)),
                line_amplitude=float(kwargs.get("line_amplitude", 100.0)),
                sigma_dex=float(kwargs.get("sigma_dex", 0.005)),
                seed=kwargs.get("seed"),
            )
        raise ValueError(f"Unknown synthetic source '{source}'")
