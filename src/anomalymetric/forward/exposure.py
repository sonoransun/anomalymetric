"""Exposure (effective area × livetime) for non-photon channels.

Cosmic-ray and neutrino datasets typically arrive as energy-binned event counts
plus an exposure curve `A_eff(E) * T_live`. The simple `FlatExposure` below
covers v1 needs; instrument-specific exposures can subclass `ForwardModel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anomalymetric.units import bin_widths_eV


@dataclass
class FlatExposure:
    """Energy-independent A_eff * T_live; same forward signature as responses."""

    exposure_cm2_s: float = 1.0
    solid_angle_sr: float = 1.0

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_dnde: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        widths = bin_widths_eV(log_energy_edges_eV)
        return (
            np.asarray(source_dnde, dtype=float)
            * widths
            * self.exposure_cm2_s
            * self.solid_angle_sr
        )


@dataclass
class TabulatedExposure:
    """Per-bin exposure curve."""

    exposure_per_bin_cm2_s: NDArray[np.float64]

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_dnde: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        widths = bin_widths_eV(log_energy_edges_eV)
        expo = np.asarray(self.exposure_per_bin_cm2_s, dtype=float)
        if expo.shape != widths.shape:
            raise ValueError(
                f"exposure has {expo.shape[0]} bins but spectrum has {widths.shape[0]}"
            )
        return np.asarray(source_dnde, dtype=float) * widths * expo
