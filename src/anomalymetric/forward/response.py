"""Instrument response: identity + a Gaussian-resolution placeholder.

Real IRF/ARF/RMF files (Fermi-LAT, Chandra, ...) are out of scope for v1 —
they require CALDB integration. The Gaussian-resolution operator below is
enough to (a) exercise the likelihood plumbing end-to-end and (b) keep the
public `ForwardModel` API stable when real responses are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anomalymetric.units import bin_centers_eV, bin_widths_eV


@dataclass
class IdentityResponse:
    """Multiply by bin width and exposure — no folding."""

    exposure_cm2_s: float = 1.0

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_dnde: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        widths = bin_widths_eV(log_energy_edges_eV)
        return np.asarray(source_dnde, dtype=float) * widths * self.exposure_cm2_s


@dataclass
class GaussianEnergyResponse:
    """Gaussian energy-dispersion smearing in log-energy space.

    `sigma_dex` is the resolution in decades — a 10% energy resolution is
    roughly 0.043 dex. The redistribution matrix is precomputed lazily.
    """

    sigma_dex: float = 0.05
    exposure_cm2_s: float = 1.0

    def _rmf(self, log_edges: NDArray[np.float64]) -> NDArray[np.float64]:
        centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        # Probability that a photon born in true-bin j is reconstructed in
        # measured-bin i: integral of a Gaussian centered at center_j over the
        # log-edges of bin i.
        from scipy.special import erf

        diff = (log_edges[None, :] - centers[:, None]) / (np.sqrt(2.0) * self.sigma_dex)
        cdf = 0.5 * (1.0 + erf(diff))
        rmf = cdf[:, 1:] - cdf[:, :-1]  # shape (n_true, n_meas)
        return rmf

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_dnde: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        edges = np.asarray(log_energy_edges_eV, dtype=float)
        widths = bin_widths_eV(edges)
        true_counts = np.asarray(source_dnde, dtype=float) * widths * self.exposure_cm2_s
        rmf = self._rmf(edges)
        return true_counts @ rmf
