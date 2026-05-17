"""Gaussian emission lines on a log-energy axis.

The line's energy width is naturally narrow in absolute terms, so we
parameterize sigma in **log10(E/eV) dex** to match instrument resolution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters


class GaussianLine:
    """Gaussian in log-E. Integrated photon flux = `amplitude` * dex normalization."""

    def __init__(
        self,
        amplitude: float = 1.0,
        E_center_eV: float = 1.0,
        sigma_dex: float = 0.02,
    ):
        self.name = "gaussian_line"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("E_center_eV", E_center_eV, min=1e-9, max=1e22),
                Parameter("sigma_dex", sigma_dex, min=1e-4, max=2.0),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        log_E = np.asarray(log_energy_centers_eV, dtype=float)
        E = 10.0**log_E
        A = self.parameters["amplitude"].value
        log_Ec = np.log10(self.parameters["E_center_eV"].value)
        sig = self.parameters["sigma_dex"].value
        # Gaussian in log-E; divide by E*ln(10) so that integral over dE = A.
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sig)
        gauss = norm * np.exp(-0.5 * ((log_E - log_Ec) / sig) ** 2)
        return A * gauss / (E * np.log(10.0))
