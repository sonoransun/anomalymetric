"""Power-law and broken-power-law non-thermal components.

`naima` provides physical synchrotron / inverse-Compton / pi-zero models; we
keep a minimal parametric stand-in here so the core has zero non-standard
dependencies. If users install the `[naima]` extra, downstream code can
substitute a `naima`-backed `Model` with the same interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters


class PowerLaw:
    """dN/dE = A * (E/E0)^{-alpha}."""

    def __init__(
        self,
        amplitude: float = 1.0,
        index: float = 2.0,
        reference_eV: float = 1.0e3,
    ):
        self.name = "powerlaw"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("index", index, min=-5.0, max=10.0),
                Parameter("reference_eV", reference_eV, min=0.0, max=np.inf, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        A = self.parameters["amplitude"].value
        alpha = self.parameters["index"].value
        E0 = self.parameters["reference_eV"].value
        return A * (E / E0) ** (-alpha)


class BrokenPowerLaw:
    """Two-slope spectrum with a soft transition at `E_break_eV`.

    Used by the cosmic-ray module to fit knee/ankle/GZK features.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        index_lo: float = 2.7,
        index_hi: float = 3.1,
        E_break_eV: float = 3.0e15,
        reference_eV: float = 1.0e12,
        smoothness: float = 5.0,
    ):
        self.name = "broken_powerlaw"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("index_lo", index_lo, min=-5.0, max=10.0),
                Parameter("index_hi", index_hi, min=-5.0, max=10.0),
                Parameter("E_break_eV", E_break_eV, min=1.0, max=1e22),
                Parameter("reference_eV", reference_eV, min=0.0, max=np.inf, frozen=True),
                Parameter("smoothness", smoothness, min=0.1, max=50.0, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        A = self.parameters["amplitude"].value
        a1 = self.parameters["index_lo"].value
        a2 = self.parameters["index_hi"].value
        Eb = self.parameters["E_break_eV"].value
        E0 = self.parameters["reference_eV"].value
        s = self.parameters["smoothness"].value
        # Smoothly broken power-law (Tracy/Band-style):
        #   dN/dE = A (E/E0)^-a1 * (1 + (E/Eb)^s)^((a1-a2)/s)
        return A * (E / E0) ** (-a1) * (1.0 + (E / Eb) ** s) ** ((a1 - a2) / s)
