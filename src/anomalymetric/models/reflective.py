"""Solar-reflection component.

For v1 we use a 5778 K Planck spectrum stand-in for the solar SED (replace
with a Kurucz template via `data/references/` once available). The free
parameters are `albedo` and a phase-angle factor that scales the reflected
flux but not the spectral shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.models.thermal import BlackBody

T_SUN_K = 5778.0


class SolarReflection:
    """Greybody-shaped reflection. Spectral shape pinned to a 5778 K blackbody."""

    def __init__(self, albedo: float = 0.1, phase_factor: float = 1.0):
        self.name = "solar_reflection"
        self._sun = BlackBody(T_K=T_SUN_K, amplitude=1.0)
        # Freeze the sun temperature; reflection only scales overall amplitude.
        self._sun.parameters["T_K"].frozen = True
        self._sun.parameters["amplitude"].frozen = True
        self.parameters = Parameters(
            [
                Parameter("albedo", albedo, min=0.0, max=1.0),
                Parameter("phase_factor", phase_factor, min=0.0, max=10.0),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = self._sun.dnde(log_energy_centers_eV)
        a = self.parameters["albedo"].value
        p = self.parameters["phase_factor"].value
        return a * p * shape
