"""Planck blackbody, expressed as dN/dE in 1/(s cm^2 eV) at the source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Model, Parameter, Parameters

K_BOLTZMANN_EV_K = 8.617333262e-5
H_PLANCK_EV_S = 4.135667696e-15
C_LIGHT_M_S = 2.99792458e8


@dataclass
class BlackBody:
    """Greybody dN/dE = A * (8 pi / (h^3 c^2)) * E^2 / (exp(E/kT) - 1).

    Free parameters: temperature `T_K` and amplitude `A` (dimensionless emissivity
    times projected emitting area divided by 4 pi d^2 for an isotropic source).
    """

    name: str = "blackbody"
    parameters: Parameters = None  # type: ignore[assignment]

    def __init__(self, T_K: float = 300.0, amplitude: float = 1.0):
        self.name = "blackbody"
        self.parameters = Parameters(
            [
                Parameter("T_K", T_K, min=1.0, max=1e8),
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        T = self.parameters["T_K"].value
        A = self.parameters["amplitude"].value
        kT = K_BOLTZMANN_EV_K * T
        # Avoid overflow in exp.
        x = np.clip(E / kT, 1e-30, 700.0)
        # Photon number spectral density: 8 pi / (h^3 c^2) E^2 / (e^{E/kT}-1)
        # In our 1/(s cm^2 eV) bookkeeping we collapse the geometric prefactor
        # into the free amplitude — `A` carries units implicitly.
        return A * E * E / np.expm1(x)
