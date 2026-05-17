"""Three-segment power-law for fitting the knee, ankle, and GZK suppression.

Layered on top of `BrokenPowerLaw` for two-segment cases, plus a triple-break
specialization used by the all-particle CR test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters


class TripleBrokenPowerLaw:
    """Three smoothly-broken power-law segments (low / knee / ankle / GZK)."""

    def __init__(
        self,
        amplitude: float = 1.8e-13,
        index_lo: float = 2.7,
        index_mid: float = 3.1,
        index_hi: float = 2.7,
        index_gzk: float = 5.0,
        E_knee_eV: float = 3.0e15,
        E_ankle_eV: float = 5.0e18,
        E_gzk_eV: float = 5.0e19,
        reference_eV: float = 1.0e12,
        smoothness: float = 5.0,
    ):
        self.name = "triple_broken_pl"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("index_lo", index_lo, min=0.0, max=5.0),
                Parameter("index_mid", index_mid, min=0.0, max=5.0),
                Parameter("index_hi", index_hi, min=0.0, max=5.0),
                Parameter("index_gzk", index_gzk, min=0.0, max=20.0),
                Parameter("E_knee_eV", E_knee_eV, min=1e12, max=1e17),
                Parameter("E_ankle_eV", E_ankle_eV, min=1e16, max=1e20),
                Parameter("E_gzk_eV", E_gzk_eV, min=1e18, max=1e21),
                Parameter("reference_eV", reference_eV, min=0.0, max=np.inf, frozen=True),
                Parameter("smoothness", smoothness, min=0.5, max=20.0, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        p = self.parameters
        A = p["amplitude"].value
        a1, a2, a3, a4 = (
            p["index_lo"].value,
            p["index_mid"].value,
            p["index_hi"].value,
            p["index_gzk"].value,
        )
        E1 = p["E_knee_eV"].value
        E2 = p["E_ankle_eV"].value
        E3 = p["E_gzk_eV"].value
        E0 = p["reference_eV"].value
        s = p["smoothness"].value
        base = A * (E / E0) ** (-a1)
        base *= (1.0 + (E / E1) ** s) ** ((a1 - a2) / s)
        base *= (1.0 + (E / E2) ** s) ** ((a2 - a3) / s)
        base *= (1.0 + (E / E3) ** s) ** ((a3 - a4) / s)
        return base
