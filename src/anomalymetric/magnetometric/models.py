"""SQUID flux-noise models (the rf-SQUID / dc-SQUID distinction lives here).

dc-SQUID and rf-SQUID are not separate `SpectrumKind`s — both yield a Gaussian
PSD channel. They differ only in the *shape* of the flux-noise floor, which is
captured by these two `Model`s (analogous to `cosmicray.knee_ankle`). Each
returns a predicted PSD S(f) on the shared log-energy axis (f = h^-1 * E).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.units import frequencies_hz


class DCSQUIDNoise:
    """dc-SQUID: white floor plus a 1/f^alpha low-frequency corner.

        S(f) = A * S_white * (1 + (f_knee / f)^alpha)

    Free: amplitude `A`. Locked: `S_white`, `f_knee_hz` (~1 Hz), `alpha` (~1).
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        S_white: float = 1.0e-12,
        f_knee_hz: float = 1.0,
        alpha: float = 1.0,
    ):
        self.name = "dc_squid_noise"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("S_white", S_white, min=0.0, max=np.inf, frozen=True),
                Parameter("f_knee_hz", f_knee_hz, min=1e-6, max=1e12, frozen=True),
                Parameter("alpha", alpha, min=0.0, max=3.0, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = frequencies_hz(log_energy_centers_eV)
        A = self.parameters["amplitude"].value
        S_white = self.parameters["S_white"].value
        f_knee = self.parameters["f_knee_hz"].value
        alpha = self.parameters["alpha"].value
        return A * S_white * (1.0 + (f_knee / f) ** alpha)


class RFSQUIDNoise:
    """rf-SQUID: dc-SQUID floor plus an rf-amplifier back-action rise at high f.

        S(f) = A * S_white * (1 + (f_knee / f)^alpha + (f / f_amp)^2)

    The extra `(f/f_amp)^2` term models tank-circuit / amplifier back-action that
    raises the noise toward the rf bias band. Free: `A`. Locked: shape constants.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        S_white: float = 1.0e-12,
        f_knee_hz: float = 1.0,
        alpha: float = 1.0,
        f_amp_hz: float = 1.0e8,
    ):
        self.name = "rf_squid_noise"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("S_white", S_white, min=0.0, max=np.inf, frozen=True),
                Parameter("f_knee_hz", f_knee_hz, min=1e-6, max=1e12, frozen=True),
                Parameter("alpha", alpha, min=0.0, max=3.0, frozen=True),
                Parameter("f_amp_hz", f_amp_hz, min=1.0, max=1e15, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = frequencies_hz(log_energy_centers_eV)
        A = self.parameters["amplitude"].value
        S_white = self.parameters["S_white"].value
        f_knee = self.parameters["f_knee_hz"].value
        alpha = self.parameters["alpha"].value
        f_amp = self.parameters["f_amp_hz"].value
        return A * S_white * (1.0 + (f_knee / f) ** alpha + (f / f_amp) ** 2)
