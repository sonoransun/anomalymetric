"""Differential-acceleration noise models for the gravitational channel.

Each is a `Model` returning a predicted PSD S(f) in (m/s^2)^2/Hz on the shared
log-energy axis (f = h^-1 * E). Free parameter is the overall amplitude; the
spectral corners/slopes are locked (a fixed instrument shape, as with the SQUID
noise models). These cover a torsion balance (Eot-Wash), a drag-free two-mass
differential accelerometer (MICROSCOPE), and a generic ground gravimeter.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.units import frequencies_hz


class TorsionBalanceNoise:
    """Eot-Wash-style torsion balance: thermal white + 1/f plus a resonance bump.

        S(f) = A * [ S_thermal * (1 + (f_knee / f)) + S_res / ((f^2 - f0^2)^2 + (f f0/Q)^2) ]
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        S_thermal: float = 1.0e-20,
        f_knee_hz: float = 1.0e-3,
        S_res: float = 1.0e-24,
        f0_hz: float = 1.0e-3,
        Q: float = 1.0e4,
    ):
        self.name = "torsion_balance_noise"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("S_thermal", S_thermal, min=0.0, max=np.inf, frozen=True),
                Parameter("f_knee_hz", f_knee_hz, min=1e-9, max=1e6, frozen=True),
                Parameter("S_res", S_res, min=0.0, max=np.inf, frozen=True),
                Parameter("f0_hz", f0_hz, min=1e-9, max=1e6, frozen=True),
                Parameter("Q", Q, min=1.0, max=1e9, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = frequencies_hz(log_energy_centers_eV)
        p = self.parameters
        A = p["amplitude"].value
        f0 = p["f0_hz"].value
        Q = p["Q"].value
        thermal = p["S_thermal"].value * (1.0 + p["f_knee_hz"].value / f)
        resonance = p["S_res"].value / ((f**2 - f0**2) ** 2 + (f * f0 / Q) ** 2)
        return A * (thermal + resonance)


class DifferentialAccelNoise:
    """MICROSCOPE-style drag-free differential accelerometer: flat floor + 1/f."""

    def __init__(
        self,
        amplitude: float = 1.0,
        S_floor: float = 1.0e-22,
        f_dragfree_hz: float = 1.0e-3,
        alpha: float = 1.0,
    ):
        self.name = "differential_accel_noise"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("S_floor", S_floor, min=0.0, max=np.inf, frozen=True),
                Parameter("f_dragfree_hz", f_dragfree_hz, min=1e-9, max=1e6, frozen=True),
                Parameter("alpha", alpha, min=0.0, max=3.0, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = frequencies_hz(log_energy_centers_eV)
        p = self.parameters
        return p["amplitude"].value * p["S_floor"].value * (
            1.0 + (p["f_dragfree_hz"].value / f) ** p["alpha"].value
        )


class GravimeterNoise:
    """Ground gravimeter: thermal + seismic (steep red) + Newtonian gradient."""

    def __init__(
        self,
        amplitude: float = 1.0,
        S_thermal: float = 1.0e-20,
        S_seismic: float = 1.0e-18,
        f_seismic_hz: float = 0.1,
        S_newtonian: float = 1.0e-19,
        f_nn_hz: float = 1.0e-2,
    ):
        self.name = "gravimeter_noise"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("S_thermal", S_thermal, min=0.0, max=np.inf, frozen=True),
                Parameter("S_seismic", S_seismic, min=0.0, max=np.inf, frozen=True),
                Parameter("f_seismic_hz", f_seismic_hz, min=1e-9, max=1e6, frozen=True),
                Parameter("S_newtonian", S_newtonian, min=0.0, max=np.inf, frozen=True),
                Parameter("f_nn_hz", f_nn_hz, min=1e-9, max=1e6, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = frequencies_hz(log_energy_centers_eV)
        p = self.parameters
        return p["amplitude"].value * (
            p["S_thermal"].value
            + p["S_seismic"].value * (p["f_seismic_hz"].value / f) ** 4
            + p["S_newtonian"].value * (p["f_nn_hz"].value / f) ** 2
        )
