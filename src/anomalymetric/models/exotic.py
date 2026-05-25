"""Matched-filter library of exotic templates.

The Loeb–Turner argument is most rigorous against a *fixed* alternative — a
laser at a known lab wavelength, an axion-decay line at a known mass, a hard
spectral cutoff at the GZK energy. A free-floating exotic component is
statistically degenerate: it will always win the likelihood ratio.

Each template here is a `Model` with locked shape parameters (so the only
freedom is amplitude, optionally with a coarse scan over center energy). The
scorer iterates the library and applies a Gross–Vitells trials correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.models.lines import GaussianLine
from anomalymetric.units import wavelength_nm_to_ev


# Standard high-power laser lines (CW or pulsed) likely to be visible in
# any artificial-light SETI residual. Wavelengths in nm.
LASER_LINES_NM = {
    "Nd_YAG_1064": 1064.0,
    "Nd_YAG_532": 532.0,
    "Nd_YAG_355": 355.0,
    "HeNe_633": 632.8,
    "ArIon_488": 488.0,
    "ArIon_514": 514.5,
    "CO2_10600": 10600.0,
}


@dataclass
class ExoticTemplate:
    name: str
    factory: Callable[[], object]


def laser_line_template(label: str, wavelength_nm: float) -> ExoticTemplate:
    """Gaussian line with center pinned to the laser energy; amplitude free."""

    def make() -> object:
        E_ev = float(wavelength_nm_to_ev(wavelength_nm))
        line = GaussianLine(amplitude=1.0, E_center_eV=E_ev, sigma_dex=0.005)
        line.name = f"laser.{label}"
        line.parameters["E_center_eV"].frozen = True
        line.parameters["sigma_dex"].frozen = True
        return line

    return ExoticTemplate(name=f"laser.{label}", factory=make)


def axion_line_template(label: str, mass_eV: float) -> ExoticTemplate:
    def make() -> object:
        line = GaussianLine(amplitude=1.0, E_center_eV=mass_eV / 2.0, sigma_dex=0.003)
        line.name = f"axion.{label}"
        line.parameters["E_center_eV"].frozen = True
        line.parameters["sigma_dex"].frozen = True
        return line

    return ExoticTemplate(name=f"axion.{label}", factory=make)


class HardCutoffPowerLaw:
    """Power-law with exponential cutoff at a *fixed* high energy.

    Useful as an exotic-tail template: a population component that looks like
    a hard injection beyond the expected cooling break.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        index: float = 1.5,
        E_cut_eV: float = 1.0e15,
        reference_eV: float = 1.0e9,
    ):
        self.name = "hard_cutoff_pl"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("index", index, min=-2.0, max=4.0, frozen=True),
                Parameter("E_cut_eV", E_cut_eV, min=1.0, max=1e22, frozen=True),
                Parameter("reference_eV", reference_eV, min=0.0, max=np.inf, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        A = self.parameters["amplitude"].value
        a = self.parameters["index"].value
        Ec = self.parameters["E_cut_eV"].value
        E0 = self.parameters["reference_eV"].value
        return A * (E / E0) ** (-a) * np.exp(-E / Ec)


def hard_cutoff_template(label: str, index: float, E_cut_eV: float) -> ExoticTemplate:
    def make() -> object:
        m = HardCutoffPowerLaw(amplitude=1.0, index=index, E_cut_eV=E_cut_eV)
        m.name = f"hardcut.{label}"
        return m

    return ExoticTemplate(name=f"hardcut.{label}", factory=make)


class GZKViolatingTail:
    """Excess flux above the GZK cutoff (~5e19 eV) — a CR-side exotic signature."""

    def __init__(self, amplitude: float = 1.0, index: float = 2.0, E_min_eV: float = 5.0e19):
        self.name = "gzk_violating_tail"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("index", index, min=0.0, max=5.0, frozen=True),
                Parameter("E_min_eV", E_min_eV, min=1e15, max=1e22, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        E = 10.0 ** np.asarray(log_energy_centers_eV, dtype=float)
        A = self.parameters["amplitude"].value
        a = self.parameters["index"].value
        Em = self.parameters["E_min_eV"].value
        # Smooth high-pass: zero below threshold, power-law above.
        x = np.clip((E - Em) / Em, 0.0, np.inf)
        return A * x * (E / Em) ** (-a)


def gzk_violating_template(label: str = "default") -> ExoticTemplate:
    def make() -> object:
        m = GZKViolatingTail()
        m.name = f"gzk.{label}"
        return m

    return ExoticTemplate(name=f"gzk.{label}", factory=make)


class BroadbandBump:
    """A log-normal bump on the analysis axis — the PSD analog of a hard cutoff.

    Used by the continuous sensor channels for *unmodeled* broadband excess
    (a smooth power bump rather than a narrow line). Returns the value directly
    (interpreted as a PSD for sensor channels), without the 1/E density factor a
    photon line carries. Center and width are locked; only amplitude is free.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        E_center_eV: float = 1.0,
        width_dex: float = 0.3,
    ):
        self.name = "broadband_bump"
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude, min=0.0, max=np.inf),
                Parameter("E_center_eV", E_center_eV, min=1e-30, max=1e22, frozen=True),
                Parameter("width_dex", width_dex, min=1e-3, max=3.0, frozen=True),
            ]
        )

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        log_E = np.asarray(log_energy_centers_eV, dtype=float)
        A = self.parameters["amplitude"].value
        log_Ec = np.log10(self.parameters["E_center_eV"].value)
        w = self.parameters["width_dex"].value
        return A * np.exp(-0.5 * ((log_E - log_Ec) / w) ** 2)


def broadband_bump_template(label: str, E_center_eV: float, width_dex: float = 0.3) -> ExoticTemplate:
    def make() -> object:
        m = BroadbandBump(amplitude=1.0, E_center_eV=E_center_eV, width_dex=width_dex)
        m.name = f"bump.{label}"
        return m

    return ExoticTemplate(name=f"bump.{label}", factory=make)


def psd_line_template(name: str, E_center_eV: float, width_dex: float = 0.02) -> ExoticTemplate:
    """A narrow PSD line (peak-height-parameterized bump) with a free amplitude.

    Shared by the magnetometric and gravitational channels for their axion /
    fifth-force / EP / oscillating-DM lines; center and width are locked, so the
    optimizer only fits the line height.
    """

    def make() -> object:
        line = BroadbandBump(amplitude=1.0, E_center_eV=E_center_eV, width_dex=width_dex)
        line.name = name
        return line

    return ExoticTemplate(name=name, factory=make)


def default_library() -> list[ExoticTemplate]:
    """Curated v1 library covering photon and CR exotic signatures."""
    templates: list[ExoticTemplate] = []
    for label, lam in LASER_LINES_NM.items():
        templates.append(laser_line_template(label, lam))
    # An axion line at m_a ~ 5 eV (illustrative — real searches scan a grid).
    templates.append(axion_line_template("5eV_demo", 5.0))
    # Hard-cutoff power laws at three reference energies.
    templates.append(hard_cutoff_template("100keV", 1.5, 1.0e5))
    templates.append(hard_cutoff_template("1TeV", 1.8, 1.0e12))
    templates.append(hard_cutoff_template("1PeV", 2.0, 1.0e15))
    templates.append(gzk_violating_template("5e19"))
    return templates
