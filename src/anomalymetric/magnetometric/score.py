"""Magnetometric PLR scorer: deviation from the SQUID noise floor.

Mirrors `cosmicray.score`: the natural hypothesis is *one* fixed-shape baseline
(the instrument flux-noise floor) with a free overall amplitude; the exotic
library holds the haloscope signatures (axion / dark-photon narrow lines) plus a
couple of broadband bumps for unmodeled excess. Scoring delegates to the shared
`loeb_turner_score` over the Gaussian (PSD) likelihood.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.magnetometric.haloscope import (
    ABRACADABRA,
    ADMX,
    SHAFT,
    HaloscopeBand,
)
from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.models.exotic import (
    ExoticTemplate,
    broadband_bump_template,
    psd_line_template,
)
from anomalymetric.score.loeb_turner import ScoreResult, loeb_turner_score
from anomalymetric.spectrum import Spectrum
from anomalymetric.units import H_PLANCK_EV_S


class SQUIDNoiseFloor:
    """Scalable SQUID flux-noise floor — the `AllParticleReference` analog."""

    def __init__(self, amplitude: float = 1.0, S_white: float = 1.0e-12, f_knee_hz: float = 1.0):
        self.name = "squid_noise_floor"
        self.S_white = S_white
        self.f_knee_hz = f_knee_hz
        self.parameters = Parameters(
            [Parameter("amplitude", amplitude, min=0.0, max=np.inf)]
        )

    def dnde(self, log_energy_centers_eV):
        from anomalymetric.magnetometric.spectrum import squid_reference_psd

        return self.parameters["amplitude"].value * squid_reference_psd(
            log_energy_centers_eV, S_white=self.S_white, f_knee_hz=self.f_knee_hz
        )


def axion_psd_line_template(label: str, m_a_eV: float, width_dex: float = 0.02) -> ExoticTemplate:
    """Axion haloscope line at the Compton energy (E_center = m_a in eV)."""
    return psd_line_template(f"axion.{label}", m_a_eV, width_dex)


def dark_photon_line_template(label: str, m_Aprime_eV: float, width_dex: float = 0.02) -> ExoticTemplate:
    """Dark-photon kinetic-mixing line at E_center = m_A' (eV)."""
    return psd_line_template(f"darkphoton.{label}", m_Aprime_eV, width_dex)


def squid_exotic_library(
    bands: Optional[list[HaloscopeBand]] = None, masses_per_band: int = 4
) -> list[ExoticTemplate]:
    """Axion + dark-photon line grids over the canned bands, plus broadband bumps."""
    if bands is None:
        bands = [ABRACADABRA, ADMX, SHAFT]
    templates: list[ExoticTemplate] = []
    for band in bands:
        for i, m in enumerate(band.mass_grid_eV(masses_per_band)):
            templates.append(axion_psd_line_template(f"{band.name}_{i}", float(m)))
    # A representative dark-photon line and two broadband bumps for unmodeled excess.
    templates.append(dark_photon_line_template("admx_mid", float(ADMX.mass_grid_eV(1)[0])))
    templates.append(broadband_bump_template("kHz", float(1.0e3 * H_PLANCK_EV_S), width_dex=0.3))
    templates.append(broadband_bump_template("MHz", float(1.0e6 * H_PLANCK_EV_S), width_dex=0.3))
    return templates


def squid_score(
    spectrum: Spectrum,
    *,
    forward: Optional[ForwardModel] = None,
) -> ScoreResult:
    """PLR vs the SQUID noise floor with the magnetometric exotic library."""
    return loeb_turner_score(
        spectrum,
        natural_components=[SQUIDNoiseFloor()],
        forward=forward,
        exotic_library=squid_exotic_library(),
    )
