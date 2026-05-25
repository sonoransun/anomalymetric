"""Gravitational-differential PLR scorer: deviation from the noise floor.

Mirrors `cosmicray.score` / `magnetometric.score`: a single fixed-shape noise
floor with a free amplitude is the natural hypothesis; the exotic library holds
fifth-force (Yukawa), equivalence-principle-violation, and oscillating-dark-matter
lines, plus broadband bumps. Scoring delegates to the shared Gaussian-PSD
`loeb_turner_score`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.gravitational.fifth_force import (
    microscope_modulation_freq_hz,
    oscillating_dm_freq_hz,
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


class GravNoiseFloor:
    """Scalable differential-acceleration noise floor — the baseline hypothesis."""

    def __init__(self, amplitude: float = 1.0):
        self.name = "grav_noise_floor"
        self.parameters = Parameters(
            [Parameter("amplitude", amplitude, min=0.0, max=np.inf)]
        )

    def dnde(self, log_energy_centers_eV):
        from anomalymetric.gravitational.spectrum import grav_reference_psd

        return self.parameters["amplitude"].value * grav_reference_psd(log_energy_centers_eV)


def _line_at_freq(prefix: str, label: str, freq_hz: float, width_dex: float = 0.02) -> ExoticTemplate:
    """A narrow PSD line at frequency `freq_hz` (E_center = h*nu); amplitude free."""
    return psd_line_template(f"{prefix}.{label}", H_PLANCK_EV_S * float(freq_hz), width_dex)


def yukawa_template(label: str, mod_freq_hz: float) -> ExoticTemplate:
    """Fifth-force line at a source-modulation frequency.

    A lab/torsion-balance Yukawa search modulates a source mass at `mod_freq_hz`
    and looks for the differential force there; the Yukawa range sets the coupling
    strength (the free amplitude), not the frequency. Use `yukawa_range_to_freq_hz`
    only for the distinct ultralight-mediator-as-dark-matter interpretation.
    """
    return _line_at_freq("yukawa", label, mod_freq_hz)


def ep_violation_template(label: str, mod_freq_hz: float) -> ExoticTemplate:
    """Equivalence-principle-violation line at the EP modulation frequency."""
    return _line_at_freq("ep", label, mod_freq_hz)


def oscillating_dm_template(label: str, m_phi_eV: float) -> ExoticTemplate:
    """Scalar-dark-matter line at the field's Compton frequency."""
    return _line_at_freq("oscdm", label, oscillating_dm_freq_hz(m_phi_eV))


def grav_exotic_library() -> list[ExoticTemplate]:
    """Yukawa + EP-violation + oscillating-DM lines plus broadband bumps."""
    templates: list[ExoticTemplate] = []
    # Fifth-force lines at candidate source-modulation frequencies (in-band).
    for f_mod in (1.0e-3, 1.0e-2, 1.0e-1):
        templates.append(yukawa_template(f"{f_mod:g}Hz", f_mod))
    # EP-violation line at the MICROSCOPE modulation frequency.
    templates.append(ep_violation_template("microscope", microscope_modulation_freq_hz()))
    # Oscillating scalar-DM lines across an ultralight mass grid.
    for m in (1.0e-17, 1.0e-15, 1.0e-13):
        templates.append(oscillating_dm_template(f"{m:g}eV", m))
    # Broadband bumps for unmodeled excess (mHz and Hz).
    templates.append(broadband_bump_template("mHz", float(1.0e-3 * H_PLANCK_EV_S), width_dex=0.3))
    templates.append(broadband_bump_template("Hz", float(1.0 * H_PLANCK_EV_S), width_dex=0.3))
    return templates


def grav_score(
    spectrum: Spectrum,
    *,
    forward: Optional[ForwardModel] = None,
) -> ScoreResult:
    """PLR vs the gravitational noise floor with the differential exotic library."""
    return loeb_turner_score(
        spectrum,
        natural_components=[GravNoiseFloor()],
        forward=forward,
        exotic_library=grav_exotic_library(),
    )
