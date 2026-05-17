"""CR-only PLR scorer: deviation from the all-particle reference spectrum.

Distinct from the unified `score.loeb_turner` flow because the natural
hypothesis here is *one* fixed reference (PDG all-particle) plus a free
overall normalization; the only exotic templates that make sense for CR are
the hard-cutoff and GZK-violating-tail entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.models.exotic import (
    ExoticTemplate,
    gzk_violating_template,
    hard_cutoff_template,
)
from anomalymetric.score.loeb_turner import ScoreResult, loeb_turner_score
from anomalymetric.spectrum import Spectrum


class AllParticleReference:
    """A scalable reference model wrapping `cr_reference_dnde`."""

    def __init__(self, amplitude: float = 1.0):
        self.name = "cr_reference"
        self.parameters = Parameters(
            [Parameter("amplitude", amplitude, min=0.0, max=np.inf)]
        )

    def dnde(self, log_energy_centers_eV):
        from anomalymetric.cosmicray.spectrum import cr_reference_dnde

        return self.parameters["amplitude"].value * cr_reference_dnde(log_energy_centers_eV)


def cr_exotic_library() -> list[ExoticTemplate]:
    return [
        hard_cutoff_template("CR_1EeV", 2.0, 1.0e18),
        hard_cutoff_template("CR_10EeV", 2.0, 1.0e19),
        gzk_violating_template("5e19"),
        gzk_violating_template("1e20"),
    ]


def cr_score(
    spectrum: Spectrum,
    *,
    forward: Optional[ForwardModel] = None,
) -> ScoreResult:
    """PLR vs PDG all-particle reference with the CR-specific exotic library."""
    return loeb_turner_score(
        spectrum,
        natural_components=[AllParticleReference()],
        forward=forward,
        exotic_library=cr_exotic_library(),
    )
