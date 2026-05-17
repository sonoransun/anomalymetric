"""Forward-model protocol.

A `ForwardModel` maps a *source* differential flux dN/dE on the analysis grid
to expected counts per bin on the *detector* grid. For an ideal detector that
is just multiplication by bin width * exposure; real detectors fold in energy
dispersion (RMF), effective area (ARF), and PSF leakage.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ForwardModel(Protocol):
    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_dnde: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts per bin on the analysis grid."""
        ...
