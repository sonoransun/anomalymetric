"""Forward-model protocol.

A `ForwardModel` maps a *source* quantity on the analysis grid to the predicted
*detector* observable per bin. The source quantity is dN/dE for Poisson channels
(photons, cosmic rays) — folded to expected counts via bin width * exposure,
energy dispersion (RMF), effective area (ARF), PSF leakage — and a predicted
power spectral density S(f) for continuous Gaussian field-sensor channels
(magnetometer/SQUID, gravimeter), where the forward is an identity pass-through
or a sensor transfer (see `forward.sensor`).
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
