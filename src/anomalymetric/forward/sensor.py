"""Forward models for continuous field-sensor channels (SQUID, gravimeter).

For these channels a `Model.dnde(log_E)` is reinterpreted to return a *predicted
power spectral density* S(f) on the analysis grid (f = 10**log_E / h). PSD is
already a per-bin density, so the default forward is an identity pass-through —
unlike photon/CR forwards, there is no bin-width or exposure multiply.

`GradiometerResponse` folds in a differential (gradiometer) transfer |H(f)|^2,
the frequency-dependent common-mode rejection of a two-arm differential sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anomalymetric.units import bin_centers_eV, ev_to_hz


@dataclass
class PSDResponse:
    """Identity pass-through for PSD channels — predicted PSD is returned as-is."""

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_psd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.asarray(source_psd, dtype=float)


@dataclass
class GradiometerResponse:
    """Differential-sensor transfer: multiply predicted PSD by |H(f)|^2.

    A first/second-order gradiometer rejects common-mode signal at low frequency
    (long wavelength compared with the baseline) and passes it near and above
    f ~ c_wave / baseline. We model the transfer as a high-pass in frequency,
    |H(f)|^2 = (f / f_corner)^(2*order) / (1 + (f / f_corner)^(2*order)), with
    `f_corner` set by the baseline. Shape is locked; it is not a fit parameter.
    """

    baseline_m: float = 1.0
    order: int = 1
    wave_speed_m_s: float = 2.99792458e8  # default: EM; set ~5000 for seismic strain

    def _transfer(self, log_energy_edges_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        f = ev_to_hz(bin_centers_eV(log_energy_edges_eV))
        f_corner = self.wave_speed_m_s / max(self.baseline_m, 1e-30)
        x = (f / f_corner) ** (2 * self.order)
        return x / (1.0 + x)

    def forward(
        self,
        log_energy_edges_eV: NDArray[np.float64],
        source_psd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        edges = np.asarray(log_energy_edges_eV, dtype=float)
        return np.asarray(source_psd, dtype=float) * self._transfer(edges)
