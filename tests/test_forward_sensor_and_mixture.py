"""Workstream C coverage: GradiometerResponse + Mixture.component_dnde."""

from __future__ import annotations

import numpy as np

from anomalymetric.forward.sensor import GradiometerResponse, PSDResponse
from anomalymetric.models.mixture import Mixture
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.units import log_frequency_grid


def test_psd_response_is_identity():
    edges = log_frequency_grid(0.0, 3.0, bins_per_decade=10)
    psd = np.linspace(1.0, 2.0, edges.shape[0] - 1)
    assert np.allclose(PSDResponse().forward(edges, psd), psd)


def test_gradiometer_high_passes():
    """The gradiometer transfer rejects low frequency and passes high frequency."""
    edges = log_frequency_grid(0.0, 9.0, bins_per_decade=10)  # 1 Hz .. 1 GHz
    psd = np.ones(edges.shape[0] - 1)
    out = GradiometerResponse(baseline_m=1.0, order=1).forward(edges, psd)
    assert np.all(out >= 0) and np.all(out <= psd + 1e-12)
    # Transfer is monotonically increasing with frequency (high-pass).
    assert out[-1] > out[0]


def test_mixture_component_dnde_sums_to_total():
    centers = np.linspace(0.0, 3.0, 12)
    a = BlackBody(T_K=300.0, amplitude=1.0)
    b = PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)
    b.name = "pl"  # avoid duplicate-name guard with the blackbody
    mix = Mixture([a, b])
    parts = mix.component_dnde(centers)
    assert set(parts.keys()) == {a.name, "pl"}
    assert np.allclose(parts[a.name] + parts["pl"], mix.dnde(centers))
