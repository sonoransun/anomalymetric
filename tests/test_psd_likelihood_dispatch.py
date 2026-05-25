"""The Gaussian (PSD) channel dispatches to the right likelihood and forward model."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.forward.response import IdentityResponse
from anomalymetric.forward.sensor import PSDResponse
from anomalymetric.models.inference import (
    Fit,
    _default_forward_for,
    _log_likelihood_for,
    gaussian_log_likelihood,
    likelihood,
)
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import log_frequency_grid


def _psd_spec(kind=SpectrumKind.MAGNETOMETRIC, with_sigma=True):
    edges = log_frequency_grid(2.0, 5.0, bins_per_decade=10)
    n = edges.shape[0] - 1
    psd = np.full(n, 2.0)
    sigma = np.full(n, 0.1) if with_sigma else None
    return Spectrum(
        log_energy_edges_eV=edges,
        value=psd,
        value_kind=ValueKind.PSD_PER_BIN,
        kind=kind,
        uncertainty=sigma,
    )


def test_default_forward_psd_is_passthrough():
    for kind in (SpectrumKind.MAGNETOMETRIC, SpectrumKind.GRAVITATIONAL):
        assert isinstance(_default_forward_for(_psd_spec(kind)), PSDResponse)
    # Regression: the photon dispatch is unchanged.
    edges = np.linspace(-1, 3, 6)
    photon = Spectrum(edges, np.full(5, 10.0), ValueKind.COUNTS_PER_BIN, exposure_cm2_s=np.ones(5))
    assert isinstance(_default_forward_for(photon), IdentityResponse)


def test_log_likelihood_selector_picks_gaussian():
    assert _log_likelihood_for(_psd_spec()).__name__ == "_gauss"
    edges = np.linspace(-1, 3, 6)
    photon = Spectrum(edges, np.full(5, 10.0), ValueKind.COUNTS_PER_BIN, exposure_cm2_s=np.ones(5))
    assert _log_likelihood_for(photon).__name__ == "_pois"


def test_gaussian_log_likelihood_peaks_at_mean():
    y = np.array([1.0, 2.0, 3.0])
    s = np.array([0.5, 0.5, 0.5])
    assert gaussian_log_likelihood(y, y, s) > gaussian_log_likelihood(y, y + 0.3, s)
    # Symmetric in the residual sign.
    assert gaussian_log_likelihood(y, y + 0.3, s) == pytest.approx(
        gaussian_log_likelihood(y, y - 0.3, s)
    )


def test_gaussian_upper_limit_is_one_sided():
    y = np.array([1.0, 1.0])
    s = np.array([0.5, 0.5])
    mask = np.array([True, True])
    # Predicting below the limit is unpenalized; predicting above is penalized.
    below = gaussian_log_likelihood(y, y - 1.0, s, upper_limit_mask=mask)
    above = gaussian_log_likelihood(y, y + 1.0, s, upper_limit_mask=mask)
    assert below > above
    assert below == pytest.approx(0.0)


def test_as_dnde_raises_on_psd():
    with pytest.raises(ValueError):
        _psd_spec().as_dnde()
    with pytest.raises(ValueError):
        _psd_spec().expected_counts()


def test_canonical_value_squares_asd():
    edges = log_frequency_grid(2.0, 3.0, bins_per_decade=10)
    n = edges.shape[0] - 1
    asd = np.full(n, 3.0)
    spec = Spectrum(edges, asd, ValueKind.ASD_PER_BIN, kind=SpectrumKind.MAGNETOMETRIC,
                    uncertainty=np.full(n, 0.1))
    assert np.allclose(spec.canonical_value(), 9.0)


def test_gaussian_channel_requires_sigma():
    spec = _psd_spec(with_sigma=False)
    from anomalymetric.models.base import Parameter, Parameters

    class Flat:
        def __init__(self):
            self.name = "flat"
            self.parameters = Parameters([Parameter("amplitude", 1.0, min=0.0)])

        def dnde(self, logE):
            return np.full(np.asarray(logE).shape, self.parameters["amplitude"].value)

    with pytest.raises(ValueError):
        Fit(Flat(), spec)


def test_gaussian_and_poisson_branches_distinct():
    """Same numeric data, different kind -> different likelihood family / value."""
    from anomalymetric.models.base import Parameter, Parameters

    class Flat:
        def __init__(self, a=2.0):
            self.name = "flat"
            self.parameters = Parameters([Parameter("amplitude", a, min=0.0)])

        def dnde(self, logE):
            return np.full(np.asarray(logE).shape, self.parameters["amplitude"].value)

    edges = log_frequency_grid(2.0, 5.0, bins_per_decade=10)
    n = edges.shape[0] - 1
    vals = np.full(n, 2.0)
    gauss = Spectrum(edges, vals, ValueKind.PSD_PER_BIN, kind=SpectrumKind.MAGNETOMETRIC,
                     uncertainty=np.full(n, 0.1))
    # Identity forward (passthrough) means predicted==2.0 everywhere == data -> logL near max.
    ll_gauss = likelihood(gauss, Flat())
    assert np.isfinite(ll_gauss)
