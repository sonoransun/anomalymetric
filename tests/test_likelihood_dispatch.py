"""The likelihood entry point picks the right forward model per channel."""

from __future__ import annotations

import numpy as np

from anomalymetric.forward.exposure import FlatExposure
from anomalymetric.forward.response import IdentityResponse
from anomalymetric.models.inference import Fit, likelihood, _default_forward_for
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind


def _photon_counts_spec():
    edges = np.linspace(-1, 3, 11)
    counts = np.full(10, 100.0)
    return Spectrum(
        log_energy_edges_eV=edges,
        value=counts,
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=SpectrumKind.PHOTON,
        exposure_cm2_s=np.ones(10),
    )


def _cr_counts_spec():
    edges = np.linspace(15, 21, 13)
    counts = np.full(12, 5.0)
    return Spectrum(
        log_energy_edges_eV=edges,
        value=counts,
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=SpectrumKind.CR_ALLPARTICLE,
        per_steradian=True,
        exposure_cm2_s=np.ones(12),
    )


def test_default_forward_photon_is_identity():
    spec = _photon_counts_spec()
    fwd = _default_forward_for(spec)
    assert isinstance(fwd, IdentityResponse)


def test_default_forward_cr_is_flat_exposure():
    spec = _cr_counts_spec()
    fwd = _default_forward_for(spec)
    assert isinstance(fwd, FlatExposure)


def test_likelihood_evaluates_finite_for_both_channels():
    photon = _photon_counts_spec()
    cr = _cr_counts_spec()
    photon_model = BlackBody(T_K=300.0, amplitude=1e-5)
    cr_model = PowerLaw(amplitude=1e-15, index=2.7, reference_eV=1e15)
    assert np.isfinite(likelihood(photon, photon_model))
    assert np.isfinite(likelihood(cr, cr_model))


def test_fit_runs_without_error_on_photon():
    spec = _photon_counts_spec()
    bb = BlackBody(T_K=300.0, amplitude=1.0)
    res = Fit(bb, spec).run()
    assert res.success or res.log_likelihood > -np.inf
