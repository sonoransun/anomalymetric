"""Workstream A: input validation, mixture name-collision, optimizer scale-invariance."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anomalymetric.ingest.synthetic import synthetic_natural
from anomalymetric.models.inference import Fit
from anomalymetric.models.mixture import Mixture
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind


# --- Spectrum.__post_init__ validation ---------------------------------------

def test_rejects_non_monotonic_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum(np.array([0.0, 2.0, 1.0]), np.array([1.0, 1.0]), ValueKind.DNDE)


def test_rejects_nonfinite_edges():
    with pytest.raises(ValueError, match="finite"):
        Spectrum(np.array([0.0, np.inf]), np.array([1.0]), ValueKind.DNDE)


def test_rejects_nonfinite_value():
    with pytest.raises(ValueError, match="finite"):
        Spectrum(np.array([0.0, 1.0]), np.array([np.nan]), ValueKind.DNDE)


def test_expected_counts_rejects_wrong_exposure_shape():
    spec = Spectrum(np.linspace(0, 1, 4), np.full(3, 1.0), ValueKind.DNDE)
    with pytest.raises(ValueError, match="must match value shape"):
        spec.expected_counts(exposure_cm2_s=np.ones(2))


def test_canonical_value_rejects_negative_asd():
    spec = Spectrum(
        np.linspace(-1, 1, 4), np.array([1.0, -2.0, 1.0]), ValueKind.ASD_PER_BIN,
        kind=SpectrumKind.MAGNETOMETRIC, uncertainty=np.ones(3),
    )
    with pytest.raises(ValueError, match="non-negative"):
        spec.canonical_value()


# --- Mixture name-collision ---------------------------------------------------

def test_mixture_rejects_duplicate_component_names():
    a = PowerLaw(amplitude=1.0, index=2.0, reference_eV=1.0)
    b = PowerLaw(amplitude=1.0, index=2.0, reference_eV=1.0)  # same .name
    with pytest.raises(ValueError, match="unique"):
        Mixture([a, b])


# --- Optimizer scale-invariance ----------------------------------------------

def test_scale_is_a_pure_reparameterization():
    """Setting Parameter.scale must not change the fit result, only conditioning."""
    spec = synthetic_natural(seed=3)

    pl1 = PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)
    res1 = Fit(copy.deepcopy(pl1), spec).run()

    pl2 = PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)
    pl2.parameters["amplitude"].scale = 1e6  # wildly different conditioning
    res2 = Fit(pl2, spec).run()

    assert res2.log_likelihood == pytest.approx(res1.log_likelihood, rel=1e-4, abs=1e-3)
    assert res2.parameter_values["amplitude"] == pytest.approx(
        res1.parameter_values["amplitude"], rel=1e-3
    )
