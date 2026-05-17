"""Synthetic → fit → check parameters within a few sigma."""

from __future__ import annotations

import numpy as np

from anomalymetric.ingest.synthetic import synthetic_natural
from anomalymetric.models.inference import Fit
from anomalymetric.models.mixture import Mixture
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody


def test_fit_recovers_blackbody_temperature():
    T_true = 300.0
    spec = synthetic_natural(
        log_e_min=-2,
        log_e_max=2,
        bins_per_decade=40,
        T_K=T_true,
        bb_amplitude=1.0,
        pl_amplitude=1e-8,  # essentially no power-law background
        exposure_cm2_s=1e10,  # high statistics
        poisson=True,
        seed=42,
    )
    bb = BlackBody(T_K=200.0, amplitude=0.5)
    res = Fit(bb, spec).run()
    assert res.success
    T_fit = res.parameter_values["T_K"]
    assert abs(T_fit - T_true) / T_true < 0.10  # within 10%


def test_fit_recovers_powerlaw_index():
    alpha_true = 2.0
    spec = synthetic_natural(
        log_e_min=2,
        log_e_max=8,
        bins_per_decade=20,
        T_K=10.0,  # blackbody negligible at these energies
        bb_amplitude=1e-30,
        pl_amplitude=1.0,
        pl_index=alpha_true,
        exposure_cm2_s=1e6,
        poisson=True,
        seed=11,
    )
    pl = PowerLaw(amplitude=0.5, index=2.5, reference_eV=1.0)
    res = Fit(pl, spec).run()
    assert res.success
    alpha_fit = res.parameter_values["index"]
    assert abs(alpha_fit - alpha_true) < 0.1


def test_mixture_fit_runs():
    spec = synthetic_natural(
        log_e_min=-2,
        log_e_max=4,
        bins_per_decade=20,
        T_K=300.0,
        bb_amplitude=1.0,
        pl_amplitude=1e-2,
        pl_index=2.0,
        exposure_cm2_s=1e6,
        poisson=True,
        seed=7,
    )
    mix = Mixture(
        [
            BlackBody(T_K=250.0, amplitude=0.8),
            PowerLaw(amplitude=5e-3, index=2.2, reference_eV=1.0),
        ]
    )
    res = Fit(mix, spec).run()
    assert res.log_likelihood > -np.inf
