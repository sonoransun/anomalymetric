"""Workstream B2: BIC/Laplace Bayes-factor scoring (no extra deps)."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.score.bayes import bayes_factor
from anomalymetric.score.loeb_turner import loeb_turner_score
from anomalymetric.units import wavelength_nm_to_ev


def _natural():
    return [BlackBody(T_K=300.0, amplitude=1.0),
            PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]


def _base():
    return synthetic_natural(log_e_min=-2, log_e_max=4, bins_per_decade=40, T_K=300.0,
                             exposure_cm2_s=1e6, poisson=True, seed=17)


def _anomalous(base):
    line_eV = float(wavelength_nm_to_ev(532.0))
    return synthetic_with_exotic(base, line_E_eV=line_eV, line_amplitude=5e3,
                                 sigma_dex=0.005, seed=18)


def test_bic_ranks_anomaly_above_background():
    base = _base()
    anom = _anomalous(base)
    rb = bayes_factor(base, _natural())
    ra = bayes_factor(anom, _natural())
    assert ra.method == "bic"
    assert ra.log_bayes_factor > rb.log_bayes_factor
    # A clean background should not favor any exotic model (lnB < 0 ~ Occam wins).
    assert rb.log_bayes_factor < 1.0


def test_bic_best_template_agrees_with_plr():
    anom = _anomalous(_base())
    rb = bayes_factor(anom, _natural())
    plr = loeb_turner_score(anom, _natural())
    assert "laser" in rb.best_template
    assert rb.best_template == plr.best_template


def test_laplace_without_prior_falls_back_to_bic():
    res = bayes_factor(_base(), _natural(), method="laplace")
    assert res.method == "bic"
    assert "fell back to BIC" in res.notes


def test_laplace_with_prior_runs_and_detects():
    base = _base()
    anom = _anomalous(base)
    rb = bayes_factor(base, _natural(), method="laplace", amplitude_prior_width=1e4)
    ra = bayes_factor(anom, _natural(), method="laplace", amplitude_prior_width=1e4)
    assert ra.method == "laplace"
    assert np.isfinite(ra.log_bayes_factor)
    assert ra.log_bayes_factor > rb.log_bayes_factor
