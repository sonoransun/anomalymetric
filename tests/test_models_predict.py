"""Smoke tests: every model produces finite, non-negative dN/dE."""

from __future__ import annotations

import numpy as np

from anomalymetric.models.exotic import (
    GZKViolatingTail,
    HardCutoffPowerLaw,
    default_library,
)
from anomalymetric.models.lines import GaussianLine
from anomalymetric.models.mixture import Mixture
from anomalymetric.models.powerlaw import BrokenPowerLaw, PowerLaw
from anomalymetric.models.reflective import SolarReflection
from anomalymetric.models.thermal import BlackBody
from anomalymetric.cosmicray.knee_ankle import TripleBrokenPowerLaw


def _finite_nonneg(arr):
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0)


def test_blackbody_smoke():
    log_E = np.linspace(-2, 4, 100)
    out = BlackBody(T_K=300.0).dnde(log_E)
    _finite_nonneg(out)
    assert out.max() > 0


def test_solar_reflection_smoke():
    log_E = np.linspace(-1, 3, 100)
    out = SolarReflection().dnde(log_E)
    _finite_nonneg(out)


def test_powerlaw_smoke():
    log_E = np.linspace(0, 12, 100)
    out = PowerLaw(amplitude=1.0, index=2.0, reference_eV=1.0).dnde(log_E)
    _finite_nonneg(out)


def test_broken_powerlaw_smoke():
    log_E = np.linspace(9, 21, 200)
    out = BrokenPowerLaw().dnde(log_E)
    _finite_nonneg(out)


def test_gaussian_line_smoke():
    log_E = np.linspace(-1, 5, 1000)
    out = GaussianLine(amplitude=1.0, E_center_eV=2.33, sigma_dex=0.01).dnde(log_E)
    _finite_nonneg(out)
    # Peak should be near 2.33 eV.
    peak_log = log_E[np.argmax(out)]
    assert abs(peak_log - np.log10(2.33)) < 0.02


def test_mixture_concatenates_parameters():
    bb = BlackBody()
    pl = PowerLaw(reference_eV=1.0)
    mix = Mixture([bb, pl])
    names = {p.name for p in mix.parameters}
    assert "blackbody.T_K" in names
    assert "powerlaw.index" in names


def test_mixture_dnde_is_sum():
    log_E = np.linspace(-2, 4, 50)
    bb = BlackBody()
    pl = PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)
    mix = Mixture([bb, pl])
    assert np.allclose(mix.dnde(log_E), bb.dnde(log_E) + pl.dnde(log_E))


def test_hard_cutoff_smoke():
    log_E = np.linspace(0, 17, 100)
    out = HardCutoffPowerLaw().dnde(log_E)
    _finite_nonneg(out)


def test_gzk_tail_zero_below_threshold():
    log_E = np.array([18.0, 19.0, 20.0])
    out = GZKViolatingTail().dnde(log_E)
    # Below 5e19 eV the tail is exactly zero.
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_default_library_factories_work():
    for tpl in default_library():
        m = tpl.factory()
        out = m.dnde(np.linspace(-2, 21, 50))
        assert np.all(np.isfinite(out))


def test_triple_broken_pl_smoke():
    log_E = np.linspace(9, 21, 100)
    out = TripleBrokenPowerLaw().dnde(log_E)
    _finite_nonneg(out)
