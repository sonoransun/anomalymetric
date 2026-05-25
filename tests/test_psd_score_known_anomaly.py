"""Injected sensor-channel signals score as anomalous; pure noise does not."""

from __future__ import annotations

import numpy as np

from anomalymetric.gravitational.fifth_force import (
    microscope_modulation_freq_hz,
    oscillating_dm_freq_hz,
)
from anomalymetric.gravitational.score import grav_score
from anomalymetric.ingest.synthetic_psd import (
    synthetic_grav_natural,
    synthetic_grav_with_exotic,
    synthetic_squid_natural,
    synthetic_squid_with_exotic,
)
from anomalymetric.magnetometric.haloscope import ADMX
from anomalymetric.magnetometric.score import squid_score
from anomalymetric.units import H_PLANCK_EV_S


def test_injected_axion_line_outranks_noise():
    bg = squid_score(synthetic_squid_natural(seed=0))
    base = synthetic_squid_natural(seed=1, noise=False)
    m = float(ADMX.mass_grid_eV(4)[2])
    anom = squid_score(synthetic_squid_with_exotic(base, line_E_eV=m, line_amplitude=8.0, seed=2))
    assert anom.test_statistic > bg.test_statistic
    assert anom.anomaly_score > bg.anomaly_score
    assert "axion" in anom.best_template


def test_injected_fifth_force_recovered():
    base = synthetic_grav_natural(seed=3, noise=False)
    f_mod = 1.0e-2  # source-modulation frequency matching a yukawa template
    anom = synthetic_grav_with_exotic(base, line_E_eV=H_PLANCK_EV_S * f_mod, line_amplitude=8.0, seed=4)
    res = grav_score(anom)
    assert res.test_statistic > 0.0
    assert "yukawa" in res.best_template


def test_injected_ep_violation_recovered():
    base = synthetic_grav_natural(seed=5, noise=False)
    f_ep = microscope_modulation_freq_hz()
    anom = synthetic_grav_with_exotic(base, line_E_eV=H_PLANCK_EV_S * f_ep, line_amplitude=8.0, seed=6)
    res = grav_score(anom)
    assert "ep" in res.best_template
    assert res.anomaly_score > 1.0


def test_injected_oscillating_dm_recovered():
    base = synthetic_grav_natural(seed=7, noise=False)
    f = oscillating_dm_freq_hz(1.0e-15)
    anom = synthetic_grav_with_exotic(base, line_E_eV=H_PLANCK_EV_S * f, line_amplitude=8.0, seed=8)
    res = grav_score(anom)
    assert "oscdm" in res.best_template


def test_pure_noise_scores_low():
    # A pure-noise draw should not produce a strong detection.
    for seed in range(4):
        res = squid_score(synthetic_squid_natural(seed=seed))
        assert res.anomaly_score < 5.0
        gres = grav_score(synthetic_grav_natural(seed=seed))
        assert gres.anomaly_score < 5.0
