"""A Gaussian-channel fit recovers an injected noise-floor amplitude."""

from __future__ import annotations

import numpy as np

from anomalymetric.gravitational.models import GravimeterNoise
from anomalymetric.gravitational.spectrum import (
    gravimeter_band_grid,
    grav_reference_psd,
    make_grav_spectrum,
)
from anomalymetric.magnetometric.models import DCSQUIDNoise
from anomalymetric.magnetometric.spectrum import (
    make_squid_spectrum,
    squid_band_grid,
    squid_reference_psd,
)
from anomalymetric.models.exotic import BroadbandBump
from anomalymetric.models.inference import Fit
from anomalymetric.models.mixture import Mixture
from anomalymetric.units import bin_centers_eV


def test_fit_recovers_squid_white_floor():
    edges = squid_band_grid(2.0, 9.0, bins_per_decade=20)
    lc = np.log10(bin_centers_eV(edges))
    true_amp = 3.0
    ref = true_amp * squid_reference_psd(lc)
    rng = np.random.default_rng(0)
    sigma = 0.02 * ref
    spec = make_squid_spectrum(edges, rng.normal(ref, sigma), sigma)
    res = Fit(DCSQUIDNoise(amplitude=1.0), spec).run()
    assert res.parameter_values["amplitude"] == np.float64(res.parameter_values["amplitude"])
    assert abs(res.parameter_values["amplitude"] - true_amp) / true_amp < 0.1


def test_fit_recovers_grav_amplitude():
    edges = gravimeter_band_grid(-5.0, 1.0, bins_per_decade=20)
    lc = np.log10(bin_centers_eV(edges))
    true_amp = 2.0
    ref = true_amp * grav_reference_psd(lc)
    rng = np.random.default_rng(1)
    sigma = 0.02 * ref
    spec = make_grav_spectrum(edges, rng.normal(ref, sigma), sigma)
    res = Fit(GravimeterNoise(amplitude=1.0), spec).run()
    assert abs(res.parameter_values["amplitude"] - true_amp) / true_amp < 0.1


def test_psd_mixture_fit_runs():
    edges = squid_band_grid(2.0, 9.0, bins_per_decade=20)
    lc = np.log10(bin_centers_eV(edges))
    ref = squid_reference_psd(lc)
    rng = np.random.default_rng(2)
    sigma = 0.05 * ref
    spec = make_squid_spectrum(edges, rng.normal(ref, sigma), sigma)
    bump = BroadbandBump(amplitude=0.0, E_center_eV=float(bin_centers_eV(edges)[len(lc) // 2]), width_dex=0.3)
    mix = Mixture([DCSQUIDNoise(amplitude=1.0), bump], name="floor+bump")
    res = Fit(mix, spec).run()
    assert np.isfinite(res.log_likelihood)
