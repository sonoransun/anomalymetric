"""Unit conversions round-trip and Spectrum invariants hold."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
from anomalymetric.units import (
    bin_centers_eV,
    bin_widths_eV,
    ev_to_hz,
    hz_to_ev,
    log_energy_grid,
    wavelength_nm_to_ev,
)


def test_hz_eV_roundtrip():
    nu = np.logspace(1, 25, 50)
    assert np.allclose(ev_to_hz(hz_to_ev(nu)), nu, rtol=1e-12)


def test_laser_wavelength_to_eV():
    # 532 nm laser is ~2.331 eV
    assert wavelength_nm_to_ev(532.0) == pytest.approx(2.331, rel=1e-3)


def test_log_energy_grid_widths():
    edges = log_energy_grid(-2, 6, bins_per_decade=10)
    widths = bin_widths_eV(edges)
    centers = bin_centers_eV(edges)
    assert widths.shape[0] == edges.shape[0] - 1
    assert centers.shape[0] == widths.shape[0]
    assert np.all(widths > 0)


def test_spectrum_shape_validation():
    edges = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        Spectrum(
            log_energy_edges_eV=edges,
            value=np.ones(3),
            value_kind=ValueKind.DNDE,
        )


def test_value_kind_canonicalization_to_dnde():
    edges = np.linspace(0, 6, 7)
    centers = bin_centers_eV(edges)
    # dN/dE = 1 / E (just a test value).
    dnde_true = 1.0 / centers
    spec_dnde = Spectrum(
        log_energy_edges_eV=edges,
        value=dnde_true,
        value_kind=ValueKind.DNDE,
    )
    assert np.allclose(spec_dnde.as_dnde(), dnde_true)

    # E dN/dE -> dN/dE
    spec_edn = Spectrum(
        log_energy_edges_eV=edges,
        value=centers * dnde_true,
        value_kind=ValueKind.EDNDE,
    )
    assert np.allclose(spec_edn.as_dnde(), dnde_true)

    # E^2 dN/dE -> dN/dE
    spec_e2 = Spectrum(
        log_energy_edges_eV=edges,
        value=centers**2 * dnde_true,
        value_kind=ValueKind.E2DNDE,
    )
    assert np.allclose(spec_e2.as_dnde(), dnde_true)


def test_counts_per_bin_requires_exposure():
    edges = np.linspace(0, 6, 7)
    spec = Spectrum(
        log_energy_edges_eV=edges,
        value=np.full(6, 10.0),
        value_kind=ValueKind.COUNTS_PER_BIN,
    )
    with pytest.raises(ValueError):
        spec.as_dnde()
    spec.exposure_cm2_s = np.ones(6) * 100.0
    dnde = spec.as_dnde()
    assert dnde.shape == (6,)


def test_from_dnde_kind_default_photon():
    edges = np.linspace(0, 6, 7)
    spec = Spectrum.from_dnde(edges, np.ones(6))
    assert spec.kind is SpectrumKind.PHOTON


def test_cosmic_ray_kind():
    edges = np.linspace(9, 21, 13)
    spec = Spectrum(
        log_energy_edges_eV=edges,
        value=np.zeros(12),
        value_kind=ValueKind.COUNTS_PER_BIN,
        kind=SpectrumKind.CR_ALLPARTICLE,
        per_steradian=True,
        exposure_cm2_s=np.ones(12),
    )
    assert spec.kind is SpectrumKind.CR_ALLPARTICLE
