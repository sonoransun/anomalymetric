"""Workstream E: optional-dependency features.

Pure helpers run always; dep-backed paths use importorskip so the suite stays
green whether or not the extra is installed. crpropa is expected to skip.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_to_dnde_per_eV_unit_conversion():
    """The naima flux->dN/dE converter is pure unit handling (astropy only)."""
    import astropy.units as u

    from anomalymetric.models.naima_templates import _to_dnde_per_eV

    E = np.array([1.0, 10.0]) * u.eV
    flux = np.array([2.0, 5.0]) / (u.eV * u.s * u.cm**2)
    assert np.allclose(_to_dnde_per_eV(flux, E), [2.0, 5.0])
    # An SED (E^2 dN/dE) input is divided by E^2 back to dN/dE.
    sed = (np.array([2.0, 5.0]) * E**2 / (u.eV * u.s * u.cm**2)).to(u.erg / u.s / u.cm**2)
    assert np.allclose(_to_dnde_per_eV(sed, E), [2.0, 5.0], rtol=1e-6)


def test_naima_templates_finite():
    pytest.importorskip("naima")
    from anomalymetric.models.naima_templates import naima_inverse_compton_template
    from anomalymetric.units import log_energy_grid

    centers = 0.5 * (lambda e: (e[:-1] + e[1:]))(log_energy_grid(9, 13, bins_per_decade=8))
    model = naima_inverse_compton_template().factory()
    dnde = model.dnde(centers)
    assert np.all(np.isfinite(dnde)) and np.any(dnde > 0)


@pytest.mark.slow
def test_dynesty_bayes_factor_runs():
    pytest.importorskip("dynesty")
    from anomalymetric.ingest.synthetic import synthetic_natural
    from anomalymetric.models.exotic import laser_line_template
    from anomalymetric.models.powerlaw import PowerLaw
    from anomalymetric.score.bayes import bayes_factor

    spec = synthetic_natural(log_e_min=-1, log_e_max=2, bins_per_decade=8,
                             exposure_cm2_s=1e6, seed=0)
    # Keep the evidence integrals low-dimensional: a single free amplitude per model
    # (frozen power-law shape) so nested sampling converges in seconds.
    pl = PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)
    pl.parameters["index"].frozen = True
    pl.parameters["reference_eV"].frozen = True
    res = bayes_factor(spec, [pl], exotic_library=[laser_line_template("Nd_YAG_532", 532.0)],
                       method="dynesty", nlive=60)
    assert res.method == "dynesty"
    assert np.isfinite(res.log_bayes_factor)


def test_crpropa_backtrack_available_or_skips():
    pytest.importorskip("crpropa")  # skips here (crpropa not installed)
    from anomalymetric.cosmicray.propagation import crpropa_backtrack

    res = crpropa_backtrack(1e20, 1, (0.0, 0.0))
    assert res.rigidity_EV > 0
