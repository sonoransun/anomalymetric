"""An injected exotic line should rank above a clean background."""

from __future__ import annotations

import numpy as np

from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.score.loeb_turner import loeb_turner_score
from anomalymetric.units import wavelength_nm_to_ev


def _natural():
    return [
        BlackBody(T_K=300.0, amplitude=1.0),
        PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0),
    ]


def test_injected_laser_line_outranks_background():
    rng_seed = 17
    base = synthetic_natural(
        log_e_min=-2,
        log_e_max=4,
        bins_per_decade=40,
        T_K=300.0,
        bb_amplitude=1.0,
        pl_amplitude=1e-3,
        pl_index=2.0,
        exposure_cm2_s=1e6,
        poisson=True,
        seed=rng_seed,
    )
    # Inject a 532 nm laser line (~2.331 eV) with a large amplitude so the
    # detection is unambiguous.
    line_eV = float(wavelength_nm_to_ev(532.0))
    anomalous = synthetic_with_exotic(
        base, line_E_eV=line_eV, line_amplitude=5e3, sigma_dex=0.005, seed=rng_seed + 1
    )

    bg = loeb_turner_score(base, _natural())
    anom = loeb_turner_score(anomalous, _natural())

    assert anom.test_statistic > bg.test_statistic
    assert anom.anomaly_score > bg.anomaly_score
    assert "laser" in anom.best_template
