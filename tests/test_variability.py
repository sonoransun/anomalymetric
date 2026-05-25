"""Workstream B3: multi-epoch variability scoring over a SpectrumSeries."""

from __future__ import annotations

import numpy as np

from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
from anomalymetric.models.exotic import laser_line_template
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.score.variability import variability_score
from anomalymetric.spectrum import SpectrumSeries
from anomalymetric.units import wavelength_nm_to_ev

LINE_EV = float(wavelength_nm_to_ev(532.0))


def _natural():
    return [BlackBody(T_K=300.0, amplitude=1.0),
            PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]


def _library():
    # Small two-template library keeps the multi-epoch loop fast.
    return [laser_line_template("Nd_YAG_532", 532.0), laser_line_template("HeNe_633", 632.8)]


def _epoch(seed, with_line):
    base = synthetic_natural(log_e_min=-1, log_e_max=3, bins_per_decade=20,
                             exposure_cm2_s=1e6, seed=seed)
    if with_line:
        return synthetic_with_exotic(base, line_E_eV=LINE_EV, line_amplitude=4e3,
                                     sigma_dex=0.01, seed=seed + 100)
    return base


def _series(line_epochs):
    specs = [_epoch(seed=s, with_line=(s in line_epochs)) for s in range(3)]
    return SpectrumSeries(epochs_mjd=np.array([0.0, 1.0, 2.0]), spectra=specs)


def test_transient_series_outscores_clean():
    clean = variability_score(_series(line_epochs=set()), _natural, exotic_library=_library())
    transient = variability_score(_series(line_epochs={1}), _natural, exotic_library=_library())
    assert transient.global_test_statistic > clean.global_test_statistic
    assert transient.anomaly_score > clean.anomaly_score
    assert "laser.Nd_YAG_532" == transient.best_template
    assert len(transient.per_epoch) == 3


def test_flux_variability_detects_brightening():
    clean = variability_score(_series(line_epochs=set()), _natural, exotic_library=_library())
    transient = variability_score(_series(line_epochs={1}), _natural, exotic_library=_library())
    # The injected-line epoch carries extra counts -> larger fractional RMS.
    assert transient.flux_variability > clean.flux_variability
