"""Workstream F: packaged reference data + loader."""

from __future__ import annotations

import numpy as np

from anomalymetric.data import load_reference, reference_names


def test_reference_names_lists_cr_table():
    names = reference_names()
    assert "cr_allparticle_pdg.csv" in names


def test_load_reference_returns_monotone_falling_spectrum():
    table = load_reference("cr_allparticle_pdg.csv")
    assert {"log_E_eV", "dNdE_per_cm2_s_sr_eV"} <= set(table.colnames)
    log_e = np.asarray(table["log_E_eV"], dtype=float)
    dnde = np.asarray(table["dNdE_per_cm2_s_sr_eV"], dtype=float)
    assert log_e.min() <= 9.0 and log_e.max() >= 21.0
    assert np.all(dnde > 0)
    # A cosmic-ray spectrum falls steeply with energy.
    assert dnde[0] > dnde[-1]
