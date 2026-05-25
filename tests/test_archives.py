"""Workstream E: archive loaders. The pure _canonicalize step is offline-tested;
the live query is gated behind --run-remote (see conftest)."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.ingest.tabular import _table_to_spectrum
from anomalymetric.spectrum import SpectrumKind, ValueKind

astropy_table = pytest.importorskip("astropy.table")
Table = astropy_table.Table


def test_canonicalize_converts_frequency_and_sorts():
    from anomalymetric.ingest.archives import _canonicalize

    # Frequencies out of order, in Hz; canonicalize -> ascending log_energy_eV in eV.
    raw = Table({"nu": [3e9, 1e9, 2e9], "flux": [3.0, 1.0, 2.0], "err": [0.3, 0.1, 0.2]})
    canon = _canonicalize(
        raw, energy_column="nu", value_column="flux", energy_unit="Hz",
        value_kind="dNdE", kind="photon", uncertainty_column="err",
    )
    log_e = np.asarray(canon["log_energy_eV"], dtype=float)
    assert np.all(np.diff(log_e) > 0)  # sorted ascending
    # 1 GHz photon energy = h*nu ~ 4.14e-6 eV -> log10 ~ -5.38.
    assert np.isclose(log_e[0], np.log10(4.135667696e-15 * 1e9), atol=1e-6)
    assert list(np.asarray(canon["value"])) == [1.0, 2.0, 3.0]  # reordered with energy

    spec = _table_to_spectrum(canon)
    assert spec.kind is SpectrumKind.PHOTON
    assert spec.value_kind is ValueKind.DNDE
    assert spec.uncertainty is not None


def test_canonicalize_requires_two_points():
    from anomalymetric.ingest.archives import _canonicalize

    raw = Table({"nu": [1e9], "flux": [1.0]})
    with pytest.raises(ValueError):
        _canonicalize(raw, energy_column="nu", value_column="flux", energy_unit="Hz")


@pytest.mark.remote_data
def test_vizier_live_query():
    from anomalymetric.ingest.archives import VizierLoader

    # A real catalog id + column mapping; only runs with --run-remote.
    spec = VizierLoader().load(
        "J/ApJ/700/597", energy_column="nu", value_column="Fnu", energy_unit="Hz"
    )
    assert spec.n_bins > 1
