"""astroquery-backed loaders (requires the `[archives]` extra).

These are thin convenience wrappers — the user is expected to know the schema of
the target archive and pass a column mapping. The load-bearing logic is the pure
`_canonicalize` step (query result -> canonical `astropy.table.Table`), which is
then handed to the existing `tabular._table_to_spectrum`. Live queries hit the
network; `_canonicalize` is fully offline-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from anomalymetric.spectrum import Spectrum


def _require_astroquery() -> None:
    try:
        import astroquery  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "ingest.archives requires the `[archives]` extra "
            "(pip install anomalymetric[archives])."
        ) from exc


def _canonicalize(
    raw_table,
    *,
    energy_column: str,
    value_column: str,
    energy_unit: str = "eV",
    value_kind: str = "dNdE",
    kind: str = "photon",
    uncertainty_column: Optional[str] = None,
    upper_limit_column: Optional[str] = None,
):
    """Map an archive table to the canonical schema `_table_to_spectrum` expects.

    The energy column is converted to eV via astropy's spectral equivalencies (so
    Hz / nm / keV / erg all work) and emitted as a single `log_energy_eV` centers
    column (edges are reconstructed downstream). Rows are sorted ascending in
    energy. `value_kind`/`kind` are broadcast as scalar string columns.
    """
    import astropy.units as u
    from astropy.table import Table

    energy = np.asarray(raw_table[energy_column], dtype=float) * u.Unit(energy_unit)
    energy_eV = energy.to(u.eV, equivalencies=u.spectral()).value
    n = energy_eV.shape[0]
    if n < 2:
        raise ValueError("archive table needs at least 2 energy points")
    cols: dict[str, Any] = {
        "log_energy_eV": np.log10(energy_eV),
        "value": np.asarray(raw_table[value_column], dtype=float),
        "value_kind": np.array([value_kind] * n),
        "kind": np.array([kind] * n),
    }
    if uncertainty_column is not None:
        cols["uncertainty"] = np.asarray(raw_table[uncertainty_column], dtype=float)
    if upper_limit_column is not None:
        cols["upper_limit_mask"] = np.asarray(raw_table[upper_limit_column], dtype=bool)
    order = np.argsort(cols["log_energy_eV"])
    return Table({key: np.asarray(val)[order] for key, val in cols.items()})


@dataclass
class VizierLoader:
    """Load a VizieR catalog row-set into a Spectrum.

    `source` is a VizieR catalog identifier; the caller supplies the column
    mapping (which column is energy, which is the flux value, units, kind).
    """

    name: str = "vizier"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        _require_astroquery()
        from astroquery.vizier import Vizier

        from anomalymetric.ingest.tabular import _table_to_spectrum

        try:
            catalogs = Vizier(columns=["**"]).get_catalogs(source)
        except Exception as exc:  # pragma: no cover - network
            raise RuntimeError(f"VizieR query for {source!r} failed: {exc}") from exc
        if len(catalogs) == 0:
            raise RuntimeError(f"VizieR returned no catalogs for {source!r}")
        return _table_to_spectrum(_canonicalize(catalogs[0], **kwargs))


@dataclass
class HEASARCLoader:
    """Load a HEASARC mission table into a Spectrum (caller supplies the mapping)."""

    name: str = "heasarc"

    def load(self, source: str, *, mission: str = "", **kwargs: Any) -> Spectrum:
        _require_astroquery()
        from astroquery.heasarc import Heasarc

        from anomalymetric.ingest.tabular import _table_to_spectrum

        try:
            table = Heasarc().query_object(source, mission=mission)
        except Exception as exc:  # pragma: no cover - network
            raise RuntimeError(
                f"HEASARC query for {source!r} (mission={mission!r}) failed: {exc}"
            ) from exc
        return _table_to_spectrum(_canonicalize(table, **kwargs))


@dataclass
class FermiLATLoader:
    name: str = "fermi_lat"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        _require_astroquery()
        raise NotImplementedError(
            "Fermi-LAT ingest is stubbed; use VizierLoader against the 4FGL catalog, "
            "or HEASARCLoader, with an explicit column mapping."
        )


@dataclass
class AugerOpenDataLoader:
    name: str = "auger"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        raise NotImplementedError(
            "Auger Open Data ingest is stubbed for v1; see notebooks/03_cosmicray_module.ipynb."
        )
