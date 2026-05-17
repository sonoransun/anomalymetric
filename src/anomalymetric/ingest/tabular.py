"""CSV / FITS / Parquet loaders.

Expected schema (any tabular input):
    log_energy_edge_lo_eV  : float   (one per bin)
    log_energy_edge_hi_eV  : float
    value                  : float
    value_kind             : str (matches `ValueKind`)
    kind                   : str (matches `SpectrumKind`), optional - default photon
    uncertainty            : float, optional
    upper_limit_mask       : bool, optional
    exposure_cm2_s         : float, optional

We also accept legacy `log_energy_eV` (single column) by reconstructing edges
as the midpoints between consecutive values (last bin width mirrors second-to-last).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind


def _table_to_spectrum(table) -> Spectrum:
    cols = {name: np.asarray(table[name]) for name in table.colnames}
    if "log_energy_edge_lo_eV" in cols and "log_energy_edge_hi_eV" in cols:
        lo = cols["log_energy_edge_lo_eV"].astype(float)
        hi = cols["log_energy_edge_hi_eV"].astype(float)
        edges = np.concatenate([lo, [hi[-1]]])
    elif "log_energy_eV" in cols:
        centers = cols["log_energy_eV"].astype(float)
        midpoints = 0.5 * (centers[1:] + centers[:-1])
        first = centers[0] - (midpoints[0] - centers[0])
        last = centers[-1] + (centers[-1] - midpoints[-1])
        edges = np.concatenate([[first], midpoints, [last]])
    else:
        raise ValueError(
            "Input table must include log_energy_edge_{lo,hi}_eV columns "
            "or a single log_energy_eV column."
        )
    value = cols["value"].astype(float)
    value_kind = ValueKind(str(cols["value_kind"][0]))
    kind = SpectrumKind(str(cols["kind"][0])) if "kind" in cols else SpectrumKind.PHOTON
    uncertainty = cols["uncertainty"].astype(float) if "uncertainty" in cols else None
    ul = cols["upper_limit_mask"].astype(bool) if "upper_limit_mask" in cols else None
    expo = cols["exposure_cm2_s"].astype(float) if "exposure_cm2_s" in cols else None
    return Spectrum(
        log_energy_edges_eV=edges,
        value=value,
        value_kind=value_kind,
        kind=kind,
        uncertainty=uncertainty,
        upper_limit_mask=ul,
        exposure_cm2_s=expo,
    )


def spectrum_to_records(spec: Spectrum) -> list[dict]:
    edges = spec.log_energy_edges_eV
    rows: list[dict] = []
    for i in range(spec.n_bins):
        row = {
            "log_energy_edge_lo_eV": float(edges[i]),
            "log_energy_edge_hi_eV": float(edges[i + 1]),
            "value": float(spec.value[i]),
            "value_kind": spec.value_kind.value,
            "kind": spec.kind.value,
        }
        if spec.uncertainty is not None:
            row["uncertainty"] = float(spec.uncertainty[i])
        if spec.upper_limit_mask is not None:
            row["upper_limit_mask"] = bool(spec.upper_limit_mask[i])
        if spec.exposure_cm2_s is not None:
            row["exposure_cm2_s"] = float(spec.exposure_cm2_s[i])
        rows.append(row)
    return rows


@dataclass
class CSVLoader:
    name: str = "csv"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        from astropy.table import Table

        return _table_to_spectrum(Table.read(source, format="csv"))


@dataclass
class FITSLoader:
    name: str = "fits"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        from astropy.table import Table

        return _table_to_spectrum(Table.read(source))


@dataclass
class ParquetLoader:
    name: str = "parquet"

    def load(self, source: str, **kwargs: Any) -> Spectrum:
        import pyarrow.parquet as pq
        from astropy.table import Table

        pa_table = pq.read_table(source)
        cols = {name: pa_table[name].to_numpy(zero_copy_only=False) for name in pa_table.schema.names}
        return _table_to_spectrum(Table(cols))


def write_spectrum(spec: Spectrum, path: str | Path) -> None:
    """Write a Spectrum to CSV/FITS/Parquet based on the path extension."""
    from astropy.table import Table

    rows = spectrum_to_records(spec)
    table = Table(rows)
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        table.write(p, format="csv", overwrite=True)
    elif ext in {".fits", ".fit"}:
        table.write(p, overwrite=True)
    elif ext == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Bypass pandas — build pyarrow.Table directly from columns.
        arrays = {name: pa.array(np.asarray(table[name])) for name in table.colnames}
        pq.write_table(pa.table(arrays), p)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .csv/.fits/.parquet")
