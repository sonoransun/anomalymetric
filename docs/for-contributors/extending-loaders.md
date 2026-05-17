# Extending loaders

Custom spectrum loaders are discovered via Python entry points — no
modification to `anomalymetric` itself is required.

## The `SpectrumLoader` protocol

From [`src/anomalymetric/ingest/base.py`](https://github.com/your-org/anomalymetric/blob/main/src/anomalymetric/ingest/base.py):

```python
class SpectrumLoader(Protocol):
    name: str
    def load(self, source: str, **kwargs: Any) -> Spectrum: ...
```

A loader is a class with:

- a `name` attribute (used as the CLI `--loader` value and the entry-point
  name);
- a `load(source, **kwargs)` method returning a `Spectrum`.

## Minimum loader

```python
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind
import numpy as np

class MyJSONLoader:
    name = "myjson"

    def load(self, source, **kwargs):
        import json
        with open(source) as fh:
            payload = json.load(fh)
        return Spectrum(
            log_energy_edges_eV=np.asarray(payload["log_edges"]),
            value=np.asarray(payload["counts"]),
            value_kind=ValueKind.COUNTS_PER_BIN,
            kind=SpectrumKind.PHOTON,
            exposure_cm2_s=np.asarray(payload["exposure"]),
        )
```

## Registering via entry points

In *your* package's `pyproject.toml`:

```toml
[project.entry-points."anomalymetric.loaders"]
myjson = "my_package.loaders:MyJSONLoader"
```

After `pip install -e .`, the loader is discoverable:

```python
from anomalymetric.ingest.base import discover
print(discover())
# {'csv': <CSVLoader>, 'fits': <FITSLoader>, 'parquet': <ParquetLoader>,
#  'synthetic': <SyntheticLoader>, 'myjson': <MyJSONLoader>}
```

The CLI picks it up automatically if you pass `--loader myjson`.

## Conventions for tabular schemas

The built-in CSV/FITS/Parquet loaders use this schema:

| Column | Required? | Notes |
| --- | --- | --- |
| `log_energy_edge_lo_eV` | yes\* | One per bin (along with `log_energy_edge_hi_eV`) |
| `log_energy_edge_hi_eV` | yes\* | Together they define bin edges |
| `log_energy_eV` | yes\* | Legacy alternative — single column of centers; edges reconstructed at midpoints |
| `value` | yes | Per-bin value |
| `value_kind` | yes | One of `dNdE`, `EdNdE`, `E2dNdE`, `nuFnu`, `photon_rate_per_bin`, `counts_per_bin` |
| `kind` | no | Defaults to `photon` if absent |
| `uncertainty` | no | Same shape as `value` |
| `upper_limit_mask` | no | Boolean column |
| `exposure_cm2_s` | depends | Required iff `value_kind == counts_per_bin` |

\* Pass either the explicit `_lo`/`_hi` pair **or** a single `log_energy_eV`
column — not both.

If you write a loader for a more idiosyncratic source format, do the
schema mapping inside `load()` and return a fully-populated `Spectrum`.
Downstream code (likelihood, scoring) reads only the `Spectrum` interface
and is agnostic to where the data came from.

## Testing your loader

Two unit-test patterns work well:

1. **Round-trip**: build a `Spectrum` programmatically, write it to your
   format, load it back, assert equality bin-for-bin.
2. **Reference fixture**: commit a tiny example file under
   `tests/data/`, load it, assert known values.

See `tests/test_spectrum_units.py` for the conversion round-trips and
`tests/test_cli_smoke.py` for an end-to-end loader-from-file test.
