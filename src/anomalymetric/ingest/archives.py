"""astroquery-backed loaders (requires the `[archives]` extra).

These are thin convenience wrappers — the user is expected to know the schema
of the target archive. v1 ships *examples* rather than a complete catalog
abstraction; real cross-archive ingest deserves its own design pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_astroquery() -> None:
    try:
        import astroquery  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "ingest.archives requires the `[archives]` extra "
            "(pip install anomalymetric[archives])."
        ) from exc


@dataclass
class VizierLoader:
    name: str = "vizier"

    def load(self, source: str, **kwargs: Any):
        _require_astroquery()
        raise NotImplementedError(
            "VizieR ingest is stubbed for v1; see notebooks/02_archive_query_example.ipynb."
        )


@dataclass
class HEASARCLoader:
    name: str = "heasarc"

    def load(self, source: str, **kwargs: Any):
        _require_astroquery()
        raise NotImplementedError(
            "HEASARC ingest is stubbed for v1; see notebooks/02_archive_query_example.ipynb."
        )


@dataclass
class FermiLATLoader:
    name: str = "fermi_lat"

    def load(self, source: str, **kwargs: Any):
        _require_astroquery()
        raise NotImplementedError(
            "Fermi-LAT ingest is stubbed for v1."
        )


@dataclass
class AugerOpenDataLoader:
    name: str = "auger"

    def load(self, source: str, **kwargs: Any):
        raise NotImplementedError(
            "Auger Open Data ingest is stubbed for v1; see notebooks/03_cosmicray_module.ipynb."
        )
