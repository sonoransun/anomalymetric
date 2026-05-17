"""High-level orchestration: load → forward → fit → score → rank.

The pipeline is a thin glue layer; everything load-bearing lives in the
sub-modules. Useful from the CLI and notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from anomalymetric.forward.base import ForwardModel
from anomalymetric.ingest.base import discover, get as get_loader
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.score.loeb_turner import ScoreResult, loeb_turner_score
from anomalymetric.score.ranking import RankedSource, rank_catalog
from anomalymetric.spectrum import Spectrum, SpectrumKind


def default_natural_factory() -> list:
    """A reasonable photon-default natural mixture: blackbody + power-law."""
    return [BlackBody(T_K=300.0), PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]


def load_spectrum(path_or_uri: str, loader: Optional[str] = None) -> Spectrum:
    """Load via an explicit loader name or by file extension."""
    if loader is None:
        ext = Path(path_or_uri).suffix.lower().lstrip(".")
        ext_map = {"csv": "csv", "fits": "fits", "fit": "fits", "parquet": "parquet"}
        loader = ext_map.get(ext)
    if loader is None:
        raise ValueError(
            f"Could not infer loader for '{path_or_uri}'. Pass loader=... "
            f"explicitly. Known: {sorted(discover())}"
        )
    return get_loader(loader).load(path_or_uri)


def score_spectrum(
    spectrum: Spectrum,
    *,
    natural_factory: Callable[[], list] = default_natural_factory,
    forward: Optional[ForwardModel] = None,
) -> ScoreResult:
    """Score a single spectrum with the default natural mixture."""
    return loeb_turner_score(spectrum, natural_factory(), forward=forward)


def score_catalog(
    catalog: Iterable[tuple[str, Spectrum]],
    *,
    natural_factory: Callable[[], list] = default_natural_factory,
    forward: Optional[ForwardModel] = None,
) -> list[RankedSource]:
    return rank_catalog(catalog, natural_factory, forward=forward)
