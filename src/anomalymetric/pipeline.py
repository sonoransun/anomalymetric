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
from anomalymetric.score.ranking import RankedSource
from anomalymetric.score.variability import VariabilityResult, variability_score
from anomalymetric.spectrum import Spectrum, SpectrumKind, SpectrumSeries


def default_natural_factory() -> list:
    """A reasonable photon-default natural mixture: blackbody + power-law."""
    return [BlackBody(T_K=300.0), PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]


def squid_natural_factory() -> list:
    """Natural baseline for the magnetometric channel: the SQUID noise floor."""
    from anomalymetric.magnetometric.score import SQUIDNoiseFloor

    return [SQUIDNoiseFloor()]


def grav_natural_factory() -> list:
    """Natural baseline for the gravitational channel: the differential noise floor."""
    from anomalymetric.gravitational.score import GravNoiseFloor

    return [GravNoiseFloor()]


def _natural_factory_for(spectrum: Spectrum) -> Callable[[], list]:
    """Pick the natural-mixture factory matching the spectrum's channel."""
    if spectrum.kind is SpectrumKind.MAGNETOMETRIC:
        return squid_natural_factory
    if spectrum.kind is SpectrumKind.GRAVITATIONAL:
        return grav_natural_factory
    return default_natural_factory


def _exotic_library_for(spectrum: Spectrum):
    """Channel-specific exotic library (None falls back to the default photon set)."""
    if spectrum.kind is SpectrumKind.MAGNETOMETRIC:
        from anomalymetric.magnetometric.score import squid_exotic_library

        return squid_exotic_library()
    if spectrum.kind is SpectrumKind.GRAVITATIONAL:
        from anomalymetric.gravitational.score import grav_exotic_library

        return grav_exotic_library()
    return None


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
    natural_factory: Optional[Callable[[], list]] = None,
    forward: Optional[ForwardModel] = None,
) -> ScoreResult:
    """Score a single spectrum, dispatching the natural mixture by channel.

    Pass `natural_factory` to override; by default photon/CR use the blackbody +
    power-law mix while the sensor channels use their noise-floor baselines.
    """
    factory = natural_factory or _natural_factory_for(spectrum)
    return loeb_turner_score(
        spectrum, factory(), forward=forward, exotic_library=_exotic_library_for(spectrum)
    )


def score_catalog(
    catalog: Iterable[tuple[str, Spectrum]],
    *,
    natural_factory: Optional[Callable[[], list]] = None,
    forward: Optional[ForwardModel] = None,
) -> list[RankedSource]:
    """Score and rank a (possibly mixed-channel) catalog by descending anomaly score."""
    ranked = [
        RankedSource(source_id=src_id, score=score_spectrum(spec, natural_factory=natural_factory, forward=forward))
        for src_id, spec in catalog
    ]
    ranked.sort(key=lambda r: r.score.anomaly_score, reverse=True)
    return ranked


def score_series(
    series: SpectrumSeries,
    *,
    natural_factory: Optional[Callable[[], list]] = None,
    forward: Optional[ForwardModel] = None,
) -> VariabilityResult:
    """Multi-epoch variability score, dispatching the natural mixture by channel."""
    if len(series) == 0:
        raise ValueError("score_series needs at least one epoch")
    factory = natural_factory or _natural_factory_for(series.spectra[0])
    library = _exotic_library_for(series.spectra[0])
    return variability_score(series, factory, forward=forward, exotic_library=library)
