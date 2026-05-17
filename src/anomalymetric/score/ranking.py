"""Batch scoring + ranking across a catalog of spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.models.exotic import ExoticTemplate
from anomalymetric.score.loeb_turner import ScoreResult, loeb_turner_score
from anomalymetric.spectrum import Spectrum


@dataclass
class RankedSource:
    source_id: str
    score: ScoreResult


def rank_catalog(
    sources: Iterable[tuple[str, Spectrum]],
    natural_components_factory,
    *,
    forward: Optional[ForwardModel] = None,
    exotic_library: Optional[list[ExoticTemplate]] = None,
) -> list[RankedSource]:
    """Score each spectrum and return ranked by descending anomaly_score.

    `natural_components_factory` is a no-arg callable that produces a fresh list
    of natural-mixture `Model`s per source (necessary since fits mutate params).
    """
    ranked: list[RankedSource] = []
    for src_id, spec in sources:
        nat = natural_components_factory()
        res = loeb_turner_score(spec, nat, forward=forward, exotic_library=exotic_library)
        ranked.append(RankedSource(source_id=src_id, score=res))
    ranked.sort(key=lambda r: r.score.anomaly_score, reverse=True)
    return ranked


def ranking_to_records(ranked: list[RankedSource]) -> list[dict]:
    rows = []
    for r in ranked:
        rows.append(
            {
                "source_id": r.source_id,
                "anomaly_score": r.score.anomaly_score,
                "test_statistic": r.score.test_statistic,
                "delta_log_likelihood": r.score.delta_log_likelihood,
                "best_template": r.score.best_template,
                "notes": r.score.notes,
            }
        )
    return rows
