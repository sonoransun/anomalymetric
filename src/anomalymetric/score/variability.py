"""Multi-epoch (variability) scoring over a `SpectrumSeries`.

Each epoch is scored with the single-epoch Loeb–Turner test; the per-template test
statistics are then summed across epochs (a sum of `n_epochs` independent chi^2_1
statistics is chi^2_{n_epochs} under the null), and the most-favored template is
trials-corrected over the library. A separate flux-constancy statistic flags plain
brightness variability independent of any template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.score.loeb_turner import ScoreResult, loeb_turner_score
from anomalymetric.score.trials import local_p_chi2
from anomalymetric.spectrum import GAUSSIAN_KINDS, Spectrum, SpectrumSeries
from typing import Callable, Optional


@dataclass
class EpochScore:
    epoch_mjd: float
    flux: float
    score: ScoreResult


@dataclass
class VariabilityResult:
    global_test_statistic: float  # max over templates of the epoch-summed TS
    anomaly_score: float  # trials-corrected -log10 p_global
    flux_variability: float  # fractional RMS of the per-epoch flux
    best_template: str
    per_epoch: list[EpochScore] = field(default_factory=list)
    notes: str = ""


def _epoch_flux(spec: Spectrum) -> float:
    """Total per-epoch flux: summed PSD (Gaussian) or summed counts (Poisson)."""
    if spec.kind in GAUSSIAN_KINDS:
        return float(np.sum(spec.canonical_value()))
    from anomalymetric.models.inference import observed_counts_from_spectrum

    return float(np.sum(observed_counts_from_spectrum(spec)))


def variability_score(
    series: SpectrumSeries,
    natural_factory: Callable[[], list],
    *,
    forward: Optional[ForwardModel] = None,
    exotic_library=None,
) -> VariabilityResult:
    """Score a multi-epoch series for a recurring/transient template + flux variability.

    `natural_factory` is a no-arg callable returning a *fresh* natural mixture per
    epoch (fits mutate parameters) — the same contract as `ranking.rank_catalog`.
    """
    if len(series) == 0:
        raise ValueError("variability_score needs at least one epoch")

    per_epoch: list[EpochScore] = []
    ts_by_template: dict[str, float] = {}
    for mjd, spec in zip(series.epochs_mjd, series.spectra):
        res = loeb_turner_score(
            spec, natural_factory(), forward=forward, exotic_library=exotic_library
        )
        per_epoch.append(EpochScore(float(mjd), _epoch_flux(spec), res))
        for t in res.per_template:
            ts_by_template[t.template_name] = ts_by_template.get(t.template_name, 0.0) + t.ts

    n_epochs = len(series)
    n_templates = max(len(ts_by_template), 1)
    best_template = max(ts_by_template, key=ts_by_template.get)
    summed_ts = ts_by_template[best_template]
    # Epoch-summed TS ~ chi^2_{n_epochs}; trials-correct over the library (Bonferroni).
    p_local = local_p_chi2(summed_ts, n_epochs)
    p_global = 1.0 - (1.0 - p_local) ** n_templates if n_templates > 1 else p_local
    anomaly_score = float(-np.log10(max(p_global, 1e-300)))

    fluxes = np.array([e.flux for e in per_epoch], dtype=float)
    mean_flux = float(np.mean(fluxes))
    frac_rms = float(np.std(fluxes) / mean_flux) if mean_flux > 0 else 0.0

    return VariabilityResult(
        global_test_statistic=float(summed_ts),
        anomaly_score=anomaly_score,
        flux_variability=frac_rms,
        best_template=best_template,
        per_epoch=per_epoch,
        notes=f"{n_epochs} epochs; {n_templates} templates; summed TS over best template",
    )
