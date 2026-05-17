"""Bayes-factor alternative to the PLR score (opt-in via the `[bayes]` extra).

Stubbed for v1: importing `dynesty` is deferred to call time so the dependency
remains optional. A reference implementation lives in notebook 04.
"""

from __future__ import annotations

from typing import Optional

from anomalymetric.forward.base import ForwardModel
from anomalymetric.spectrum import Spectrum


def bayes_factor(
    spectrum: Spectrum,
    natural_components: list,
    exotic_component,
    *,
    forward: Optional[ForwardModel] = None,
    nlive: int = 250,
) -> float:
    """Return ln(B) = ln Z_alt - ln Z_nat using dynesty nested sampling.

    Raises ImportError if the `[bayes]` extra is not installed.
    """
    try:
        import dynesty  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "score.bayes requires the `[bayes]` extra (pip install anomalymetric[bayes])."
        ) from exc

    raise NotImplementedError(
        "Bayes-factor scoring is intentionally stubbed in v1; see "
        "notebooks/04_matched_filter_library.ipynb for a reference workflow."
    )
