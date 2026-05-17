"""Anomalymetric: rank emissions by deviation from natural-source mixtures.

Top-level re-exports keep the import surface small for notebook use.
"""

from anomalymetric.spectrum import Spectrum, SpectrumKind, SpectrumSeries, ValueKind
from anomalymetric.score.loeb_turner import loeb_turner_score, ScoreResult

__all__ = [
    "Spectrum",
    "SpectrumKind",
    "SpectrumSeries",
    "ValueKind",
    "loeb_turner_score",
    "ScoreResult",
]

__version__ = "0.1.0"
