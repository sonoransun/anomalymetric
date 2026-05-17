"""Gross-Vitells / Bonferroni trials factor sanity checks."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from anomalymetric.score.trials import gross_vitells_global_p, local_p


def test_local_p_matches_chi2_half():
    # For a single test under the boundary, p_local = 0.5 * chi2_1.sf(TS).
    for ts in [1.0, 4.0, 9.0, 16.0, 25.0]:
        expected = 0.5 * chi2.sf(ts, df=1)
        assert local_p(ts) == np.float64(expected)


def test_bonferroni_increases_with_trials():
    ts = 16.0  # ~4 sigma local
    p1 = gross_vitells_global_p(ts, n_trials=1)
    p10 = gross_vitells_global_p(ts, n_trials=10)
    p100 = gross_vitells_global_p(ts, n_trials=100)
    assert p1 < p10 < p100
    # With 100 trials a 4-sigma local should drop well below 4-sigma global.
    assert p100 > p1


def test_zero_ts_gives_p_one():
    assert local_p(0.0) == 1.0
    assert gross_vitells_global_p(0.0, n_trials=10) == 1.0
