"""Workstream B1: upper-limit synthesis, round-trip, and one-sided penalties."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.ingest.synthetic import synthetic_natural
from anomalymetric.ingest.synthetic_psd import synthetic_squid_natural
from anomalymetric.ingest.tabular import spectrum_to_records, write_spectrum
from anomalymetric.models.inference import gaussian_log_likelihood, poisson_log_likelihood
from anomalymetric.pipeline import load_spectrum


def test_synthetic_sets_upper_limit_mask():
    spec = synthetic_natural(seed=0, ul_fraction=0.2)
    assert spec.upper_limit_mask is not None
    assert spec.upper_limit_mask.sum() > 0
    assert spec.upper_limit_mask.shape == spec.value.shape


def test_upper_limit_mask_round_trips_parquet(tmp_path):
    spec = synthetic_squid_natural(seed=1, ul_fraction=0.15)
    assert spec.upper_limit_mask is not None and spec.upper_limit_mask.sum() > 0
    # The records carry the bool column...
    rows = spectrum_to_records(spec)
    assert any("upper_limit_mask" in r for r in rows)
    # ...and a write/read round-trip preserves it.
    p = tmp_path / "ul.parquet"
    write_spectrum(spec, p)
    back = load_spectrum(str(p))
    assert back.upper_limit_mask is not None
    assert np.array_equal(back.upper_limit_mask, spec.upper_limit_mask)


def test_poisson_ul_penalty_is_one_sided_and_monotone():
    k = np.array([5.0])
    mask = np.array([True])
    # Predicting below the limit: ~no penalty; above: increasingly penalized.
    ll_low = poisson_log_likelihood(k, np.array([1.0]), upper_limit_mask=mask)
    ll_at = poisson_log_likelihood(k, np.array([5.0]), upper_limit_mask=mask)
    ll_high = poisson_log_likelihood(k, np.array([20.0]), upper_limit_mask=mask)
    assert ll_low > ll_at > ll_high
    assert ll_low == pytest.approx(0.0, abs=0.05)  # mu << k -> log P(N<=k) ~ 0


def test_poisson_ul_well_defined_at_zero_count():
    # k=0 UL: log P(N<=0 | mu) = -mu, finite for any mu (no divide-by-k blowup).
    ll = poisson_log_likelihood(np.array([0.0]), np.array([3.0]), upper_limit_mask=np.array([True]))
    assert ll == pytest.approx(-3.0, abs=1e-9)


def test_gaussian_ul_penalty_one_sided():
    y = np.array([2.0])
    s = np.array([0.5])
    mask = np.array([True])
    below = gaussian_log_likelihood(y, np.array([0.0]), s, upper_limit_mask=mask)
    above = gaussian_log_likelihood(y, np.array([4.0]), s, upper_limit_mask=mask)
    assert below > above
    assert below == pytest.approx(0.0)
