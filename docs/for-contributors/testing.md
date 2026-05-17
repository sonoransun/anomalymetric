# Testing

The full suite is 31 pytest tests, runs in about a minute, and exercises
every load-bearing path end-to-end.

```
.venv/bin/pytest                                 # full suite
.venv/bin/pytest tests/test_score_known_anomaly.py -v
.venv/bin/pytest tests/test_fit_recovers_truth.py::test_fit_recovers_powerlaw_index -v
```

## Test file map

| File | What it covers |
| --- | --- |
| `test_spectrum_units.py` | `ValueKind` canonicalization round-trips and `Spectrum` invariants |
| `test_models_predict.py` | Every model produces finite, non-negative `dN/dE`; mixture is the sum of components |
| `test_likelihood_dispatch.py` | `SpectrumKind` routes to the right forward model and produces a finite log L |
| `test_fit_recovers_truth.py` | Synthetic data → fit → assert parameters within tolerance of the injected truth |
| `test_score_known_anomaly.py` | Injected 532 nm laser line ranks above clean blackbody, and the best template is `"laser.*"` |
| `test_trials_correction.py` | Gross-Vitells / Bonferroni global p-values match the chi² survival function |
| `test_cli_smoke.py` | `anomalymetric generate ... ; anomalymetric score ...` works end-to-end |

## Writing new tests

Three patterns work well in this codebase:

**1. Synthetic round-trip.** Use `synthetic_natural(...)` to generate ground
truth, fit, assert recovery. Always pass a `seed` so the test is deterministic.

```python
def test_my_model_recovers():
    spec = synthetic_natural(T_K=300., exposure_cm2_s=1e6, seed=11)
    res = Fit(MyModel(...), spec).run()
    assert abs(res.parameter_values["my_param"] - truth) / truth < 0.1
```

**2. Smoke / shape.** For new models, assert finite + non-negative on a wide
energy range. `tests/test_models_predict.py:_finite_nonneg` is the helper.

**3. CLI end-to-end via `typer.testing.CliRunner`.** See
`tests/test_cli_smoke.py` for the pattern. Use `tmp_path` from pytest, never
hardcode `/tmp` paths.

## Avoiding flake

- **Seed every RNG.** Both `numpy.random.default_rng(seed)` calls in the
  package accept a seed; pass one.
- **Don't depend on absolute log-likelihoods.** Compare *differences* —
  `assert anom_score.test_statistic > bg_score.test_statistic` rather than
  `assert anom_score.test_statistic > 1000`.
- **Don't rely on optimizer iteration counts.** `Fit.run()` may legitimately
  take a different number of Nelder-Mead steps when scipy or numpy versions
  shift. Assert on parameter values and log-likelihoods, not on counters.

## Optional-extra tests

If your test needs `[archives]` or `[cr]`, gate the import:

```python
import pytest
astroquery = pytest.importorskip("astroquery")
```

This way the test is silently skipped on a default `[dev]` install rather
than erroring out.

## Continuous integration

CI is intentionally not configured in v1 — the workflow file at
`.github/workflows/docs.yml` only deploys the documentation. Adding a
test workflow is a known follow-up (one short `pytest` matrix run).
