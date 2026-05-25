"""Shared pytest config: auto-mark heavy modules `slow`, gate `remote_data` tests."""

from __future__ import annotations

import pytest

# Modules dominated by full optimizer fits / template-library scoring. Deselect
# during fast iteration with `pytest -m "not slow"`.
_SLOW_MODULES = {
    "test_score_known_anomaly",
    "test_psd_score_known_anomaly",
    "test_psd_fit_recovers_truth",
    "test_fit_recovers_truth",
    "test_bayes_factor",
    "test_cli_smoke",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-remote", action="store_true", default=False,
        help="run tests marked `remote_data` (live network / external archives)",
    )


def pytest_collection_modifyitems(config, items):
    run_remote = config.getoption("--run-remote")
    skip_remote = pytest.mark.skip(reason="needs --run-remote (live network)")
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        if module in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
        if "remote_data" in item.keywords and not run_remote:
            item.add_marker(skip_remote)
