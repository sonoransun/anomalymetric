"""Smoke test: CLI generates + scores end-to-end."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from anomalymetric.cli import app


def test_generate_and_score(tmp_path: Path):
    runner = CliRunner()
    bb_path = tmp_path / "bb.parquet"
    anom_path = tmp_path / "anom.parquet"
    rank_path = tmp_path / "rank.csv"

    r = runner.invoke(
        app,
        [
            "generate",
            "synthetic",
            "--kind",
            "blackbody",
            "-o",
            str(bb_path),
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0, r.output

    r = runner.invoke(
        app,
        [
            "generate",
            "synthetic",
            "--kind",
            "exotic_line",
            "--line-ev",
            "2.331",
            "--line-amplitude",
            "5000",
            "-o",
            str(anom_path),
            "--seed",
            "1",
        ],
    )
    assert r.exit_code == 0, r.output

    r = runner.invoke(
        app,
        ["score", str(bb_path), str(anom_path), "-o", str(rank_path)],
    )
    assert r.exit_code == 0, r.output
    assert rank_path.exists()
    text = rank_path.read_text()
    # Anomalous file should appear above the clean blackbody.
    bb_idx = text.find("bb.parquet")
    anom_idx = text.find("anom.parquet")
    assert bb_idx != -1 and anom_idx != -1
    assert anom_idx < bb_idx
