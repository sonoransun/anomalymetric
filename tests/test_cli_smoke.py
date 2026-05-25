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


def test_version_and_list_commands():
    runner = CliRunner()
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and "anomalymetric" in r.output

    r = runner.invoke(app, ["list", "channels"])
    assert r.exit_code == 0 and "magnetometric" in r.output and "gravitational" in r.output

    r = runner.invoke(app, ["list", "loaders"])
    assert r.exit_code == 0 and "squid" in r.output

    r = runner.invoke(app, ["list", "templates", "--channel", "gravitational"])
    assert r.exit_code == 0 and "ep.microscope" in r.output


def test_generate_and_score_squid(tmp_path: Path):
    runner = CliRunner()
    bg = tmp_path / "squid_bg.parquet"
    axion = tmp_path / "squid_axion.parquet"
    rank = tmp_path / "squid_rank.csv"

    r = runner.invoke(app, ["generate", "squid", "--kind", "noise_floor", "-o", str(bg), "--seed", "0"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        ["generate", "squid", "--kind", "axion_line", "--mass-ev", "4e-6",
         "--line-amplitude", "8", "-o", str(axion), "--seed", "1"],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["score", str(bg), str(axion), "-o", str(rank)])
    assert r.exit_code == 0, r.output
    text = rank.read_text()
    assert text.find("squid_axion") < text.find("squid_bg")  # anomaly ranks first


def test_generate_and_score_gravimeter(tmp_path: Path):
    runner = CliRunner()
    bg = tmp_path / "grav_bg.parquet"
    ep = tmp_path / "grav_ep.parquet"
    rank = tmp_path / "grav_rank.csv"

    r = runner.invoke(app, ["generate", "gravimeter", "--kind", "noise_floor", "-o", str(bg), "--seed", "0"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        ["generate", "gravimeter", "--kind", "ep_violation", "--line-amplitude", "8",
         "-o", str(ep), "--seed", "1"],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["score", str(bg), str(ep), "-o", str(rank)])
    assert r.exit_code == 0, r.output
    text = rank.read_text()
    assert text.find("grav_ep") < text.find("grav_bg")
