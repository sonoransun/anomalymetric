"""Typer-based CLI: `anomalymetric generate|score|fit|query ...`"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Loeb-Turner-style emission anomaly metric.")
generate_app = typer.Typer(help="Generate synthetic spectra.")
app.add_typer(generate_app, name="generate")


@generate_app.command("synthetic")
def cli_generate_synthetic(
    output: Path = typer.Option(..., "-o", "--output", help="Output path (.csv/.fits/.parquet)"),
    kind: str = typer.Option("blackbody", help="blackbody | exotic_line"),
    T: float = typer.Option(300.0, help="Blackbody temperature in K"),
    line_eV: float = typer.Option(2.33, help="Exotic line energy in eV (kind=exotic_line)"),
    line_amplitude: float = typer.Option(100.0, help="Line amplitude (kind=exotic_line)"),
    log_e_min: float = typer.Option(-3.0),
    log_e_max: float = typer.Option(6.0),
    bins_per_decade: int = typer.Option(20),
    seed: Optional[int] = typer.Option(0, help="RNG seed"),
) -> None:
    from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
    from anomalymetric.ingest.tabular import write_spectrum

    if kind == "blackbody":
        spec = synthetic_natural(
            log_e_min=log_e_min,
            log_e_max=log_e_max,
            bins_per_decade=bins_per_decade,
            T_K=T,
            seed=seed,
        )
    elif kind == "exotic_line":
        base = synthetic_natural(
            log_e_min=log_e_min,
            log_e_max=log_e_max,
            bins_per_decade=bins_per_decade,
            T_K=T,
            seed=seed,
        )
        spec = synthetic_with_exotic(
            base,
            line_E_eV=line_eV,
            line_amplitude=line_amplitude,
            seed=(seed + 1) if seed is not None else None,
        )
    else:
        raise typer.BadParameter(f"Unknown kind '{kind}'")
    write_spectrum(spec, output)
    typer.echo(f"Wrote {spec.n_bins}-bin {kind} spectrum to {output}")


@app.command("score")
def cli_score(
    inputs: list[Path] = typer.Argument(..., help="One or more spectrum files"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Ranking CSV"),
) -> None:
    """Score a catalog of spectra and write a ranked CSV."""
    from anomalymetric.pipeline import load_spectrum, score_catalog
    from anomalymetric.score.ranking import ranking_to_records

    catalog = [(str(p), load_spectrum(str(p))) for p in inputs]
    ranked = score_catalog(catalog)
    rows = ranking_to_records(ranked)
    for r in rows:
        typer.echo(
            f"{r['source_id']:<40s}  score={r['anomaly_score']:.3f}  "
            f"TS={r['test_statistic']:.2f}  best={r['best_template']}"
        )
    if output is not None:
        import csv

        with open(output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        typer.echo(f"Wrote ranking to {output}")


@app.command("fit")
def cli_fit(
    input_path: Path = typer.Argument(..., help="Spectrum file"),
    model: str = typer.Option("blackbody+powerlaw", help="Model spec"),
) -> None:
    """Fit a chosen natural model to a single spectrum and print MLE values."""
    from anomalymetric.models.inference import Fit
    from anomalymetric.models.mixture import Mixture
    from anomalymetric.models.powerlaw import PowerLaw
    from anomalymetric.models.thermal import BlackBody
    from anomalymetric.pipeline import load_spectrum

    spec = load_spectrum(str(input_path))
    if model == "blackbody+powerlaw":
        components = [BlackBody(), PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]
    elif model == "blackbody":
        components = [BlackBody()]
    elif model == "powerlaw":
        components = [PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]
    else:
        raise typer.BadParameter(f"Unknown model '{model}'")
    mix = Mixture(components, name="natural")
    res = Fit(mix, spec).run()
    typer.echo(f"log L = {res.log_likelihood:.3f}  success={res.success}")
    for k, v in res.parameter_values.items():
        typer.echo(f"  {k} = {v:.4g}")


@app.command("query")
def cli_query(
    archive: str = typer.Argument(..., help="Archive name (vizier|heasarc|fermi_lat|auger)"),
    query: str = typer.Argument(..., help="Archive-specific query string"),
) -> None:
    """Query an external archive (requires the [archives] extra)."""
    from anomalymetric.ingest import archives

    cls = {
        "vizier": archives.VizierLoader,
        "heasarc": archives.HEASARCLoader,
        "fermi_lat": archives.FermiLATLoader,
        "auger": archives.AugerOpenDataLoader,
    }.get(archive)
    if cls is None:
        raise typer.BadParameter(f"Unknown archive '{archive}'")
    cls().load(query)


if __name__ == "__main__":  # pragma: no cover
    app()
