"""Typer-based CLI: `anomalymetric generate|score|fit|query ...`"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Loeb-Turner-style emission anomaly metric.")
generate_app = typer.Typer(help="Generate synthetic spectra.")
app.add_typer(generate_app, name="generate")
list_app = typer.Typer(help="List available channels, loaders, and exotic templates.")
app.add_typer(list_app, name="list")


def _version_callback(value: bool) -> None:
    if value:
        from anomalymetric import __version__

        typer.echo(f"anomalymetric {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """Loeb-Turner-style emission anomaly metric."""


@list_app.command("channels")
def cli_list_channels() -> None:
    """List the supported measurement channels (SpectrumKind values)."""
    from anomalymetric.spectrum import GAUSSIAN_KINDS, SpectrumKind

    for k in SpectrumKind:
        family = "Gaussian PSD" if k in GAUSSIAN_KINDS else "Poisson counts"
        typer.echo(f"{k.value:<16s} {family}")


@list_app.command("loaders")
def cli_list_loaders() -> None:
    """List registered ingest loaders (entry-point + built-in)."""
    from anomalymetric.ingest.base import discover

    for name in sorted(discover()):
        typer.echo(name)


@list_app.command("templates")
def cli_list_templates(
    channel: str = typer.Option(
        "photon", help="photon | cosmicray | magnetometric | gravitational"
    ),
) -> None:
    """List the exotic-template library for a channel."""
    if channel == "photon":
        from anomalymetric.models.exotic import default_library as lib
    elif channel == "cosmicray":
        from anomalymetric.cosmicray.score import cr_exotic_library as lib
    elif channel == "magnetometric":
        from anomalymetric.magnetometric.score import squid_exotic_library as lib
    elif channel == "gravitational":
        from anomalymetric.gravitational.score import grav_exotic_library as lib
    else:
        raise typer.BadParameter(f"Unknown channel '{channel}'")
    for template in lib():
        typer.echo(template.name)


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


@generate_app.command("squid")
def cli_generate_squid(
    output: Path = typer.Option(..., "-o", "--output", help="Output path (.csv/.fits/.parquet)"),
    kind: str = typer.Option("noise_floor", help="noise_floor | axion_line | dark_photon | bump"),
    mass_eV: float = typer.Option(4.0e-6, help="Signal mass in eV (line center = m); axion/dark_photon"),
    line_amplitude: float = typer.Option(6.0, help="Line peak as a multiple of the local noise floor"),
    log_f_min: float = typer.Option(2.0, help="log10(f/Hz) lower edge"),
    log_f_max: float = typer.Option(9.0, help="log10(f/Hz) upper edge"),
    bins_per_decade: int = typer.Option(20),
    sigma_frac: float = typer.Option(0.05, help="Per-bin Gaussian sigma as a fraction of the PSD"),
    seed: Optional[int] = typer.Option(0, help="RNG seed"),
) -> None:
    """Generate a synthetic magnetometric (SQUID) PSD spectrum."""
    from anomalymetric.ingest.synthetic_psd import (
        synthetic_squid_natural,
        synthetic_squid_with_exotic,
    )
    from anomalymetric.ingest.tabular import write_spectrum

    base = synthetic_squid_natural(
        log_f_min=log_f_min, log_f_max=log_f_max, bins_per_decade=bins_per_decade,
        sigma_frac=sigma_frac, seed=seed, noise=(kind == "noise_floor"),
    )
    if kind == "noise_floor":
        spec = base
    elif kind in {"axion_line", "dark_photon", "bump"}:
        spec = synthetic_squid_with_exotic(
            base, line_E_eV=mass_eV, line_amplitude=line_amplitude,
            width_dex=(0.3 if kind == "bump" else 0.02),
            seed=(seed + 1) if seed is not None else None,
        )
    else:
        raise typer.BadParameter(f"Unknown kind '{kind}'")
    write_spectrum(spec, output)
    typer.echo(f"Wrote {spec.n_bins}-bin magnetometric '{kind}' spectrum to {output}")


@generate_app.command("gravimeter")
def cli_generate_gravimeter(
    output: Path = typer.Option(..., "-o", "--output", help="Output path (.csv/.fits/.parquet)"),
    kind: str = typer.Option("noise_floor", help="noise_floor | yukawa | ep_violation | osc_dm | bump"),
    mass_eV: float = typer.Option(1.0e-15, help="Scalar-DM mass in eV (osc_dm)"),
    mod_freq_hz: float = typer.Option(1.0e-2, help="Source-modulation frequency in Hz (yukawa)"),
    line_amplitude: float = typer.Option(6.0, help="Line peak as a multiple of the local noise floor"),
    log_f_min: float = typer.Option(-5.0, help="log10(f/Hz) lower edge"),
    log_f_max: float = typer.Option(1.0, help="log10(f/Hz) upper edge"),
    bins_per_decade: int = typer.Option(20),
    sigma_frac: float = typer.Option(0.05, help="Per-bin Gaussian sigma as a fraction of the PSD"),
    seed: Optional[int] = typer.Option(0, help="RNG seed"),
) -> None:
    """Generate a synthetic gravitational-differential PSD spectrum."""
    from anomalymetric.gravitational.fifth_force import (
        microscope_modulation_freq_hz,
        oscillating_dm_freq_hz,
    )
    from anomalymetric.ingest.synthetic_psd import (
        synthetic_grav_natural,
        synthetic_grav_with_exotic,
    )
    from anomalymetric.ingest.tabular import write_spectrum
    from anomalymetric.units import H_PLANCK_EV_S

    base = synthetic_grav_natural(
        log_f_min=log_f_min, log_f_max=log_f_max, bins_per_decade=bins_per_decade,
        sigma_frac=sigma_frac, seed=seed, noise=(kind == "noise_floor"),
    )
    if kind == "noise_floor":
        spec = base
    else:
        if kind == "yukawa":
            line_E_eV = H_PLANCK_EV_S * mod_freq_hz
        elif kind == "ep_violation":
            line_E_eV = H_PLANCK_EV_S * microscope_modulation_freq_hz()
        elif kind == "osc_dm":
            line_E_eV = H_PLANCK_EV_S * oscillating_dm_freq_hz(mass_eV)
        elif kind == "bump":
            line_E_eV = H_PLANCK_EV_S * 1.0e-3
        else:
            raise typer.BadParameter(f"Unknown kind '{kind}'")
        spec = synthetic_grav_with_exotic(
            base, line_E_eV=line_E_eV, line_amplitude=line_amplitude,
            width_dex=(0.3 if kind == "bump" else 0.02),
            seed=(seed + 1) if seed is not None else None,
        )
    write_spectrum(spec, output)
    typer.echo(f"Wrote {spec.n_bins}-bin gravitational '{kind}' spectrum to {output}")


@app.command("score")
def cli_score(
    inputs: list[Path] = typer.Argument(..., help="One or more spectrum files"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Ranking CSV"),
    loader: Optional[str] = typer.Option(None, "--loader", help="Force a loader (else inferred from extension)"),
) -> None:
    """Score a catalog of spectra and write a ranked CSV."""
    from anomalymetric.pipeline import load_spectrum, score_catalog
    from anomalymetric.score.ranking import ranking_to_records

    catalog = [(str(p), load_spectrum(str(p), loader=loader)) for p in inputs]
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
    loader: Optional[str] = typer.Option(None, "--loader", help="Force a loader (else inferred from extension)"),
) -> None:
    """Fit a chosen natural model to a single spectrum and print MLE values."""
    from anomalymetric.models.inference import Fit
    from anomalymetric.models.mixture import Mixture
    from anomalymetric.models.powerlaw import PowerLaw
    from anomalymetric.models.thermal import BlackBody
    from anomalymetric.pipeline import load_spectrum

    spec = load_spectrum(str(input_path), loader=loader)
    if model == "blackbody+powerlaw":
        components = [BlackBody(), PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]
    elif model == "blackbody":
        components = [BlackBody()]
    elif model == "powerlaw":
        components = [PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0)]
    elif model == "squid_noise":
        from anomalymetric.magnetometric.score import SQUIDNoiseFloor

        components = [SQUIDNoiseFloor()]
    elif model == "grav_noise":
        from anomalymetric.gravitational.score import GravNoiseFloor

        components = [GravNoiseFloor()]
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
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Write the spectrum to a file"),
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
    spec = cls().load(query)
    typer.echo(
        f"Loaded {spec.n_bins}-bin {spec.kind.value} spectrum "
        f"({spec.value_kind.value}) from {archive}:{query}"
    )
    if output is not None:
        from anomalymetric.ingest.tabular import write_spectrum

        write_spectrum(spec, output)
        typer.echo(f"Wrote spectrum to {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
