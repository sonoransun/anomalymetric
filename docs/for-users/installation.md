# Installation

`anomalymetric` requires Python 3.10+. The lightweight core uses only `numpy`,
`scipy`, `astropy`, `pyarrow`, and `typer`.

## Recommended: editable install with the `[dev]` extra

```
git clone https://github.com/your-org/anomalymetric
cd anomalymetric
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest`, `matplotlib`, and the jupyter stack used by
the example notebooks. Verify with:

```
.venv/bin/pytest
```

(31 tests should pass in about a minute.)

## Optional extras

| Extra | What it adds | Install command |
| --- | --- | --- |
| `archives` | astroquery loaders (VizieR, HEASARC, Fermi-LAT, Auger) | `.venv/bin/pip install -e ".[archives]"` |
| `bayes` | emcee + dynesty for opt-in Bayes-factor scoring | `.venv/bin/pip install -e ".[bayes]"` |
| `cr` | CRPropa3 wrapper for cosmic-ray propagation | `.venv/bin/pip install -e ".[cr]"` |
| `naima` | non-thermal SED models (synchrotron, IC, π⁰) | `.venv/bin/pip install -e ".[naima]"` |
| `docs` | mkdocs-material + Mermaid plugin + figure-generation deps | `.venv/bin/pip install -e ".[docs]"` |

Stack extras with commas: `pip install -e ".[dev,archives,bayes]"`.

## System dependencies

For `[cr]`: CRPropa3 requires HDF5 and Boost headers. See the
[CRPropa3 install notes](https://github.com/CRPropa/CRPropa3).

For nicer GIFs from the figure scripts: `apt install gifsicle` (optional;
scripts work without it).
