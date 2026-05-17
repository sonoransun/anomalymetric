# Architecture

The package follows a gammapy-inspired `Dataset / Model / Fit` shape so the
implementation can later be retargeted onto gammapy without breaking user code.

## Module dependency graph

```mermaid
graph LR
  subgraph foundation
    spectrum[spectrum]
    units[units]
  end
  subgraph subsystems
    ingest[ingest]
    forward[forward]
    models[models]
    score[score]
    cosmicray[cosmicray]
  end
  subgraph entrypoints
    pipeline[pipeline]
    cli[cli]
  end

  units --> spectrum
  spectrum --> ingest
  spectrum --> forward
  spectrum --> models
  spectrum --> score
  spectrum --> cosmicray
  units --> ingest
  units --> models
  units --> forward
  forward --> models
  models --> score
  models --> cosmicray
  score --> cosmicray
  forward --> cosmicray
  ingest --> pipeline
  forward --> pipeline
  score --> pipeline
  models --> pipeline
  pipeline --> cli
```

`spectrum` and `units` are the foundation; nothing imports above them. `cli`
and `pipeline` are the only entry points users should call.

## Class structure

```mermaid
classDiagram
  class Spectrum {
    +log_energy_edges_eV
    +value
    +ValueKind value_kind
    +SpectrumKind kind
    +exposure_cm2_s
    +upper_limit_mask
    +meta
    +as_dnde()
    +expected_counts()
  }
  class SpectrumSeries {
    +epochs_mjd
    +spectra: list~Spectrum~
  }

  class Model {
    <<Protocol>>
    +name
    +parameters: Parameters
    +dnde(log_E)
  }
  class BlackBody
  class SolarReflection
  class PowerLaw
  class BrokenPowerLaw
  class GaussianLine
  class HardCutoffPowerLaw
  class GZKViolatingTail
  class TripleBrokenPowerLaw
  class Mixture {
    +components: list~Model~
    +dnde(log_E)
    +component_dnde(log_E)
  }

  Model <|.. BlackBody
  Model <|.. SolarReflection
  Model <|.. PowerLaw
  Model <|.. BrokenPowerLaw
  Model <|.. GaussianLine
  Model <|.. HardCutoffPowerLaw
  Model <|.. GZKViolatingTail
  Model <|.. TripleBrokenPowerLaw
  Mixture o-- Model

  class ForwardModel {
    <<Protocol>>
    +forward(log_edges, dnde)
  }
  class IdentityResponse
  class GaussianEnergyResponse
  class FlatExposure
  class TabulatedExposure
  ForwardModel <|.. IdentityResponse
  ForwardModel <|.. GaussianEnergyResponse
  ForwardModel <|.. FlatExposure
  ForwardModel <|.. TabulatedExposure

  class Fit {
    +model: Model
    +spectrum: Spectrum
    +forward: ForwardModel
    +run()
    +profile_likelihood(name, scan)
  }
  Fit --> Model
  Fit --> Spectrum
  Fit --> ForwardModel

  class ExoticTemplate {
    +name
    +factory() -> Model
  }
  class ScoreResult {
    +delta_log_likelihood
    +test_statistic
    +anomaly_score
    +best_template
    +per_template
  }
  SpectrumSeries o-- Spectrum
```

`Model` and `ForwardModel` are `typing.Protocol`s — implement them by writing
a class with the right attributes; no inheritance required. `Mixture`,
`ExoticTemplate`, `Fit`, and `ScoreResult` are concrete dataclasses.

## End-to-end pipeline

```mermaid
flowchart TD
  A[Spectrum file<br/>CSV / FITS / Parquet] -->|SpectrumLoader| B[Spectrum<br/>ValueKind + SpectrumKind]
  B --> C{kind?}
  C -->|photon| D1[IdentityResponse /<br/>GaussianEnergyResponse]
  C -->|cr_*| D2[FlatExposure /<br/>TabulatedExposure]
  C -->|neutrino| D3[Effective area]
  D1 --> E[Fit&#40;natural&#41;<br/>log L_nat]
  D2 --> E
  D3 --> E
  E --> F[Loop over exotic library<br/>Fit&#40;natural + template&#41;]
  F --> G[max TS = -2 ln L_nat/L_alt]
  G --> H[Gross&#8211;Vitells trials correction]
  H --> I[ScoreResult<br/>anomaly_score]
  I --> J[Ranking CSV / Parquet]
```

## Package layout

```
src/anomalymetric/
├── __init__.py              # top-level re-exports
├── spectrum.py              # Spectrum + SpectrumSeries
├── units.py                 # log10(E/eV) helpers; Hz/nm/eV conversions
├── forward/                 # response, exposure, distance/K-correction
├── models/                  # physical Models + exotic library + Fit
├── score/                   # loeb_turner, trials, ranking, bayes (stub)
├── cosmicray/               # CR factories + reference + knee/ankle + CR scoring
├── ingest/                  # loaders + entry-point plugin registry
├── pipeline.py              # orchestration
└── cli.py                   # Typer CLI
```

Things deliberately **not** in this tree:

- A separate `fit/` package. Fitting is a method on a model; merging into
  `models/inference.py` avoids circular imports.
- A `dataset.py` aggregator. The `(Spectrum, Model, ForwardModel, Fit)` tuple
  is enough — adding a `Dataset` wrapper before we have real instrument
  responses would be premature.
- Configuration files (YAML / TOML). Construct objects in Python.

For *why* the architecture is shaped this way, read
[`design-decisions.md`](design-decisions.md).
