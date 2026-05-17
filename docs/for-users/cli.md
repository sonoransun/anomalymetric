# CLI reference

All subcommands live under `anomalymetric` (installed by the package). They
take their input from any registered loader (CSV, FITS, Parquet, or an
entry-point plugin).

## `anomalymetric generate synthetic`

Generate a synthetic spectrum for testing or demos.

```
anomalymetric generate synthetic -o OUTPUT [--kind {blackbody,exotic_line}]
                                          [--t TEMPERATURE_K]
                                          [--line-ev EV]
                                          [--line-amplitude FLOAT]
                                          [--log-e-min FLOAT]
                                          [--log-e-max FLOAT]
                                          [--bins-per-decade INT]
                                          [--seed INT]
```

Examples:

```
# blackbody at 300 K, default grid (-3 to +6 in log10 eV)
anomalymetric generate synthetic --kind blackbody --t 300 -o /tmp/bb.parquet --seed 0

# same background with an injected 532 nm laser line
anomalymetric generate synthetic --kind exotic_line \
    --line-ev 2.331 --line-amplitude 5000 \
    -o /tmp/anom.parquet --seed 1
```

## `anomalymetric score`

Score one or more spectra against the default natural mixture and exotic
library, write a ranked CSV.

```
anomalymetric score INPUT [INPUT ...] [-o OUTPUT.csv]
```

Each row is one source; columns are `source_id, anomaly_score,
test_statistic, delta_log_likelihood, best_template, notes`.

## `anomalymetric fit`

Run an MLE fit of a single named model against a spectrum.

```
anomalymetric fit INPUT [--model {blackbody+powerlaw,blackbody,powerlaw}]
```

Prints the maximum-likelihood parameter values and the log-likelihood.
Useful for sanity-checking a natural-mixture choice before scoring.

## `anomalymetric query`

Stub for external-archive queries. Requires the `[archives]` extra
(`astroquery`). All four current backends (`vizier`, `heasarc`, `fermi_lat`,
`auger`) raise `NotImplementedError` and direct the user to the example
notebook — the goal of the v1 stubs is to fix the CLI contract.

## Exit codes

- `0` — success
- `1` — bad arguments (Typer's default)
- `2` — runtime error (loader failure, fit non-convergence, etc.)
