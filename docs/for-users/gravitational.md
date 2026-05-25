# Gravitational differential

The gravitational module scores a **differential acceleration** PSD — the output
of a gradiometer, torsion balance, or drag-free two-test-mass accelerometer — for
anomalous lines or bumps above the instrument noise floor. Like the magnetometric
channel it is a continuous **Gaussian** PSD on the shared `log10(E/eV)` axis
(`E = hν`), not Poisson counts.

![Field-sensor channel coverage](../img/sensor-coverage.svg)

## The noise floor as the natural baseline

`gravitational.spectrum.grav_reference_psd` mixes three terms in (m/s²)²/Hz:
thermal (white), seismic (steep red below a corner), and Newtonian
gravity-gradient noise. `gravitational.score.GravNoiseFloor` scales it with one
free amplitude.

```python
from anomalymetric.gravitational.spectrum import gravimeter_band_grid, grav_reference_psd
import numpy as np
edges = gravimeter_band_grid(-5.0, 1.0, bins_per_decade=20)   # 10 µHz … 10 Hz
centers = 0.5 * (edges[:-1] + edges[1:])
floor = grav_reference_psd(np.log10(centers))
```

`gravitational.models` provides `TorsionBalanceNoise` (Eöt-Wash), 
`DifferentialAccelNoise` (MICROSCOPE drag-free), and `GravimeterNoise`.

## Scoring for differential signatures

```python
from anomalymetric.gravitational.score import grav_score
result = grav_score(spectrum)             # spectrum.kind == GRAVITATIONAL
print(result.anomaly_score, result.best_template)
```

The exotic library targets four "differential" signatures, each a narrow line at
an **in-band** frequency (amplitude free, center/width locked):

- **Equivalence-principle violation** — a line at the orbital/spin modulation
  frequency of a two-mass test (MICROSCOPE/Eöt-Wash). The fitted amplitude back-
  converts to an Eötvös parameter via `fifth_force.ep_eta_to_amplitude`.
- **Fifth force (Yukawa)** — a line at the source-modulation frequency; the
  Yukawa range sets coupling strength, not frequency.
- **Oscillating scalar dark matter** — a line at the field's Compton frequency
  `ν = m_φ c²/h` (`fifth_force.oscillating_dm_freq_hz`).
- **Broadband bumps** for unmodeled excess.

```python
from anomalymetric.gravitational.fifth_force import microscope_modulation_freq_hz
microscope_modulation_freq_hz()           # ≈ 1.68e-4 Hz (MICROSCOPE orbit)
```

## Generating synthetic data

```
anomalymetric generate gravimeter --kind noise_floor -o bg.parquet --seed 0
anomalymetric generate gravimeter --kind ep_violation --line-amplitude 8 -o sig.parquet --seed 1
anomalymetric score bg.parquet sig.parquet -o rank.csv
```

`--kind` also accepts `yukawa`, `osc_dm`, and `bump`. See
`notebooks/06_gravitational_differential.ipynb` for a worked example.

## Out of scope for v1

- Real strain / gravimeter backends (a `[grav]` extra stub).
- Calibrated amplitude → physics-parameter inference with full systematics.
- Cross-correlation of a detector network.
