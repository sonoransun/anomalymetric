# Extending models

Two contracts let you add new physics without touching the rest of the
package: the `Model` protocol and the `ExoticTemplate` factory.

## The `Model` protocol

From [`src/anomalymetric/models/base.py`](https://github.com/your-org/anomalymetric/blob/main/src/anomalymetric/models/base.py):

```python
class Model(Protocol):
    name: str
    parameters: Parameters
    def dnde(self, log_energy_centers_eV) -> NDArray[np.float64]: ...
```

A model is anything with:

1. `name`: a short string used as a parameter prefix inside `Mixture`.
2. `parameters`: a `Parameters` container (essentially a list of `Parameter`
   dataclasses with name, value, min/max, frozen flag).
3. `dnde(log_E)`: differential number flux in `1/(s cm² eV)` at each bin
   center (per-steradian for intrinsically per-sr models).

That is the whole contract. No subclassing required — duck typing through
`typing.Protocol`. The existing components (blackbody, power-law, Gaussian
line) are reference implementations.

## A minimal example

A non-thermal cutoff Gaussian — say, a 1.4 GHz pulsar peak modeled as a
Gaussian on a log-energy axis:

```python
from anomalymetric.models.base import Parameter, Parameters
import numpy as np

class PulsarPeak:
    def __init__(self, amplitude=1.0, E_center_eV=1.4e-6, sigma_dex=0.05):
        self.name = "pulsar_peak"
        self.parameters = Parameters([
            Parameter("amplitude", amplitude, min=0.0),
            Parameter("E_center_eV", E_center_eV, min=1e-9, max=1e22),
            Parameter("sigma_dex", sigma_dex, min=1e-3, max=2.0),
        ])

    def dnde(self, log_E):
        log_Ec = np.log10(self.parameters["E_center_eV"].value)
        sig = self.parameters["sigma_dex"].value
        A = self.parameters["amplitude"].value
        E = 10.0**log_E
        gauss = (1.0/(np.sqrt(2*np.pi)*sig)) * np.exp(-0.5*((log_E-log_Ec)/sig)**2)
        return A * gauss / (E * np.log(10.0))
```

That's the entire integration. Drop it into a `Mixture` or pass it to `Fit`.

## Adding to the exotic library

Templates are factories that produce a frozen-shape `Model`. Example: a new
4-eV axion line.

```python
from anomalymetric.models.exotic import ExoticTemplate, axion_line_template

my_axion = axion_line_template("4eV_axion", 4.0)
# my_axion.factory()  -> a GaussianLine with center=2.0 eV, frozen shape
```

To use a custom library, build a list and pass it explicitly:

```python
from anomalymetric.score.loeb_turner import loeb_turner_score
custom_library = [my_axion, *default_library()]
score = loeb_turner_score(spectrum, natural, exotic_library=custom_library,
                          n_trials=len(custom_library))
```

## `Mixture` parameter sharing

`Mixture` makes a *copy* of each component's parameter list with a
`{name}.{param}` prefix. Mutating the mixture's parameters does *not* feed
back to the original component until `_sync_components()` runs (which is
called inside `dnde` and `component_dnde`). So if you write code that mutates
both, prefer to mutate via the mixture.

## PSD / sensor-channel models (Gaussian)

For the magnetometric and gravitational channels the *same* `Model` protocol
applies, but `dnde(log_E)` is reinterpreted to return a **predicted power
spectral density** `S(f)` at `f = 10**log_E / h` (use
`anomalymetric.units.frequencies_hz`), not a `dN/dE`. PSDs of independent
processes add linearly, so `Mixture` is still correct: a noise-floor model plus
a signal template sum in power. The likelihood is selected by the spectrum's
`SpectrumKind` (`MAGNETOMETRIC` / `GRAVITATIONAL` → Gaussian; everything else →
Poisson), and the Gaussian channel reads its per-bin σ from
`Spectrum.uncertainty`, so a PSD model needs no extra plumbing:

```python
from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.units import frequencies_hz

class MySensorNoise:
    def __init__(self, amplitude=1.0, S_white=1e-12, f_knee_hz=1.0):
        self.name = "my_sensor_noise"
        self._S_white, self._f_knee = S_white, f_knee_hz
        self.parameters = Parameters([Parameter("amplitude", amplitude, min=0.0)])

    def dnde(self, log_E):  # returns S(f), a PSD
        f = frequencies_hz(log_E)
        A = self.parameters["amplitude"].value
        return A * self._S_white * (1.0 + self._f_knee / f)
```

Narrow signal lines reuse `models.exotic.psd_line_template(name, E_center_eV,
width_dex)` (peak-height-parameterized — its amplitude is the line height, which
keeps the optimizer well-conditioned regardless of the channel's absolute scale).
See `magnetometric/models.py` and `gravitational/models.py` for the shipped noise
floors.

## When you need real physics

For physically-motivated non-thermal models — synchrotron, inverse Compton,
π⁰ decay — install the `[naima]` extra and wrap a `naima` model:

```python
import naima
from astropy import units as u

class NaimaSynchrotron:
    def __init__(self, ...):
        self._naima = naima.models.Synchrotron(...)
        self.name = "synchrotron"
        self.parameters = Parameters([...])
    def dnde(self, log_E):
        E = 10.0**log_E * u.eV
        return self._naima.flux(E, distance=...).to(1/u.s/u.cm**2/u.eV).value
```

The wrapper keeps the gammapy-inspired API stable; only the inside of
`dnde` changes.

## Fitting

Anything implementing the `Model` protocol works with `Fit`:

```python
from anomalymetric.models.inference import Fit
result = Fit(my_model, spectrum).run()
print(result.parameter_values)
```

The default optimizer is Nelder-Mead (with an L-BFGS-B refinement only if it
fails to converge), searching in `x / Parameter.scale` space. For parameters
spanning many decades, set `Parameter.scale` to the parameter's natural
magnitude so the search is well-conditioned (the default `scale=1.0` leaves
behavior unchanged).
