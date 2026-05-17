# Cosmic rays

The cosmic-ray module reuses the same `Spectrum`, `Model`, `Fit`, and PLR
machinery as the photon path — the difference is the *reference spectrum*
(the natural-mixture stand-in) and the *exotic library*.

## All-particle reference

The PDG cosmic-ray review (Workman+ 2022) compiles the
"all-particle" spectrum: a broken power-law with breaks near the knee
(~3 PeV), the second knee, and the ankle (~5 EeV), suppressing above the GZK
cutoff (~5×10¹⁹ eV). `anomalymetric.cosmicray.spectrum.cr_reference_dnde`
returns this canonical shape:

```python
from anomalymetric.cosmicray.spectrum import cr_allparticle_grid, cr_reference_dnde
edges = cr_allparticle_grid(9, 21, bins_per_decade=10)
centers = 0.5 * (edges[:-1] + edges[1:])
dnde = cr_reference_dnde(centers)
```

## Fitting the spectrum

`anomalymetric.cosmicray.knee_ankle.TripleBrokenPowerLaw` is a three-break
power-law with free indices, break energies, and amplitude:

```python
from anomalymetric.cosmicray.knee_ankle import TripleBrokenPowerLaw
from anomalymetric.models.inference import Fit
from anomalymetric.cosmicray.spectrum import make_cr_spectrum

spec = make_cr_spectrum(edges, counts, exposure_cm2_s_sr)
model = TripleBrokenPowerLaw()
result = Fit(model, spec).run()
```

The `cr_reference_dnde` curve is what the broken-power-law fit recovers under
default Poisson noise — see
[`notebooks/03_cosmicray_module.ipynb`](https://github.com/your-org/anomalymetric/blob/main/notebooks/03_cosmicray_module.ipynb).

## CR-specific scoring

The unified `loeb_turner_score` with the default exotic library covers
photon-side anomalies. For *just* the cosmic-ray channel, use
`cosmicray.score.cr_score`:

```python
from anomalymetric.cosmicray.score import cr_score
result = cr_score(spec)
print(result.anomaly_score, result.best_template)
```

The natural hypothesis is the all-particle reference with one free
normalization; the exotic library is narrowed to two physically motivated
templates: a hard-cutoff power-law and a GZK-violating tail. See
[`cosmicray/score.py`](https://github.com/your-org/anomalymetric/blob/main/src/anomalymetric/cosmicray/score.py)
for the construction.

## Source backtracking

Cosmic rays above ~50 EeV deflect only by a few degrees in the galactic
magnetic field. The included `UniformFieldDeflection` helper gives a
small-angle estimate; a real backtracker is a `[cr]` extra and wraps
[CRPropa3](https://crpropa.desy.de/).

```python
from anomalymetric.cosmicray.propagation import UniformFieldDeflection
print(UniformFieldDeflection(B_uG=1.0, L_kpc=8.0).deflection_deg(1e20, Z=1))
# ~0.04 degrees
```

## Out of scope for v1

- Composition discrimination (mass-A inference from Xmax).
- Anisotropy / clustering analysis on the sky.
- Magnetic-rigidity reconstruction.
