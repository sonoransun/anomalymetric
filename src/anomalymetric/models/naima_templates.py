"""naima-backed non-thermal SED models as exotic templates (needs the `[naima]` extra).

`naima` (https://naima.readthedocs.io) provides physical synchrotron, inverse-
Compton, and pion-decay radiative models for a parent particle population. The
`NaimaModel` adapter exposes one as an anomalymetric `Model` (a `dnde(log_E)`
returning 1/(s cm^2 eV)), with a single free dimensionless amplitude so it slots
into the matched-filter `ExoticTemplate` library.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameter, Parameters
from anomalymetric.models.exotic import ExoticTemplate


def _require_naima():
    try:
        import naima  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "models.naima_templates requires the `[naima]` extra "
            "(pip install anomalymetric[naima])."
        ) from exc
    return naima


def _to_dnde_per_eV(flux, energy) -> NDArray[np.float64]:
    """Convert a naima flux Quantity to a plain 1/(s cm^2 eV) array.

    naima's `.flux(...)` returns a differential photon flux dN/dE; if a model hands
    back an SED (E^2 dN/dE) instead, divide by E^2. Pure unit handling — testable
    without running a radiative model.
    """
    import astropy.units as u

    target = 1.0 / (u.eV * u.s * u.cm**2)
    try:
        return np.asarray(flux.to(target).value, dtype=float)
    except u.UnitConversionError:
        sed = flux.to(u.erg / u.s / u.cm**2)
        return np.asarray((sed / (energy**2)).to(target).value, dtype=float)


class NaimaModel:
    """Adapter wrapping a naima radiative model as an anomalymetric `Model`."""

    def __init__(self, naima_model, name: str, distance_kpc: float = 1.0):
        self._m = naima_model
        self.name = name
        self._distance_kpc = distance_kpc
        self.parameters = Parameters([Parameter("amplitude", 1.0, min=0.0)])

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        import astropy.units as u

        energy = (10.0 ** np.asarray(log_energy_centers_eV, dtype=float)) * u.eV
        flux = self._m.flux(energy, distance=self._distance_kpc * u.kpc)
        return self.parameters["amplitude"].value * _to_dnde_per_eV(flux, energy)


def _ecpl(naima):
    import astropy.units as u

    return naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e33 / u.eV, e_0=1.0 * u.TeV, alpha=2.1, e_cutoff=20.0 * u.TeV
    )


def naima_synchrotron_template(label: str = "default", B_uG: float = 100.0) -> ExoticTemplate:
    def make() -> object:
        naima = _require_naima()
        import astropy.units as u

        model = naima.models.Synchrotron(_ecpl(naima), B=B_uG * u.uG)
        return NaimaModel(model, name=f"naima.synchrotron.{label}")

    return ExoticTemplate(name=f"naima.synchrotron.{label}", factory=make)


def naima_inverse_compton_template(label: str = "default") -> ExoticTemplate:
    def make() -> object:
        naima = _require_naima()

        model = naima.models.InverseCompton(_ecpl(naima), seed_photon_fields=["CMB"])
        return NaimaModel(model, name=f"naima.ic.{label}")

    return ExoticTemplate(name=f"naima.ic.{label}", factory=make)


def naima_pion_decay_template(label: str = "default", nh_per_cm3: float = 1.0) -> ExoticTemplate:
    def make() -> object:
        naima = _require_naima()
        import astropy.units as u

        model = naima.models.PionDecay(_ecpl(naima), nh=nh_per_cm3 * u.cm**-3)
        return NaimaModel(model, name=f"naima.pion.{label}")

    return ExoticTemplate(name=f"naima.pion.{label}", factory=make)


def naima_library() -> list[ExoticTemplate]:
    """A small non-thermal SED library (synchrotron + IC + pion decay)."""
    return [
        naima_synchrotron_template(),
        naima_inverse_compton_template(),
        naima_pion_decay_template(),
    ]
