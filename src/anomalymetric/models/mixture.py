"""Additive mixture of `Model` components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anomalymetric.models.base import Parameters


class Mixture:
    """Sum of component dN/dE evaluations; parameters concatenated with prefixes."""

    def __init__(self, components: list, name: str = "mixture"):
        if not components:
            raise ValueError("Mixture requires at least one component")
        names = [c.name for c in components]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Mixture component names must be unique; duplicates would alias "
                f"their prefixed parameters. Duplicated: {dupes}"
            )
        self.name = name
        self.components = list(components)
        params: list = []
        for c in self.components:
            for p in c.parameters.params:
                p_prefixed = type(p)(
                    name=f"{c.name}.{p.name}",
                    value=p.value,
                    min=p.min,
                    max=p.max,
                    frozen=p.frozen,
                    scale=p.scale,
                )
                params.append(p_prefixed)
        self.parameters = Parameters(params)

    def _sync_components(self) -> None:
        # Mutating self.parameters does not mutate the original component
        # parameters; sync values back before evaluating.
        for c in self.components:
            for p in c.parameters.params:
                shared = self.parameters[f"{c.name}.{p.name}"]
                p.value = shared.value
                p.frozen = shared.frozen

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]:
        self._sync_components()
        total = np.zeros_like(np.asarray(log_energy_centers_eV, dtype=float))
        for c in self.components:
            total = total + c.dnde(log_energy_centers_eV)
        return total

    def component_dnde(
        self, log_energy_centers_eV: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        self._sync_components()
        return {c.name: c.dnde(log_energy_centers_eV) for c in self.components}
