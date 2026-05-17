"""Model protocol + a `Parameters` container.

Naming and shape deliberately mirror gammapy's `Model` / `Parameters` so the
package can later be retargeted onto gammapy without breaking user code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class Parameter:
    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    frozen: bool = False
    scale: float = 1.0  # for optimizer pre-conditioning

    def clamp(self, x: float) -> float:
        return float(min(max(x, self.min), self.max))


@dataclass
class Parameters:
    params: list[Parameter] = field(default_factory=list)

    def __iter__(self):
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, key: str | int) -> Parameter:
        if isinstance(key, int):
            return self.params[key]
        for p in self.params:
            if p.name == key:
                return p
        raise KeyError(key)

    @property
    def free(self) -> list[Parameter]:
        return [p for p in self.params if not p.frozen]

    def free_values(self) -> NDArray[np.float64]:
        return np.array([p.value for p in self.free], dtype=float)

    def set_free_values(self, x: Iterable[float]) -> None:
        for p, v in zip(self.free, x):
            p.value = p.clamp(float(v))

    def free_bounds(self) -> list[tuple[float, float]]:
        return [(p.min, p.max) for p in self.free]

    def values_dict(self) -> dict[str, float]:
        return {p.name: p.value for p in self.params}


class Model(Protocol):
    """A physical source model evaluated on a log-energy axis.

    Implementations must:
      * own a `Parameters` instance under `.parameters`
      * implement `dnde(log_energy_centers_eV)` returning per-bin dN/dE
        in 1/(s cm^2 eV) (per-sr if the model is intrinsically per-sr).
    """

    name: str
    parameters: Parameters

    def dnde(self, log_energy_centers_eV: NDArray[np.float64]) -> NDArray[np.float64]: ...
