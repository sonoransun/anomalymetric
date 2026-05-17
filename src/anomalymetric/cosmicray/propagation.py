"""Cosmic-ray source back-tracking through the galactic magnetic field.

A real backtracker is intricate; for v1 we wrap `CRPropa3` (heavy optional
dependency) and ship a minimal closed-form deflection estimate that works for
a uniform field. Use this only to demonstrate the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformFieldDeflection:
    """Larmor deflection in a uniform B field over path length `L`.

    deflection_angle ≈ Z e B L / E   (small-angle, ultra-relativistic).
    """

    B_uG: float = 1.0  # microgauss
    L_kpc: float = 8.0

    def deflection_deg(self, E_eV: float, Z: int = 1) -> float:
        E_EeV = E_eV / 1.0e18
        if E_EeV <= 0:
            raise ValueError("Energy must be positive.")
        # Standard estimate: theta ~ 0.5° * Z * (B/uG) * (L/kpc) / (E/EeV)
        return 0.5 * Z * self.B_uG * self.L_kpc / E_EeV


def crpropa_backtrack(*args, **kwargs):  # pragma: no cover - optional dep
    try:
        import crpropa  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cosmicray.propagation.crpropa_backtrack requires the `[cr]` extra "
            "(pip install anomalymetric[cr])."
        ) from exc
    raise NotImplementedError(
        "CRPropa3 backtracking is stubbed for v1; see notebooks/03_cosmicray_module.ipynb."
    )
