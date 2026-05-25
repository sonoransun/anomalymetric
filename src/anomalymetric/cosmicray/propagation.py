"""Cosmic-ray source back-tracking through the galactic magnetic field.

A real backtracker is intricate; for v1 we wrap `CRPropa3` (heavy optional
dependency) and ship a minimal closed-form deflection estimate that works for
a uniform field. Use this only to demonstrate the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktrackResult:
    """Outcome of a single back-tracked cosmic ray."""

    arrival_direction_deg: tuple[float, float]  # (lon, lat) at the field boundary
    total_deflection_deg: float
    rigidity_EV: float


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


def crpropa_backtrack(
    energy_eV: float,
    charge_Z: int,
    start_lonlat_deg: tuple[float, float],
    *,
    field: str = "JF12",
    max_path_kpc: float = 20.0,
    step_kpc: float = 0.1,
) -> BacktrackResult:  # pragma: no cover - optional heavy dep (crpropa)
    """Back-track a cosmic ray through a Galactic field model with CRPropa3.

    Propagates the charge-reversed candidate from `start_lonlat_deg` outward to the
    `max_path_kpc` boundary and reports the entry direction and total deflection.
    Requires the `[cr]` extra (CRPropa3, a compiled C++/SWIG package).
    """
    try:
        import crpropa
    except ImportError as exc:
        raise ImportError(
            "cosmicray.propagation.crpropa_backtrack requires the `[cr]` extra "
            "(pip install anomalymetric[cr])."
        ) from exc

    if field != "JF12":
        raise ValueError(f"only the JF12 field model is wired in v1, got {field!r}")

    lon0, lat0 = (np.deg2rad(a) for a in start_lonlat_deg)
    bfield = crpropa.JF12Field()
    sim = crpropa.ModuleList()
    sim.add(crpropa.PropagationCK(bfield, 1e-4, step_kpc * crpropa.kpc, step_kpc * crpropa.kpc))
    sim.add(crpropa.MaximumTrajectoryLength(max_path_kpc * crpropa.kpc))

    direction = crpropa.Vector3d(np.cos(lat0) * np.cos(lon0),
                                 np.cos(lat0) * np.sin(lon0), np.sin(lat0))
    # Reverse the charge to back-track to the source side.
    pid = crpropa.nucleusId(1, -int(charge_Z)) if charge_Z else -2212
    cosmic_ray = crpropa.Candidate(pid, energy_eV * crpropa.eV,
                                   crpropa.Vector3d(0, 0, 0), direction)
    sim.run(cosmic_ray, True)

    final_dir = cosmic_ray.current.getDirection()
    lon = np.rad2deg(np.arctan2(final_dir.y, final_dir.x))
    lat = np.rad2deg(np.arcsin(np.clip(final_dir.z, -1.0, 1.0)))
    cos_sep = float(np.clip(final_dir.dot(direction), -1.0, 1.0))
    deflection = float(np.rad2deg(np.arccos(cos_sep)))
    rigidity_EV = energy_eV / max(abs(int(charge_Z)), 1) / 1.0e18
    return BacktrackResult((float(lon), float(lat)), deflection, rigidity_EV)
