"""Spectrum dataclass + canonicalization to expected counts per bin.

Why a `value_kind` enum: "differential flux" means at least three incompatible
things across radio, X/gamma, and cosmic-ray astronomy. We refuse to guess —
ingest declares `value_kind` and the library converts to a single canonical
representation (`photon_rate_per_bin` for photons; an analogous count-per-bin
for CR/neutrino once exposure is known). Likelihoods downstream are Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from anomalymetric.units import bin_centers_eV, bin_widths_eV


class ValueKind(str, Enum):
    DNDE = "dNdE"
    EDNDE = "EdNdE"
    E2DNDE = "E2dNdE"
    NUFNU = "nuFnu"
    PHOTON_RATE_PER_BIN = "photon_rate_per_bin"
    COUNTS_PER_BIN = "counts_per_bin"


class SpectrumKind(str, Enum):
    PHOTON = "photon"
    CR_PROTON = "cr_proton"
    CR_NUCLEUS = "cr_nucleus"
    CR_ALLPARTICLE = "cr_allparticle"
    NEUTRINO = "neutrino"


@dataclass
class Spectrum:
    """A binned spectrum on a log10(E/eV) axis.

    Use `Spectrum.from_dnde(...)` etc. for ingest; downstream code consumes
    `expected_counts(...)` which canonicalizes to per-bin counts given an
    exposure (or assumes unit exposure for a forward-folded predicted-rate
    spectrum).
    """

    log_energy_edges_eV: NDArray[np.float64]
    value: NDArray[np.float64]
    value_kind: ValueKind
    kind: SpectrumKind = SpectrumKind.PHOTON
    uncertainty: Optional[NDArray[np.float64]] = None
    upper_limit_mask: Optional[NDArray[np.bool_]] = None
    per_steradian: bool = False
    solid_angle_sr: Optional[float] = None
    exposure_cm2_s: Optional[NDArray[np.float64]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.log_energy_edges_eV = np.asarray(self.log_energy_edges_eV, dtype=float)
        self.value = np.asarray(self.value, dtype=float)
        if self.log_energy_edges_eV.ndim != 1:
            raise ValueError("log_energy_edges_eV must be 1-D")
        if self.value.shape[0] != self.log_energy_edges_eV.shape[0] - 1:
            raise ValueError(
                f"value has {self.value.shape[0]} bins but "
                f"{self.log_energy_edges_eV.shape[0]} edges were given "
                "(expected len(edges) = len(value) + 1)."
            )
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=float)
            if self.uncertainty.shape != self.value.shape:
                raise ValueError("uncertainty shape must match value shape")
        if self.upper_limit_mask is not None:
            self.upper_limit_mask = np.asarray(self.upper_limit_mask, dtype=bool)
            if self.upper_limit_mask.shape != self.value.shape:
                raise ValueError("upper_limit_mask shape must match value shape")
        if self.per_steradian and self.solid_angle_sr is None:
            # Tolerated for CR all-particle spectra where the field is unitful but
            # solid angle is implicit; downstream code that needs it will raise.
            pass

    @property
    def n_bins(self) -> int:
        return int(self.value.shape[0])

    @property
    def energy_centers_eV(self) -> NDArray[np.float64]:
        return bin_centers_eV(self.log_energy_edges_eV)

    @property
    def bin_widths_eV(self) -> NDArray[np.float64]:
        return bin_widths_eV(self.log_energy_edges_eV)

    def as_dnde(self) -> NDArray[np.float64]:
        """Return differential number flux dN/dE in 1/(s cm^2 eV) (per sr if applicable)."""
        E = self.energy_centers_eV
        v = self.value
        if self.value_kind is ValueKind.DNDE:
            return v
        if self.value_kind is ValueKind.EDNDE:
            return v / E
        if self.value_kind is ValueKind.E2DNDE:
            return v / (E * E)
        if self.value_kind is ValueKind.NUFNU:
            # nu F_nu [erg/s/cm^2] -> E dN/dE photon flux requires dividing by E^2
            # then converting erg->eV; nuFnu == E^2 dN/dE in energy-flux form.
            erg_per_eV = 1.602176634e-12
            return (v / erg_per_eV) / (E * E)
        if self.value_kind is ValueKind.PHOTON_RATE_PER_BIN:
            return v / self.bin_widths_eV
        if self.value_kind is ValueKind.COUNTS_PER_BIN:
            if self.exposure_cm2_s is None:
                raise ValueError(
                    "COUNTS_PER_BIN spectra need `exposure_cm2_s` to convert to dN/dE."
                )
            return v / self.bin_widths_eV / self.exposure_cm2_s
        raise ValueError(f"Unhandled value_kind {self.value_kind!r}")

    def expected_counts(
        self, exposure_cm2_s: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Canonical Poisson-friendly representation.

        For a forward-folded *predicted* spectrum we expect the caller to have
        already multiplied by exposure (no extra factor here). For an observed
        differential flux, pass `exposure_cm2_s` (per bin) and the conversion is
        `mu_i = dN/dE_i * dE_i * A_i * T_i`.
        """
        if self.value_kind is ValueKind.COUNTS_PER_BIN:
            return self.value.copy()
        if self.value_kind is ValueKind.PHOTON_RATE_PER_BIN:
            if exposure_cm2_s is None:
                return self.value.copy()
            return self.value * np.asarray(exposure_cm2_s, dtype=float)

        dnde = self.as_dnde()
        widths = self.bin_widths_eV
        if exposure_cm2_s is None:
            if self.exposure_cm2_s is None:
                # No exposure available — return per-bin number flux (rate) and let
                # downstream code attach exposure. This matches the predicted-rate
                # convention used by Model.predict().
                return dnde * widths
            exposure_cm2_s = self.exposure_cm2_s
        return dnde * widths * np.asarray(exposure_cm2_s, dtype=float)

    @classmethod
    def from_dnde(
        cls,
        log_energy_edges_eV: NDArray[np.float64],
        dnde: NDArray[np.float64],
        *,
        kind: SpectrumKind = SpectrumKind.PHOTON,
        uncertainty: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> "Spectrum":
        return cls(
            log_energy_edges_eV=log_energy_edges_eV,
            value=np.asarray(dnde, dtype=float),
            value_kind=ValueKind.DNDE,
            kind=kind,
            uncertainty=uncertainty,
            **kwargs,
        )

    @classmethod
    def from_counts(
        cls,
        log_energy_edges_eV: NDArray[np.float64],
        counts: NDArray[np.float64],
        exposure_cm2_s: NDArray[np.float64],
        *,
        kind: SpectrumKind = SpectrumKind.PHOTON,
        **kwargs: Any,
    ) -> "Spectrum":
        return cls(
            log_energy_edges_eV=log_energy_edges_eV,
            value=np.asarray(counts, dtype=float),
            value_kind=ValueKind.COUNTS_PER_BIN,
            kind=kind,
            exposure_cm2_s=np.asarray(exposure_cm2_s, dtype=float),
            **kwargs,
        )


@dataclass
class SpectrumSeries:
    """Multi-epoch container — the Loeb–Turner case leans on temporal behavior."""

    epochs_mjd: NDArray[np.float64]
    spectra: list[Spectrum]

    def __post_init__(self) -> None:
        self.epochs_mjd = np.asarray(self.epochs_mjd, dtype=float)
        if len(self.spectra) != self.epochs_mjd.shape[0]:
            raise ValueError("epochs_mjd and spectra must align 1:1")

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self):
        return iter(self.spectra)
