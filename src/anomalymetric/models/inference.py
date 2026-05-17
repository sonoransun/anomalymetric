"""MLE fitting + 1-D profile likelihood.

Likelihood is Poisson on expected counts per bin:

    log L = sum_i [ k_i log mu_i - mu_i - log(k_i!) ]

For upper-limit bins we substitute the Feldman-Cousins-style profile-likelihood
upper-limit convention: treat the bin as Poisson with `k_i = 0` and add a
penalty `mu_i / sigma_UL_i` only if `mu_i > sigma_UL_i`. v1 keeps this simple;
real censored-data handling is a known follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

from anomalymetric.forward.base import ForwardModel
from anomalymetric.forward.response import IdentityResponse
from anomalymetric.spectrum import Spectrum, SpectrumKind, ValueKind


def poisson_log_likelihood(
    observed_counts: NDArray[np.float64],
    expected_counts: NDArray[np.float64],
    upper_limit_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    k = np.asarray(observed_counts, dtype=float)
    mu = np.clip(np.asarray(expected_counts, dtype=float), 1e-300, np.inf)
    if upper_limit_mask is None:
        contrib = k * np.log(mu) - mu - gammaln(k + 1.0)
        return float(np.sum(contrib))
    mask = np.asarray(upper_limit_mask, dtype=bool)
    detected = ~mask
    contrib = (
        k[detected] * np.log(mu[detected]) - mu[detected] - gammaln(k[detected] + 1.0)
    )
    # For UL bins use a one-sided penalty (Rolke-style) — penalize predictions
    # above the reported upper limit `k_i`.
    excess = np.maximum(mu[mask] - k[mask], 0.0)
    penalty = -0.5 * (excess / np.clip(k[mask], 1e-12, np.inf)) ** 2
    return float(np.sum(contrib) + np.sum(penalty))


@dataclass
class FitResult:
    success: bool
    log_likelihood: float
    parameter_values: dict[str, float]
    message: str = ""
    n_iter: int = 0


def observed_counts_from_spectrum(
    spec: Spectrum, exposure_cm2_s: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    """Convert any Spectrum to Poisson-friendly observed counts per bin."""
    if spec.value_kind is ValueKind.COUNTS_PER_BIN:
        return spec.value.copy()
    # For dN/dE inputs, attach the spectrum's recorded exposure (if any).
    return spec.expected_counts(exposure_cm2_s=exposure_cm2_s)


class Fit:
    """Minimize -log L over the free parameters of a `Model`."""

    def __init__(
        self,
        model,
        spectrum: Spectrum,
        forward: ForwardModel | None = None,
    ):
        self.model = model
        self.spectrum = spectrum
        if forward is None:
            forward = _default_forward_for(spectrum)
        self.forward = forward
        self._observed = observed_counts_from_spectrum(spectrum)
        self._edges = spectrum.log_energy_edges_eV
        self._centers = spectrum.energy_centers_eV
        self._log_centers = np.log10(self._centers)

    def predicted_counts(self) -> NDArray[np.float64]:
        dnde = self.model.dnde(self._log_centers)
        return self.forward.forward(self._edges, dnde)

    def neg_log_likelihood(self, x: NDArray[np.float64]) -> float:
        self.model.parameters.set_free_values(x)
        mu = self.predicted_counts()
        if not np.all(np.isfinite(mu)) or np.any(mu < 0):
            return 1e30
        return -poisson_log_likelihood(
            self._observed, mu, upper_limit_mask=self.spectrum.upper_limit_mask
        )

    def run(self, method: str = "Nelder-Mead") -> FitResult:
        free = self.model.parameters.free
        if not free:
            ll = -self.neg_log_likelihood(np.array([]))
            return FitResult(
                success=True,
                log_likelihood=ll,
                parameter_values=self.model.parameters.values_dict(),
                message="no free parameters",
            )
        x0 = self.model.parameters.free_values()
        bounds = self.model.parameters.free_bounds()
        # Nelder-Mead is gradient-free and robust to poorly-scaled parameters.
        # L-BFGS-B can run as a refinement step but tends to fall into local
        # basins when blackbody T and amplitude have wildly different scales.
        res = minimize(
            self.neg_log_likelihood,
            x0,
            method=method,
            bounds=bounds,
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000},
        )
        # Refinement: L-BFGS-B from the Nelder-Mead optimum for sharper convergence.
        try:
            res2 = minimize(
                self.neg_log_likelihood,
                res.x,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res2.fun < res.fun:
                res = res2
        except Exception:
            pass
        self.model.parameters.set_free_values(res.x)
        return FitResult(
            success=bool(res.success),
            log_likelihood=-float(res.fun),
            parameter_values=self.model.parameters.values_dict(),
            message=str(res.message),
            n_iter=int(getattr(res, "nit", 0)),
        )

    def profile_likelihood(
        self, parameter_name: str, scan: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Profile log L over `parameter_name`: maximize over all *other* free params."""
        target = self.model.parameters[parameter_name]
        was_frozen = target.frozen
        target.frozen = True
        try:
            ll_curve = np.empty_like(scan, dtype=float)
            for i, v in enumerate(scan):
                target.value = float(v)
                ll_curve[i] = -self.neg_log_likelihood(self.model.parameters.free_values())
                # Re-fit nuisance to maximize for this fixed value.
                res = self.run()
                ll_curve[i] = res.log_likelihood
            return ll_curve
        finally:
            target.frozen = was_frozen


def _default_forward_for(spectrum: Spectrum) -> ForwardModel:
    if spectrum.kind in {SpectrumKind.PHOTON}:
        return IdentityResponse(exposure_cm2_s=1.0)
    from anomalymetric.forward.exposure import FlatExposure

    return FlatExposure(exposure_cm2_s=1.0, solid_angle_sr=1.0)


def likelihood(spectrum: Spectrum, model, forward: ForwardModel | None = None) -> float:
    """Channel-dispatching likelihood entry point (used by the scorer)."""
    fit = Fit(model, spectrum, forward=forward)
    mu = fit.predicted_counts()
    return poisson_log_likelihood(
        fit._observed, mu, upper_limit_mask=spectrum.upper_limit_mask
    )
