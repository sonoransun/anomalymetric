"""MLE fitting + 1-D profile likelihood.

Likelihood is Poisson on expected counts per bin:

    log L = sum_i [ k_i log mu_i - mu_i - log(k_i!) ]

For upper-limit bins the detected term is replaced by the one-sided Poisson tail
`log P(N <= k_i | mu_i)` (the regularized upper incomplete gamma), which is the
probability the model is consistent with observing no more than the reported
limit. Gaussian (PSD) channels use the analogous one-sided half-normal penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaincc, gammaln

from anomalymetric.forward.base import ForwardModel
from anomalymetric.forward.response import IdentityResponse
from anomalymetric.spectrum import GAUSSIAN_KINDS, Spectrum, SpectrumKind, ValueKind


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
    # Upper-limit bins: use the proper one-sided Poisson tail, the probability of
    # observing no more than the reported limit `k_i` given the model mean `mu_i`:
    #   P(N <= k | mu) = Q(k+1, mu) = gammaincc(k+1, mu)   (regularized upper gamma).
    # This is ~0 penalty when mu << k, falls off smoothly above the limit, and is
    # well-defined at k=0 (gives log P = -mu) — unlike the old divide-by-k form.
    cdf = gammaincc(k[mask] + 1.0, mu[mask])
    ll_ul = np.sum(np.log(np.clip(cdf, 1e-300, 1.0)))
    return float(np.sum(contrib) + ll_ul)


def gaussian_log_likelihood(
    observed_value: NDArray[np.float64],
    expected_value: NDArray[np.float64],
    sigma: NDArray[np.float64],
    upper_limit_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Chi-square-style Gaussian log-likelihood for continuous PSD channels.

        log L = sum_i [ -0.5 ((y_i - mu_i) / sigma_i)^2 - log(sigma_i) - 0.5 log(2 pi) ]

    `sigma` is the per-bin standard deviation of the PSD estimate. Unlike the
    Poisson case the variance is not a function of the mean, so it must be
    supplied explicitly. Upper-limit bins use the same one-sided convention as
    `poisson_log_likelihood`: only predictions *above* the reported limit are
    penalized.
    """
    y = np.asarray(observed_value, dtype=float)
    mu = np.asarray(expected_value, dtype=float)
    s = np.clip(np.asarray(sigma, dtype=float), 1e-300, np.inf)
    norm = -np.log(s) - 0.5 * np.log(2.0 * np.pi)
    if upper_limit_mask is None:
        resid = (y - mu) / s
        return float(np.sum(-0.5 * resid * resid + norm))
    mask = np.asarray(upper_limit_mask, dtype=bool)
    detected = ~mask
    resid = (y[detected] - mu[detected]) / s[detected]
    ll_det = np.sum(-0.5 * resid * resid + norm[detected])
    excess = np.maximum(mu[mask] - y[mask], 0.0) / s[mask]
    ll_ul = np.sum(-0.5 * excess * excess)
    return float(ll_det + ll_ul)


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


def _resolve_sigma(spec: Spectrum) -> NDArray[np.float64]:
    """Per-bin Gaussian sigma (in PSD units) for a continuous sensor channel.

    The data must carry it via `spectrum.uncertainty`. If the values are an
    amplitude spectral density (ASD) and the uncertainty is on the ASD, propagate
    to PSD space (sigma_PSD = 2 * ASD * sigma_ASD).
    """
    if spec.uncertainty is None:
        raise ValueError(
            f"Gaussian channel {spec.kind.value!r} needs a per-bin sigma; set "
            "`spectrum.uncertainty` (the PSD estimate's standard deviation)."
        )
    sig = np.asarray(spec.uncertainty, dtype=float)
    if spec.value_kind is ValueKind.ASD_PER_BIN:
        return 2.0 * np.asarray(spec.value, dtype=float) * sig
    return sig


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
        self._is_gaussian = spectrum.kind in GAUSSIAN_KINDS
        if self._is_gaussian:
            self._observed = spectrum.canonical_value()
            self._sigma = _resolve_sigma(spectrum)
        else:
            self._observed = observed_counts_from_spectrum(spectrum)
            self._sigma = None
        self._log_likelihood = _log_likelihood_for(spectrum)
        self._edges = spectrum.log_energy_edges_eV
        self._centers = spectrum.energy_centers_eV
        self._log_centers = np.log10(self._centers)

    def predicted_value(self) -> NDArray[np.float64]:
        """Forward-folded model prediction: expected counts (Poisson) or PSD (Gaussian)."""
        source = self.model.dnde(self._log_centers)
        return self.forward.forward(self._edges, source)

    # Backwards-compatible alias (existing call sites / notebooks use this name).
    predicted_counts = predicted_value

    def neg_log_likelihood(
        self, u: NDArray[np.float64], scales: Optional[NDArray[np.float64]] = None
    ) -> float:
        # `u` is in scaled (preconditioned) optimizer space; `scales` maps it back
        # to physical units. `scales=None` is the identity (used by direct callers
        # such as `profile_likelihood`), so behavior is unchanged when scale==1.
        x = u if scales is None else np.asarray(u, dtype=float) * scales
        self.model.parameters.set_free_values(x)
        mu = self.predicted_value()
        if not np.all(np.isfinite(mu)):
            return 1e30
        # Poisson means must be non-negative; a Gaussian residual may be negative,
        # so only reject negative predictions for the Poisson channels.
        if not self._is_gaussian and np.any(mu < 0):
            return 1e30
        return -self._log_likelihood(
            self._observed, mu, self._sigma, self.spectrum.upper_limit_mask
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
        # Per-parameter preconditioning: the optimizer searches in u = x / scale so
        # parameters spanning many decades are well-conditioned. With scale==1 (all
        # current models) u == x and every path below is identical to the unscaled
        # case. Non-positive scales are treated as 1.0 (never divide a bound by 0).
        scales = np.array([p.scale if p.scale > 0 else 1.0 for p in free], dtype=float)
        x0 = self.model.parameters.free_values()
        bounds = self.model.parameters.free_bounds()
        u0 = x0 / scales
        bounds_u = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scales)]

        def obj(u: NDArray[np.float64]) -> float:
            return self.neg_log_likelihood(u, scales)

        # Nelder-Mead is gradient-free and robust to poorly-scaled parameters.
        # A few thousand iterations is ample for the low-dimensional mixtures here;
        # the matched-filter amplitude seed (see score.loeb_turner) keeps starts close.
        res = minimize(
            obj,
            u0,
            method=method,
            bounds=bounds_u,
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
        )
        # L-BFGS-B refinement only when Nelder-Mead did not already converge — it is
        # the expensive step on extreme surfaces (e.g. a giant injected line) and
        # buys nothing once the simplex has settled.
        refine_note = ""
        if not res.success:
            try:
                res2 = minimize(obj, res.x, method="L-BFGS-B", bounds=bounds_u)
                if res2.fun < res.fun:
                    res = res2
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                refine_note = f" (L-BFGS-B refinement skipped: {exc})"
        self.model.parameters.set_free_values(res.x * scales)
        return FitResult(
            success=bool(res.success),
            log_likelihood=-float(res.fun),
            parameter_values=self.model.parameters.values_dict(),
            message=str(res.message) + refine_note,
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
                # Re-fit the nuisance parameters to maximize logL at this fixed value.
                ll_curve[i] = self.run().log_likelihood
            return ll_curve
        finally:
            target.frozen = was_frozen


def _default_forward_for(spectrum: Spectrum) -> ForwardModel:
    if spectrum.kind is SpectrumKind.PHOTON:
        return IdentityResponse(exposure_cm2_s=1.0)
    if spectrum.kind in GAUSSIAN_KINDS:
        from anomalymetric.forward.sensor import PSDResponse

        return PSDResponse()
    from anomalymetric.forward.exposure import FlatExposure

    return FlatExposure(exposure_cm2_s=1.0, solid_angle_sr=1.0)


def _log_likelihood_for(spectrum: Spectrum):
    """Select the likelihood family for `spectrum.kind`.

    Returns a callable `(observed, expected, sigma, upper_limit_mask) -> float`
    so `Fit` and `likelihood()` share a single call shape across the Poisson
    (photon/CR) and Gaussian (sensor PSD) channels. Mirrors `_default_forward_for`.
    """
    if spectrum.kind in GAUSSIAN_KINDS:

        def _gauss(observed, expected, sigma, ul_mask):
            return gaussian_log_likelihood(observed, expected, sigma, upper_limit_mask=ul_mask)

        return _gauss

    def _pois(observed, expected, sigma, ul_mask):  # sigma is unused for Poisson
        return poisson_log_likelihood(observed, expected, upper_limit_mask=ul_mask)

    return _pois


def likelihood(spectrum: Spectrum, model, forward: ForwardModel | None = None) -> float:
    """Channel-dispatching likelihood entry point (used by the scorer)."""
    fit = Fit(model, spectrum, forward=forward)
    mu = fit.predicted_value()
    return fit._log_likelihood(
        fit._observed, mu, fit._sigma, spectrum.upper_limit_mask
    )
