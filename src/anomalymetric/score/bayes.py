"""Bayes-factor scoring — a model-comparison alternative to the PLR score.

Three methods, all reusing the existing `Fit` machinery and returning the same
`BayesResult`:

* ``"bic"`` (default, no extra deps): the BIC/Schwarz approximation
  ``ln B ≈ Δln L − ½ Δk ln N``. Each template adds one free amplitude, so this is
  the PLR test statistic penalized by a model-complexity term. Robust and cheap.
* ``"laplace"`` (no extra deps): BIC plus a 1-D Occam factor from the curvature of
  the amplitude likelihood at the MLE — only when a finite ``amplitude_prior_width``
  is supplied (the templates' amplitudes are otherwise improper, ``max=inf``);
  falls back to BIC with a note otherwise.
* ``"dynesty"`` (needs the ``[bayes]`` extra): nested-sampling evidence ratio
  ``ln Z_alt − ln Z_nat``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.spectrum import Spectrum


@dataclass
class TemplateBayes:
    template_name: str
    log_bayes_factor: float
    delta_log_likelihood: float
    delta_k: int


@dataclass
class BayesResult:
    log_bayes_factor: float  # max over templates
    best_template: str
    per_template: list[TemplateBayes] = field(default_factory=list)
    natural_parameters: dict[str, float] = field(default_factory=dict)
    method: str = "bic"
    notes: str = ""


def bayes_factor(
    spectrum: Spectrum,
    natural_components: list,
    *,
    forward: Optional[ForwardModel] = None,
    exotic_library=None,
    method: str = "bic",
    amplitude_prior_width: Optional[float] = None,
    nlive: int = 250,
) -> BayesResult:
    """Compare each exotic template against the natural mixture by Bayes factor."""
    if method == "dynesty":
        return _bayes_factor_dynesty(
            spectrum, natural_components, forward=forward,
            exotic_library=exotic_library, nlive=nlive,
        )
    if method not in ("bic", "laplace"):
        raise ValueError(f"Unknown bayes method {method!r}; use bic|laplace|dynesty.")

    from anomalymetric.models.exotic import default_library
    from anomalymetric.score.loeb_turner import loeb_turner_score

    if exotic_library is None:
        exotic_library = default_library()
    score = loeb_turner_score(
        spectrum, natural_components, forward=forward,
        exotic_library=exotic_library, n_trials=1,
    )
    n_bins = spectrum.n_bins
    notes = ""
    use_laplace = method == "laplace" and amplitude_prior_width is not None
    if method == "laplace" and amplitude_prior_width is None:
        notes = "laplace requested without amplitude_prior_width; fell back to BIC."

    per: list[TemplateBayes] = []
    for tscore, template in zip(score.per_template, exotic_library):
        delta_k = len(template.factory().parameters.free)
        dll = tscore.delta_log_likelihood
        if use_laplace:
            occam = _laplace_occam(
                spectrum, natural_components, template, forward,
                amplitude_prior_width, tscore.amplitude, score.natural_parameters,
            )
            ln_b = dll + occam if np.isfinite(occam) else dll - 0.5 * delta_k * np.log(n_bins)
        else:
            ln_b = dll - 0.5 * delta_k * np.log(n_bins)
        per.append(TemplateBayes(tscore.template_name, float(ln_b), float(dll), int(delta_k)))

    best = max(per, key=lambda t: t.log_bayes_factor)
    return BayesResult(
        log_bayes_factor=best.log_bayes_factor,
        best_template=best.template_name,
        per_template=per,
        natural_parameters=score.natural_parameters,
        method="laplace" if use_laplace else "bic",
        notes=notes or f"N_bins = {n_bins}",
    )


def _laplace_occam(spectrum, natural_components, template, forward, prior_width,
                   amp_mle, nat_param_values) -> float:
    """1-D Laplace Occam factor for the template amplitude at its MLE.

    ``ln B_laplace = Δln L + ½ ln(2π) − ½ ln I_amp − ln(prior_width)`` where
    ``I_amp`` is the curvature (Fisher information) of −ln L in the amplitude. Returns
    NaN on a degenerate/boundary curvature so the caller falls back to BIC.
    """
    from anomalymetric.models.inference import Fit
    from anomalymetric.models.mixture import Mixture
    from anomalymetric.score.loeb_turner import _clone

    alt = Mixture([_clone(m) for m in natural_components] + [template.factory()],
                  name=f"alt.{template.name}")
    for name, val in nat_param_values.items():
        if name in [p.name for p in alt.parameters.params]:
            alt.parameters[name].value = val
    amp_name = f"{template.name}.amplitude"
    if amp_name not in [p.name for p in alt.parameters.params]:
        return float("nan")
    alt.parameters[amp_name].value = max(float(amp_mle), 0.0)
    fit = Fit(alt, spectrum, forward=forward)
    free_names = [p.name for p in alt.parameters.free]
    if amp_name not in free_names:
        return float("nan")
    j = free_names.index(amp_name)
    x = alt.parameters.free_values()
    a0 = x[j]
    h = max(abs(a0) * 1e-3, 1e-9)

    def nll(amp: float) -> float:
        xx = x.copy()
        xx[j] = amp
        return fit.neg_log_likelihood(xx)

    # Central difference, but at the amplitude>=0 boundary use a forward stencil.
    if a0 - h <= 0.0:
        curv = (nll(a0 + 2 * h) - 2 * nll(a0 + h) + nll(a0)) / h**2
    else:
        curv = (nll(a0 + h) - 2 * nll(a0) + nll(a0 - h)) / h**2
    if not np.isfinite(curv) or curv <= 0:
        return float("nan")
    return 0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(curv) - np.log(prior_width)


def _bayes_factor_dynesty(spectrum, natural_components, *, forward=None,
                          exotic_library=None, nlive=250) -> BayesResult:
    """Nested-sampling evidence ratio ln Z_alt − ln Z_nat (needs the [bayes] extra)."""
    try:
        from dynesty import NestedSampler
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "score.bayes method='dynesty' requires the `[bayes]` extra "
            "(pip install anomalymetric[bayes])."
        ) from exc

    from anomalymetric.models.exotic import default_library
    from anomalymetric.models.inference import Fit
    from anomalymetric.models.mixture import Mixture
    from anomalymetric.score.loeb_turner import _clone

    if exotic_library is None:
        exotic_library = default_library()

    def _evidence(model) -> float:
        fit = Fit(model, spectrum, forward=forward)
        free = model.parameters.free
        if not free:
            return -fit.neg_log_likelihood(np.array([]))
        bounds = model.parameters.free_bounds()
        scale = float(np.max(np.abs(fit._observed))) or 1.0
        lo = np.array([b[0] if np.isfinite(b[0]) else 0.0 for b in bounds])
        hi = np.array([b[1] if np.isfinite(b[1]) else scale * 1e3 for b in bounds])

        def loglike(u):
            return -fit.neg_log_likelihood(np.asarray(u, dtype=float))

        def ptransform(cube):
            return lo + np.asarray(cube) * (hi - lo)

        sampler = NestedSampler(loglike, ptransform, len(free), nlive=nlive)
        # Bounded stopping so the run always terminates promptly; dlogz=0.5 is loose
        # but ample for a Bayes-factor estimate, and maxiter caps pathological runs.
        sampler.run_nested(print_progress=False, dlogz=0.5, maxiter=20000)
        return float(sampler.results.logz[-1])

    nat = Mixture([_clone(m) for m in natural_components], name="natural")
    lnz_nat = _evidence(nat)
    per: list[TemplateBayes] = []
    for template in exotic_library:
        alt = Mixture([_clone(m) for m in natural_components] + [template.factory()],
                      name=f"alt.{template.name}")
        lnz_alt = _evidence(alt)
        per.append(TemplateBayes(template.name, float(lnz_alt - lnz_nat), float("nan"),
                                 len(template.factory().parameters.free)))
    best = max(per, key=lambda t: t.log_bayes_factor)
    return BayesResult(best.log_bayes_factor, best.template_name, per, {}, "dynesty",
                       notes=f"ln Z_nat = {lnz_nat:.3f}; nlive = {nlive}")
