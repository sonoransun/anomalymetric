"""Profile-likelihood-ratio scorer over the exotic-template library.

For each template T:

    M_nat   = natural mixture (frozen-shape + free amplitudes, indices, ...)
    M_alt   = M_nat + amplitude(T) * T(E)
    logL_nat = max over nuisance params of log L(data | M_nat)
    logL_alt = max over nuisance params of log L(data | M_alt)
    TS_T    = -2 (logL_nat - logL_alt)     # always >= 0 by construction

`score` is `max_T TS_T` with the trials correction applied (see trials.py).
The breakdown returned includes the per-template TS and the post-fit amplitude
of each natural component, so callers can present *why* a source ranks high.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from anomalymetric.forward.base import ForwardModel
from anomalymetric.models.base import Parameter
from anomalymetric.models.exotic import ExoticTemplate, default_library
from anomalymetric.models.inference import Fit
from anomalymetric.models.mixture import Mixture
from anomalymetric.score.trials import gross_vitells_global_p
from anomalymetric.spectrum import Spectrum


@dataclass
class TemplateScore:
    template_name: str
    ts: float
    delta_log_likelihood: float
    amplitude: float
    log_likelihood_natural: float
    log_likelihood_alternative: float


@dataclass
class ScoreResult:
    delta_log_likelihood: float  # raw max(logL_alt) - logL_nat
    test_statistic: float  # max TS over templates
    anomaly_score: float  # trials-corrected (-log10 p_global)
    best_template: str
    per_template: list[TemplateScore] = field(default_factory=list)
    natural_parameters: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def loeb_turner_score(
    spectrum: Spectrum,
    natural_components: list,
    *,
    forward: Optional[ForwardModel] = None,
    exotic_library: Optional[list[ExoticTemplate]] = None,
    n_trials: Optional[int] = None,
) -> ScoreResult:
    """Compute the Loeb–Turner-style score for `spectrum`.

    Parameters
    ----------
    spectrum : Spectrum
        Observed spectrum (any `value_kind`; counts dispatch happens inside `Fit`).
    natural_components : list of Model
        Components making up the natural mixture (e.g., BlackBody, PowerLaw,
        SolarReflection). Their nuisance parameters are profiled out.
    forward : ForwardModel, optional
        Detector forward model. Defaults to identity for photons, flat exposure
        for cosmic rays/neutrinos.
    exotic_library : list of ExoticTemplate, optional
        Defaults to `models.exotic.default_library()`.
    n_trials : int, optional
        Effective number of independent trials for the look-elsewhere correction.
        Defaults to `len(exotic_library)` (the discrete-template case). Pass an
        explicit value for a continuous line-energy scan (~ scan_range / line_width);
        the explicit value is honored even if it is smaller than the library size.
    """
    if exotic_library is None:
        exotic_library = default_library()

    # Fit the natural mixture once.
    natural = Mixture([_clone(m) for m in natural_components], name="natural")
    nat_fit = Fit(natural, spectrum, forward=forward)
    nat_result = nat_fit.run()
    nat_pred = nat_fit.predicted_value()

    per_template: list[TemplateScore] = []
    for template in exotic_library:
        alt_components = [_clone(m) for m in natural_components] + [template.factory()]
        alt = Mixture(alt_components, name=f"alt.{template.name}")
        # Warm-start the alt mixture from the natural fit values.
        for name, value in nat_result.parameter_values.items():
            if name in [p.name for p in alt.parameters.params]:
                alt.parameters[name].value = value
        alt_fit = Fit(alt, spectrum, forward=forward)
        # Seed the template amplitude with its closed-form matched-filter estimate
        # so the optimizer starts at the right order of magnitude. Without this,
        # channels whose amplitudes span many decades (e.g. PSD lines far below
        # the noise floor) cannot be reached from a fixed default start.
        _warm_start_template_amplitude(alt, alt_fit, nat_pred, template.name)
        alt_result = alt_fit.run()
        d_ll = alt_result.log_likelihood - nat_result.log_likelihood
        ts = max(0.0, 2.0 * d_ll)
        # Exact key: the template is the last component, so its amplitude parameter
        # is `"{template.name}.amplitude"`. A substring match would alias templates
        # whose names are prefixes of one another.
        amp_key = f"{template.name}.amplitude"
        amp = float(alt_result.parameter_values.get(amp_key, np.nan))
        per_template.append(
            TemplateScore(
                template_name=template.name,
                ts=ts,
                delta_log_likelihood=d_ll,
                amplitude=amp,
                log_likelihood_natural=nat_result.log_likelihood,
                log_likelihood_alternative=alt_result.log_likelihood,
            )
        )

    best = max(per_template, key=lambda t: t.ts)
    trials = n_trials if n_trials is not None else len(exotic_library)
    p_global = gross_vitells_global_p(best.ts, n_trials=trials)
    anomaly_score = -np.log10(max(p_global, 1e-300))
    return ScoreResult(
        delta_log_likelihood=best.delta_log_likelihood,
        test_statistic=best.ts,
        anomaly_score=float(anomaly_score),
        best_template=best.template_name,
        per_template=per_template,
        natural_parameters=nat_result.parameter_values,
        notes=f"natural log L = {nat_result.log_likelihood:.3f}; n_trials = {trials}",
    )


def _clone(model):
    """Deep-copy a Model so multiple fits don't share Parameter state."""
    return copy.deepcopy(model)


def _warm_start_template_amplitude(alt, alt_fit, nat_pred, template_name) -> None:
    """Set the template's amplitude to its weighted-least-squares matched-filter estimate.

    For a fixed template shape `t` (unit amplitude) added on top of the natural
    prediction, the amplitude that best explains the residual `r = data - nat`
    under inverse-variance weights `w` is `sum(w r t) / sum(w t^2)`. This is the
    exact MLE for the linear amplitude and gives the optimizer a starting point
    of the correct magnitude regardless of the channel's absolute scale.
    """
    amp_key = f"{template_name}.amplitude"
    try:
        amp_param = alt.parameters[amp_key]
    except KeyError:
        return
    if amp_param.frozen:
        return
    amp_param.value = 1.0  # evaluate the unit-amplitude template contribution
    unit_dnde = alt.component_dnde(alt_fit._log_centers)[template_name]
    t_pred = alt_fit.forward.forward(alt_fit._edges, unit_dnde)
    if alt_fit._is_gaussian:
        var = np.clip(alt_fit._sigma, 1e-300, np.inf) ** 2
    else:
        var = np.clip(nat_pred, 1.0, np.inf)  # Poisson variance ~ mean
    resid = alt_fit._observed - nat_pred
    denom = float(np.sum(t_pred * t_pred / var))
    if denom <= 0 or not np.isfinite(denom):
        amp_param.value = 0.0
        return
    amp0 = float(np.sum(resid * t_pred / var) / denom)
    amp_param.value = amp_param.clamp(max(amp0, 0.0))
