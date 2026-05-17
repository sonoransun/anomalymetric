"""Look-elsewhere / trials-factor correction (Gross & Vitells 2010).

For a single test the null distribution of `TS = -2 ln(L_null/L_alt)` is
chi^2 with one degree of freedom (Wilks). When the alternative scans over an
unconstrained parameter (e.g., line energy), the global p-value is

    p_global ≈ p_local + <N(TS_ref)> * exp(-(TS - TS_ref) / 2)

where `<N>` is the expected number of upcrossings at a reference threshold.
For a discrete template library the trials factor reduces to a Bonferroni
correction `p_global = 1 - (1 - p_local)^N ≈ N * p_local` for small p_local.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def local_p(ts: float) -> float:
    """Chi^2_1 survival function (one-sided test)."""
    if ts <= 0:
        return 1.0
    # One-sided: TS distributed as 0.5 delta(0) + 0.5 chi2_1 under the boundary.
    return 0.5 * float(chi2.sf(ts, df=1))


def gross_vitells_global_p(ts: float, n_trials: int = 1) -> float:
    """Bonferroni-style global p for a finite trials factor.

    For continuous scans (line-energy sweeps) the proper Gross-Vitells formula
    uses an empirical `<N(TS_ref)>`; v1 supplies the Bonferroni approximation,
    which is conservative and matches the discrete-template-library case
    exactly. Replace with a calibrated <N> when scanning a continuous parameter.
    """
    p_loc = local_p(ts)
    if n_trials <= 1:
        return p_loc
    return float(1.0 - (1.0 - p_loc) ** n_trials)
