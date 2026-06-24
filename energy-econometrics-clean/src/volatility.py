"""
Conditional volatility models for energy commodity returns.

Three workhorse specifications are estimated, all with Student-t innovations to
absorb the heavy tails that ARCH-LM flags in the diagnostics step:

*  **GARCH(1,1)**   -- symmetric benchmark.
*  **GJR-GARCH(1,1)** -- adds a sign-dependent term (gamma) to test for an
   asymmetric ("leverage") response of volatility to the sign of shocks.
*  **EGARCH(1,1)** -- models log-variance, so positivity is automatic and the
   asymmetry enters multiplicatively.

A note worth making in interviews: in *equity* markets gamma > 0 (bad news lifts
volatility more than good news). In energy markets the asymmetry is frequently
weak or even reversed, because a positive price shock often signals a supply
scare that is itself destabilising. The estimation here lets the data speak.

Beyond fitting, the module runs a genuine out-of-sample exercise: a rolling
one-step-ahead variance forecast evaluated against realised squared returns with
the QLIKE and MSE loss functions, which is the honest way to compare volatility
models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

# Specification registry. ``o=1`` switches on the asymmetric term.
SPECS: dict[str, dict] = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
}


def fit_one(returns: pd.Series, spec: str = "GJR-GARCH(1,1)", dist: str = "t") -> ARCHModelResult:
    """Fit a single specification from :data:`SPECS`."""
    if spec not in SPECS:
        raise KeyError(f"Unknown spec '{spec}'. Choose from {list(SPECS)}.")
    model = arch_model(returns.dropna(), mean="Constant", dist=dist, **SPECS[spec])
    return model.fit(disp="off")


def fit_all(returns: pd.Series, dist: str = "t") -> dict[str, ARCHModelResult]:
    """Fit every specification and return them keyed by name."""
    return {name: fit_one(returns, name, dist) for name in SPECS}


def comparison_table(fits: dict[str, ARCHModelResult]) -> pd.DataFrame:
    """Rank fitted models by information criteria."""
    rows = [
        {
            "model": name,
            "log_lik": round(res.loglikelihood, 1),
            "n_params": int(res.num_params),
            "AIC": round(res.aic, 1),
            "BIC": round(res.bic, 1),
        }
        for name, res in fits.items()
    ]
    tbl = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    tbl["selected"] = ""
    tbl.loc[0, "selected"] = "<= best AIC"
    return tbl


def persistence(res: ARCHModelResult) -> float:
    """Volatility persistence (alpha + beta, plus half the asymmetry term).

    Values close to 1 indicate long memory in volatility -- shocks decay slowly.
    Only defined for the GARCH-class (not EGARCH) parameterisation.
    """
    p = res.params
    a = p.get("alpha[1]", 0.0)
    b = p.get("beta[1]", 0.0)
    g = p.get("gamma[1]", 0.0)
    return float(a + b + 0.5 * g)


@dataclass
class ForecastEvaluation:
    spec: str
    n_test: int
    qlike: float
    mse: float


def rolling_forecast_eval(
    returns: pd.Series,
    spec: str = "GJR-GARCH(1,1)",
    test_size: int = 500,
    refit_every: int = 25,
) -> tuple[pd.Series, ForecastEvaluation]:
    """Expanding-window one-step-ahead variance forecast, then score it.

    The model is re-estimated every ``refit_every`` days (a pragmatic compromise
    between a daily refit and a single static fit) and produces a genuine
    out-of-sample variance forecast for each day in the final ``test_size``
    observations. Forecasts are scored against realised squared returns with:

    *  **QLIKE** = mean( log(h) + r^2 / h )  -- robust to the noise in r^2 as a
       volatility proxy and the standard loss for this task.
    *  **MSE**   = mean( (r^2 - h)^2 ).
    """
    r = returns.dropna()
    n = len(r)
    split = n - test_size
    forecasts = np.full(test_size, np.nan)

    res = None
    for i in range(test_size):
        if i % refit_every == 0:
            window = r.iloc[: split + i]
            res = arch_model(window, mean="Constant", dist="t", **SPECS[spec]).fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        forecasts[i] = fc.variance.values[-1, 0]

    realised = (r.iloc[split:].values) ** 2
    h = forecasts
    qlike = float(np.mean(np.log(h) + realised / h))
    mse = float(np.mean((realised - h) ** 2))

    fc_series = pd.Series(np.sqrt(forecasts), index=r.index[split:], name="forecast_vol")
    return fc_series, ForecastEvaluation(spec, test_size, round(qlike, 4), round(mse, 2))
