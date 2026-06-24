"""
Seasonality in energy commodity volatility.

Energy commodities -- natural gas especially -- carry calendar structure that
financial assets do not. Heating demand concentrates consumption and storage
stress in winter, so both the level and the *volatility* of gas prices show a
pronounced seasonal pattern. This module quantifies the volatility seasonality
two ways:

*  a non-parametric monthly volatility profile (annualised standard deviation of
   returns by calendar month), and
*  a regression of squared returns on month dummies, which gives a formal test
   of whether the seasonal differences are statistically distinguishable from
   noise.

For crude (Brent/WTI) the seasonal signal is weaker; running the same code on
those series and finding a flat profile is itself an informative result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_volatility(returns: pd.Series) -> pd.DataFrame:
    """Annualised return volatility by calendar month."""
    r = returns.dropna()
    by_month = r.groupby(r.index.month)
    vol = by_month.std() * np.sqrt(252)
    mean = by_month.mean() * 252
    out = pd.DataFrame(
        {
            "month": [_MONTHS[m - 1] for m in vol.index],
            "annual_vol_%": vol.round(1).values,
            "annual_mean_%": mean.round(1).values,
            "n_obs": by_month.size().values,
        }
    )
    return out


def seasonal_regression(returns: pd.Series) -> pd.DataFrame:
    """Regress squared returns on month dummies (December as the base month).

    A jointly significant set of dummies is formal evidence of volatility
    seasonality. Coefficients are differences in mean squared return relative to
    December.
    """
    r = returns.dropna()
    r2 = (r ** 2).rename("r2")
    month = pd.Categorical(r.index.month, categories=range(1, 13))
    dummies = pd.get_dummies(month, prefix="m", drop_first=True).astype(float)
    dummies.index = r.index
    X = sm.add_constant(dummies)
    model = sm.OLS(r2.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    names = ["const"] + [f"month_{_MONTHS[m - 1]}" for m in range(2, 13)]
    out = pd.DataFrame(
        {
            "term": names,
            "coef": np.round(model.params, 3),
            "t_HAC": np.round(model.tvalues, 2),
            "p_value": np.round(model.pvalues, 4),
        }
    )
    out.attrs["f_pvalue"] = float(model.f_pvalue)
    out.attrs["r_squared"] = float(model.rsquared)
    return out
