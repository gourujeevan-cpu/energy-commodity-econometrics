"""
Tail-risk estimation and backtesting from a fitted GARCH-t model.

Conditional volatility is only useful to a desk if it translates into a risk
number that survives validation. This module turns the fitted model into
one-day Value-at-Risk (VaR) and Expected Shortfall (ES), then subjects the VaR
to the two standard regulatory-style coverage tests:

*  **Kupiec POF** (unconditional coverage): are there the right *number* of
   breaches?  Computed in log-space so it is numerically stable for long
   samples (the naive product form underflows and returns NaN).
*  **Christoffersen** (independence + conditional coverage): are the breaches
   *spread out*, or do they cluster -- the dangerous failure mode where a model
   is fine in calm periods but blind during a crisis?

VaR/ES use the standardised Student-t with the degrees of freedom estimated by
the volatility model, rescaled to unit variance so the conditional volatility
carries the scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _t_var_es(alpha: float, nu: float) -> tuple[float, float]:
    """Standardised (unit-variance) Student-t VaR and ES multipliers at level alpha.

    Returns the lower-tail quantile and the expected shortfall multiplier, each
    scaled so the innovation has variance one.
    """
    scale = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)                       # raw t quantile (negative)
    var_mult = q * scale
    pdf_q = stats.t.pdf(q, nu)
    es_raw = -(nu + q ** 2) / (nu - 1) * pdf_q / alpha
    es_mult = es_raw * scale
    return var_mult, es_mult


def value_at_risk(res, alpha: float = 0.01) -> pd.DataFrame:
    """One-day VaR and ES paths (as positive loss numbers) from a fitted model."""
    mu = res.params["mu"]
    nu = res.params["nu"]
    sigma = res.conditional_volatility
    var_mult, es_mult = _t_var_es(alpha, nu)
    var = -(mu + sigma * var_mult)
    es = -(mu + sigma * es_mult)
    return pd.DataFrame({"VaR": var, "ES": es}, index=sigma.index)


def kupiec_pof(breaches: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    x = breaches
    pi_hat = x / n
    # log-likelihoods under H0 (coverage = alpha) and H1 (coverage = pi_hat)
    ll_h0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    if x == 0 or x == n:
        ll_h1 = 0.0
    else:
        ll_h1 = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    lr = -2 * (ll_h0 - ll_h1)
    p = 1 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p)


def christoffersen(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Christoffersen independence and conditional-coverage tests.

    ``hits`` is a 0/1 array of VaR exceedances in time order.
    """
    hits = np.asarray(hits).astype(int)
    # transition counts n_ij = from state i to state j
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits[:-1], hits[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, a, b):
        if p in (0.0, 1.0):
            return 0.0
        return a * np.log(1 - p) + b * np.log(p)

    ll_ind_h0 = _ll(pi, n00 + n10, n01 + n11)
    ll_ind_h1 = _ll(pi01, n00, n01) + _ll(pi11, n10, n11)
    lr_ind = -2 * (ll_ind_h0 - ll_ind_h1)

    x = int(hits.sum())
    n = len(hits)
    lr_uc, _ = kupiec_pof(x, n, alpha)
    lr_cc = lr_uc + lr_ind  # conditional coverage = unconditional + independence

    return {
        "LR_independence": round(float(lr_ind), 3),
        "p_independence": round(float(1 - stats.chi2.cdf(lr_ind, df=1)), 4),
        "LR_cond_coverage": round(float(lr_cc), 3),
        "p_cond_coverage": round(float(1 - stats.chi2.cdf(lr_cc, df=2)), 4),
    }


@dataclass
class VaRBacktest:
    alpha: float
    n: int
    expected_breaches: float
    observed_breaches: int
    breach_rate: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen: dict


def backtest_var(returns: pd.Series, res, alpha: float = 0.01) -> VaRBacktest:
    """Full VaR backtest: coverage rate, Kupiec and Christoffersen."""
    r = returns.dropna()
    var = value_at_risk(res, alpha)["VaR"].reindex(r.index)
    hits = (r < -var).astype(int).values
    x = int(hits.sum())
    n = len(r)
    lr, p = kupiec_pof(x, n, alpha)
    chris = christoffersen(hits, alpha)
    return VaRBacktest(
        alpha=alpha,
        n=n,
        expected_breaches=round(alpha * n, 1),
        observed_breaches=x,
        breach_rate=round(x / n, 4),
        kupiec_lr=round(lr, 3),
        kupiec_p=round(p, 4),
        christoffersen=chris,
    )


def backtest_table(returns: pd.Series, res, levels=(0.05, 0.01)) -> pd.DataFrame:
    """Backtest at several confidence levels and tabulate."""
    rows = []
    for a in levels:
        bt = backtest_var(returns, res, a)
        rows.append(
            {
                "level": f"{a:.0%}",
                "expected": bt.expected_breaches,
                "observed": bt.observed_breaches,
                "breach_rate": f"{bt.breach_rate:.2%}",
                "kupiec_p": bt.kupiec_p,
                "christoffersen_cc_p": bt.christoffersen["p_cond_coverage"],
                "pass_5pct": "yes" if min(bt.kupiec_p, bt.christoffersen["p_cond_coverage"]) > 0.05 else "no",
            }
        )
    return pd.DataFrame(rows)
