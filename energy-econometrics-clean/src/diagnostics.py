"""
Pre-modelling diagnostics: order of integration and conditional heteroskedasticity.

Two questions are settled here before any volatility model is fitted:

1.  *What do we model -- the price or the return?*  Energy spot prices typically
    contain a unit root (they are I(1)), while returns are stationary (I(0)). We
    confirm this with two complementary tests that flip the null hypothesis:
    the Augmented Dickey-Fuller test (H0: unit root) and the KPSS test
    (H0: stationarity). Agreement across the pair is far more persuasive than
    either alone.

2.  *Is there volatility clustering worth modelling?*  We test the squared
    returns for autocorrelation (Ljung-Box) and run Engle's ARCH-LM test. Strong
    rejections are the empirical licence for a GARCH-type specification.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    name: str
    adf_stat: float
    adf_p: float
    kpss_stat: float
    kpss_p: float

    @property
    def verdict(self) -> str:
        """Combine the two tests into a plain-language order-of-integration call."""
        adf_stationary = self.adf_p < 0.05          # ADF rejects unit root
        kpss_stationary = self.kpss_p > 0.05         # KPSS fails to reject stationarity
        if adf_stationary and kpss_stationary:
            return "stationary  I(0)"
        if not adf_stationary and not kpss_stationary:
            return "non-stationary  I(1)"
        return "borderline / inconclusive"


def test_stationarity(series: pd.Series, name: str) -> StationarityResult:
    """Run ADF and KPSS on a single series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS p-value look-up table clipping
        adf = adfuller(series.dropna(), autolag="AIC")
        kp = kpss(series.dropna(), regression="c", nlags="auto")
    return StationarityResult(name, adf[0], adf[1], kp[0], kp[1])


def stationarity_table(ps) -> pd.DataFrame:
    """Tabulate stationarity tests for both the log price and the log return."""
    rows = [
        test_stationarity(ps.log_price, "log price"),
        test_stationarity(ps.ret, "log return"),
    ]
    return pd.DataFrame(
        [
            {
                "series": r.name,
                "ADF stat": round(r.adf_stat, 3),
                "ADF p": round(r.adf_p, 4),
                "KPSS stat": round(r.kpss_stat, 3),
                "KPSS p": round(r.kpss_p, 4),
                "verdict": r.verdict,
            }
            for r in rows
        ]
    )


def arch_effects(returns: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Test returns for ARCH effects (volatility clustering).

    Reports the Ljung-Box statistic on squared returns and Engle's ARCH-LM test,
    alongside the unconditional skew and excess kurtosis that motivate a
    fat-tailed (Student-t) error distribution.
    """
    r = returns.dropna()
    lb = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
    lm_stat, lm_p, _, _ = het_arch(r, nlags=lags)
    return pd.DataFrame(
        [
            {"test": f"Ljung-Box Q({lags}) on squared returns",
             "statistic": round(float(lb["lb_stat"].iloc[0]), 1),
             "p_value": float(lb["lb_pvalue"].iloc[0])},
            {"test": f"Engle ARCH-LM({lags})",
             "statistic": round(float(lm_stat), 1),
             "p_value": float(lm_p)},
            {"test": "excess kurtosis (unconditional)",
             "statistic": round(float(r.kurtosis()), 2),
             "p_value": np.nan},
            {"test": "skewness (unconditional)",
             "statistic": round(float(r.skew()), 3),
             "p_value": np.nan},
        ]
    )
