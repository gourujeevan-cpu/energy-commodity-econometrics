"""
Markov regime-switching model for energy commodity returns.

A single-regime GARCH model assumes one set of dynamics for all time. Energy
markets plainly violate that: long calm stretches are punctuated by sharp,
short-lived crises (the 2000-01 California crisis, 2008, Winter Storm Uri in
Feb 2021, the 2021-22 European gas shock). A two-state Markov-switching model
captures this directly by letting the mean and variance jump between a *calm*
and a *crisis* regime, with the transitions governed by a hidden Markov chain.

The estimation yields three things an analyst actually wants:

*  the volatility of each regime,
*  the transition probability matrix and the implied **expected duration** of
   each regime (1 / (1 - p_stay)), and
*  the **smoothed probability** of being in the crisis regime on any given day,
   which can be overlaid on the price path to date the historical episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


@dataclass
class RegimeModel:
    result: object
    high_vol_state: int
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    transition: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_duration: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def crisis_probability(self) -> pd.Series:
        """Smoothed P(crisis regime) for every date in the sample."""
        p = self.result.smoothed_marginal_probabilities[self.high_vol_state]
        return pd.Series(np.asarray(p), index=p.index, name="p_crisis")

    def summary_frame(self) -> pd.DataFrame:
        """One row per regime: daily vol, annualised vol, persistence, duration."""
        rows = []
        for s in range(len(self.sigma)):
            rows.append(
                {
                    "regime": "crisis" if s == self.high_vol_state else "calm",
                    "daily_vol_%": round(float(self.sigma[s]), 2),
                    "annual_vol_%": round(float(self.sigma[s]) * np.sqrt(252), 1),
                    "stay_prob": round(float(self.transition[s, s]), 3),
                    "expected_duration_days": round(float(self.expected_duration[s]), 1),
                }
            )
        return pd.DataFrame(rows)


def fit_regimes(returns: pd.Series, k_regimes: int = 2) -> RegimeModel:
    """Estimate a Markov-switching model with switching mean and variance."""
    r = returns.dropna()
    res = MarkovRegression(
        r, k_regimes=k_regimes, trend="c", switching_variance=True
    ).fit()

    sigma2 = np.array([res.params[f"sigma2[{i}]"] for i in range(k_regimes)])
    sigma = np.sqrt(sigma2)
    high = int(np.argmax(sigma2))

    # statsmodels stores the (time-invariant) transition matrix as the first
    # slice of regime_transition; columns are "from", rows are "to".
    P = np.asarray(res.regime_transition[:, :, 0])
    stay = np.diag(P)
    duration = 1.0 / (1.0 - stay)

    return RegimeModel(
        result=res,
        high_vol_state=high,
        sigma=sigma,
        transition=P,
        expected_duration=duration,
    )


def crisis_episodes(model: RegimeModel, threshold: float = 0.5, min_days: int = 5) -> pd.DataFrame:
    """Extract contiguous high-probability crisis episodes from the smoothed path.

    Useful for sanity-checking that the statistically identified regime lines up
    with known market events, and for a clean table in the README.
    """
    p = model.crisis_probability
    in_crisis = (p > threshold).astype(int)
    change = in_crisis.diff().fillna(in_crisis.iloc[0])

    episodes = []
    start = None
    for date, flag in in_crisis.items():
        if flag == 1 and start is None:
            start = date
        elif flag == 0 and start is not None:
            length = (p.index.get_loc(date) - p.index.get_loc(start))
            if length >= min_days:
                episodes.append((start, prev, length))
            start = None
        prev = date
    if start is not None:
        length = len(p) - p.index.get_loc(start)
        if length >= min_days:
            episodes.append((start, p.index[-1], length))

    return pd.DataFrame(episodes, columns=["start", "end", "trading_days"])
