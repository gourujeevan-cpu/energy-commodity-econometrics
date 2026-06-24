"""
Figure generation for the analysis.

All figures use a single restrained house style (muted palette, light grid,
no chartjunk) so the repository's output looks deliberate rather than default.
Every function takes already-computed objects and returns a Matplotlib figure;
saving is handled by the caller, which keeps plotting and I/O separate.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

# --- house style -----------------------------------------------------------
INK = "#1f2933"
ACCENT = "#2b6cb0"
WARN = "#c05621"
CALM = "#718096"
GRID = "#e2e8f0"

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_price_history(ps) -> plt.Figure:
    """Price (top) and log price (bottom) over the full sample."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(ps.price.index, ps.price.values, color=ACCENT, lw=0.8)
    axes[0].set_title(f"{ps.name}: spot price")
    axes[0].set_ylabel("price")
    axes[1].plot(ps.log_price.index, ps.log_price.values, color=INK, lw=0.8)
    axes[1].set_title("log price")
    axes[1].set_ylabel("log price")
    fig.tight_layout()
    return fig


def plot_returns(ps) -> plt.Figure:
    """Return series -- the eye-test for volatility clustering."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(ps.ret.index, ps.ret.values, color=INK, lw=0.5)
    ax.set_title(f"{ps.name}: daily log returns (%)")
    ax.set_ylabel("return (%)")
    fig.tight_layout()
    return fig


def plot_acf_squared(ps, lags: int = 40) -> plt.Figure:
    """ACF of squared returns -- visual evidence of ARCH effects."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    plot_acf(ps.ret.dropna() ** 2, lags=lags, ax=ax, color=ACCENT, vlines_kwargs={"colors": ACCENT})
    ax.set_title(f"{ps.name}: ACF of squared returns")
    fig.tight_layout()
    return fig


def plot_conditional_volatility(ps, res, spec_name: str) -> plt.Figure:
    """Estimated conditional (annualised) volatility from the chosen model."""
    cv = res.conditional_volatility * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(cv.index, cv.values, color=WARN, lw=0.7)
    ax.set_title(f"{ps.name}: conditional volatility ({spec_name}, annualised %)")
    ax.set_ylabel("annualised vol (%)")
    fig.tight_layout()
    return fig


def plot_regimes(ps, regime_model) -> plt.Figure:
    """Price with the smoothed crisis-regime probability shaded beneath it."""
    p_crisis = regime_model.crisis_probability
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(ps.price.index, ps.price.values, color=INK, lw=0.7)
    axes[0].set_title(f"{ps.name}: price and identified crisis regime")
    axes[0].set_ylabel("price")

    axes[1].fill_between(p_crisis.index, 0, p_crisis.values, color=WARN, alpha=0.6, lw=0)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("P(crisis)")
    axes[1].axhline(0.5, color=CALM, ls="--", lw=0.8)

    # light shading on the price panel where crisis prob is high
    high = p_crisis > 0.5
    axes[0].fill_between(p_crisis.index, *axes[0].get_ylim(),
                         where=high.values, color=WARN, alpha=0.10, lw=0)
    fig.tight_layout()
    return fig


def plot_seasonality(monthly: pd.DataFrame) -> plt.Figure:
    """Monthly annualised volatility profile."""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [WARN if v >= monthly["annual_vol_%"].median() else ACCENT
              for v in monthly["annual_vol_%"]]
    ax.bar(monthly["month"], monthly["annual_vol_%"], color=colors)
    ax.set_title("Volatility seasonality (annualised %, by calendar month)")
    ax.set_ylabel("annualised vol (%)")
    fig.tight_layout()
    return fig


def plot_var_backtest(ps, res, var_path: pd.Series, alpha: float) -> plt.Figure:
    """Returns with the VaR line and breaches highlighted."""
    r = ps.ret.reindex(var_path.index)
    breach = r < -var_path
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(r.index, r.values, color=CALM, lw=0.4, label="return")
    ax.plot(var_path.index, -var_path.values, color=ACCENT, lw=0.8,
            label=f"{alpha:.0%} VaR")
    ax.scatter(r.index[breach], r.values[breach], color=WARN, s=10, zorder=3,
               label="breach")
    ax.set_title(f"{ps.name}: {alpha:.0%} one-day VaR and exceedances")
    ax.set_ylabel("return (%)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_return_distribution(ps, res) -> plt.Figure:
    """Standardised-residual histogram against the fitted Student-t density."""
    from scipy import stats
    std_resid = (res.resid / res.conditional_volatility).dropna()
    nu = res.params["nu"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(std_resid, bins=120, density=True, color=GRID, edgecolor="none")
    x = np.linspace(std_resid.min(), std_resid.max(), 400)
    scale = np.sqrt((nu - 2) / nu)
    ax.plot(x, stats.t.pdf(x / scale, nu) / scale, color=WARN, lw=1.5,
            label=f"Student-t (nu={nu:.1f})")
    ax.plot(x, stats.norm.pdf(x), color=ACCENT, lw=1.2, ls="--", label="Normal")
    ax.set_title(f"{ps.name}: standardised residuals vs fitted density")
    ax.set_xlim(-6, 6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
