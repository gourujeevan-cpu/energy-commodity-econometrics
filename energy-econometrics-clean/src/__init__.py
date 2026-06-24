"""Energy commodity econometrics toolkit.

A small, dependency-light library for the volatility econometrics of energy
commodity prices: stationarity diagnostics, the GARCH family with Student-t
innovations, Markov regime-switching, VaR/ES with coverage backtests, and
volatility seasonality. Commodity-agnostic by construction.
"""
from . import data, diagnostics, volatility, regime, risk, seasonality, plots

__all__ = ["data", "diagnostics", "volatility", "regime", "risk", "seasonality", "plots"]
__version__ = "0.1.0"
