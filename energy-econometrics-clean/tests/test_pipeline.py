"""
Smoke tests for the core pipeline.

These are not a full statistical test suite; they verify that each module runs
end to end on the bundled data and returns objects with the expected shape and
sane values. Run with: ``pytest tests/`` from the repository root.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, diagnostics, volatility, regime, risk, seasonality


@pytest.fixture(scope="module")
def ps():
    return data.load_local()


def test_data_loads_clean(ps):
    assert len(ps.frame) > 5000
    assert ps.ret.isna().sum() == 0
    assert (ps.price > 0).all()


def test_returns_are_stationary(ps):
    res = diagnostics.test_stationarity(ps.ret, "ret")
    assert res.adf_p < 0.05          # ADF rejects unit root
    assert "I(0)" in res.verdict


def test_arch_effects_present(ps):
    tbl = diagnostics.arch_effects(ps.ret)
    lb_p = tbl.loc[tbl["test"].str.contains("Ljung"), "p_value"].iloc[0]
    assert lb_p < 0.01               # strong volatility clustering


def test_garch_family_fits(ps):
    fits = volatility.fit_all(ps.ret)
    assert set(fits) == set(volatility.SPECS)
    for res in fits.values():
        assert np.isfinite(res.loglikelihood)
    gjr = fits["GJR-GARCH(1,1)"]
    assert 0.8 < volatility.persistence(gjr) < 1.0   # high but stationary


def test_regimes_separate(ps):
    rm = regime.fit_regimes(ps.ret)
    # crisis regime must have higher volatility than calm
    assert rm.sigma[rm.high_vol_state] == rm.sigma.max()
    assert rm.sigma.max() > 2 * rm.sigma.min()


def test_var_backtest_runs(ps):
    fit = volatility.fit_one(ps.ret, "GJR-GARCH(1,1)")
    bt = risk.backtest_var(ps.ret, fit, alpha=0.01)
    assert 0 < bt.breach_rate < 0.05
    assert 0.0 <= bt.kupiec_p <= 1.0


def test_kupiec_numerically_stable():
    # the bug this guards against: long samples underflowing to NaN
    lr, p = risk.kupiec_pof(breaches=358, n=6860, alpha=0.05)
    assert np.isfinite(lr) and np.isfinite(p)


def test_seasonality_significant(ps):
    reg = seasonality.seasonal_regression(ps.ret)
    assert reg.attrs["f_pvalue"] < 0.01
