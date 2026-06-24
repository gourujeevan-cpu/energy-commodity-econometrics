"""
End-to-end analysis runner.

Executes the full pipeline on a chosen commodity and writes every figure and
results table into ``outputs/``. Run from the repository root:

    python run_analysis.py                 # Henry Hub, bundled snapshot
    python run_analysis.py --commodity brent --live   # Brent, fresh from FRED

The script is intentionally linear and commented so a reader can follow the
analysis as a narrative; the reusable logic lives in ``src/``.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

from src import data, diagnostics, volatility, regime, risk, seasonality, plots

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, name: str):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> outputs/figures/{name}.png")


def _save_table(df: pd.DataFrame, name: str):
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  table  -> outputs/tables/{name}.csv")


def main(commodity: str, live: bool):
    print(f"\nLoading {commodity} ...")
    ps = data.load(commodity, prefer_live=live)
    print(ps.describe().to_string())
    _save_table(ps.describe().rename("value").reset_index().rename(columns={"index": "stat"}),
                "00_summary")

    # 1. Stationarity & ARCH effects -------------------------------------
    print("\n[1] Diagnostics")
    stn = diagnostics.stationarity_table(ps)
    arch_tbl = diagnostics.arch_effects(ps.ret)
    print(stn.to_string(index=False))
    print(arch_tbl.to_string(index=False))
    _save_table(stn, "01_stationarity")
    _save_table(arch_tbl, "02_arch_effects")

    # 2. Volatility models -----------------------------------------------
    print("\n[2] Volatility models")
    fits = volatility.fit_all(ps.ret)
    comp = volatility.comparison_table(fits)
    print(comp.to_string(index=False))
    _save_table(comp, "03_model_comparison")

    best_name = comp.loc[0, "model"]
    # Prefer GJR for the risk layer (has an interpretable closed-form residual);
    # report persistence for the GARCH-class models.
    gjr = fits["GJR-GARCH(1,1)"]
    print(f"\n  Selected by AIC: {best_name}")
    print(f"  GJR-GARCH persistence (a+b+g/2): {volatility.persistence(gjr):.4f}")
    print(f"  GJR-GARCH asymmetry gamma: {gjr.params.get('gamma[1]', float('nan')):.4f} "
          f"(>0 = equity-style leverage; ~0 or <0 = energy-style)")
    params = gjr.params.round(4).rename("estimate").reset_index().rename(columns={"index": "param"})
    _save_table(params, "04_gjr_params")

    # 3. Out-of-sample volatility forecast evaluation --------------------
    print("\n[3] Rolling out-of-sample volatility forecast (last 500 days)")
    eval_rows = []
    for spec in ["GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)"]:
        _, ev = volatility.rolling_forecast_eval(ps.ret, spec=spec, test_size=500, refit_every=25)
        eval_rows.append({"model": ev.spec, "QLIKE": ev.qlike, "MSE": ev.mse})
        print(f"  {ev.spec:15s} QLIKE={ev.qlike:.4f}  MSE={ev.mse:.2f}")
    _save_table(pd.DataFrame(eval_rows).sort_values("QLIKE"), "05_oos_forecast")

    # 4. Regime switching -------------------------------------------------
    print("\n[4] Markov regime-switching")
    rm = regime.fit_regimes(ps.ret, k_regimes=2)
    print(rm.summary_frame().to_string(index=False))
    episodes = regime.crisis_episodes(rm, threshold=0.5, min_days=10)
    print(f"  crisis episodes (>=10 trading days): {len(episodes)}")
    _save_table(rm.summary_frame(), "06_regime_summary")
    _save_table(episodes, "07_crisis_episodes")

    # 5. VaR / ES backtest ------------------------------------------------
    print("\n[5] VaR / ES backtest (GJR-t)")
    bt = risk.backtest_table(ps.ret, gjr, levels=(0.05, 0.01))
    print(bt.to_string(index=False))
    _save_table(bt, "08_var_backtest")

    # 6. Seasonality ------------------------------------------------------
    print("\n[6] Volatility seasonality")
    monthly = seasonality.monthly_volatility(ps.ret)
    seas_reg = seasonality.seasonal_regression(ps.ret)
    print(monthly.to_string(index=False))
    print(f"  seasonal dummy regression joint F p-value: {seas_reg.attrs['f_pvalue']:.3e}")
    _save_table(monthly, "09_seasonality")
    _save_table(seas_reg, "10_seasonal_regression")

    # 7. Figures ----------------------------------------------------------
    print("\n[7] Figures")
    _save_fig(plots.plot_price_history(ps), "fig01_price_history")
    _save_fig(plots.plot_returns(ps), "fig02_returns")
    _save_fig(plots.plot_acf_squared(ps), "fig03_acf_squared")
    _save_fig(plots.plot_conditional_volatility(ps, gjr, "GJR-GARCH"), "fig04_cond_vol")
    _save_fig(plots.plot_regimes(ps, rm), "fig05_regimes")
    _save_fig(plots.plot_seasonality(monthly), "fig06_seasonality")
    var_path = risk.value_at_risk(gjr, alpha=0.01)["VaR"]
    _save_fig(plots.plot_var_backtest(ps, gjr, var_path, 0.01), "fig07_var_backtest")
    _save_fig(plots.plot_return_distribution(ps, gjr), "fig08_residual_dist")

    print("\nDone. See outputs/ for figures and tables.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Energy commodity volatility econometrics.")
    ap.add_argument("--commodity", default="henry_hub",
                    choices=["henry_hub", "brent", "wti"],
                    help="which series to analyse (default: henry_hub)")
    ap.add_argument("--live", action="store_true",
                    help="fetch fresh data from FRED instead of the bundled snapshot")
    args = ap.parse_args()
    main(args.commodity, args.live)
