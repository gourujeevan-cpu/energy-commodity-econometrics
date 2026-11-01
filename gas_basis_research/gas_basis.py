"""Monthly EU-US natural gas price-basis research (spot proxy; NOT a trading backtest)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EU_ALIASES = {"ngas_eur", "natural gas europe", "natural gas, europe", "natural gas (europe)", "gas_europe"}
US_ALIASES = {"ngas_us", "natural gas us", "natural gas, us", "natural gas (u.s.)", "gas_us"}


def _monthly_dates(s):
    raw = s.astype(str).str.strip().str.replace(r"^(\d{4})M(\d{2})$", r"\1-\2-01", regex=True)
    dt = pd.to_datetime(raw, errors="coerce")
    if dt.isna().any():
        raise ValueError("Unparseable monthly dates; use YYYY-MM-DD or YYYYMMM format")
    return dt.dt.to_period("M").dt.to_timestamp()


def _clean_prices(frame):
    df = frame[["date", "europe_usd_mmbtu", "us_usd_mmbtu"]].copy()
    df["date"] = _monthly_dates(df["date"])
    for key in ["europe_usd_mmbtu", "us_usd_mmbtu"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate price months are not allowed")
    df = df.sort_values("date").set_index("date")
    df = df.dropna()
    if (df <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if len(df) < 50:
        raise ValueError("Need at least 50 valid monthly observations")
    return df


def load_worldbank(path):
    """Recognize header placement and World Bank NGAS_EUR / NGAS_US names."""
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    for i in range(min(15, len(raw))):
        vals = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        eu = next((j for j, name in enumerate(vals) if name in EU_ALIASES), None)
        us = next((j for j, name in enumerate(vals) if name in US_ALIASES), None)
        date = next((j for j, name in enumerate(vals) if name in {"date", "month"}), None)
        if None not in (eu, us, date):
            frame = raw.iloc[i + 1:, [date, eu, us]].copy()
            frame.columns = ["date", "europe_usd_mmbtu", "us_usd_mmbtu"]
            return _clean_prices(frame)
    raise ValueError("Cannot locate World Bank gas columns. Expected Date, NGAS_EUR, NGAS_US")


def load_prices_csv(path):
    return _clean_prices(pd.read_csv(path))


def load_storage(path):
    """Optional GIE AGSI export; date,fill_pct (0-100), last known observation each month."""
    s = pd.read_csv(path)
    if not {"date", "fill_pct"}.issubset(s):
        raise ValueError("Storage CSV must contain date,fill_pct")
    s["date"] = pd.to_datetime(s["date"], errors="raise")
    s["fill_pct"] = pd.to_numeric(s["fill_pct"], errors="raise")
    if not s["fill_pct"].between(0, 100).all():
        raise ValueError("GIE AGSI storage fullness must lie between 0 and 100 percent")
    s = s.sort_values("date").drop_duplicates("date", keep="last")
    s["date"] = s["date"].dt.to_period("M").dt.to_timestamp()
    return s.groupby("date")["fill_pct"].last()


def make_panel(prices, storage=None, start="2015-04"):
    """Predict next-month log-basis change with only end-of-month observed inputs."""
    p = prices.loc[pd.Timestamp(start):].copy()
    p["log_basis"] = np.log(p["europe_usd_mmbtu"] / p["us_usd_mmbtu"])
    p["basis_z"] = (p["log_basis"] - p["log_basis"].rolling(12, min_periods=12).mean()) / p["log_basis"].rolling(12, min_periods=12).std().replace(0, np.nan)
    p["eu_return"] = np.log(p["europe_usd_mmbtu"]).diff()
    p["us_return"] = np.log(p["us_usd_mmbtu"]).diff()
    p["winter"] = p.index.month.isin([11, 12, 1, 2, 3]).astype(float)
    p["month_sin"] = np.sin(2 * np.pi * p.index.month / 12)
    p["month_cos"] = np.cos(2 * np.pi * p.index.month / 12)
    features = ["basis_z", "eu_return", "us_return", "winter", "month_sin", "month_cos"]
    if storage is not None:
        p = p.join(storage.rename("storage_fill_pct"))
        p["storage_deficit"] = 100 - p["storage_fill_pct"]
        features.append("storage_deficit")
    p["y_next_basis_change"] = p["log_basis"].shift(-1) - p["log_basis"]
    return p.dropna(subset=features + ["y_next_basis_change"]), features


def walk_forward(panel, features, min_train=48, alpha=10.0):
    """Expanding-window training; no shuffling or future features."""
    if len(panel) <= min_train + 6:
        raise ValueError("Not enough aligned monthly rows for meaningful out-of-sample test")
    rows = []
    for i in range(min_train, len(panel)):
        tr, test = panel.iloc[:i], panel.iloc[[i]]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(tr[features], tr["y_next_basis_change"])
        pred = float(model.predict(test[features])[0])
        seasonal = tr.loc[tr.index.month == test.index[0].month, "y_next_basis_change"]
        rows.append({"date": str(test.index[0].date()), "actual": float(test["y_next_basis_change"].iloc[0]),
                     "ridge": pred, "zero": 0.0, "seasonal": float(seasonal.mean()) if len(seasonal) else 0.0})
    return pd.DataFrame(rows)


def metrics(predictions):
    y = predictions["actual"].to_numpy()
    out = {"out_of_sample_months": len(y)}
    for name in ["ridge", "zero", "seasonal"]:
        pred = predictions[name].to_numpy()
        out[name] = {"mae": float(np.abs(pred-y).mean()), "rmse": float(np.sqrt(np.mean((pred-y)**2))),
                     "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y)))}
    best_baseline = min(out["zero"]["mae"], out["seasonal"]["mae"])
    out["mae_improvement_vs_better_baseline_pct"] = 100 * (best_baseline - out["ridge"]["mae"]) / best_baseline if best_baseline else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--worldbank", type=Path, help="World Bank Pink Sheet monthly XLSX")
    source.add_argument("--prices", type=Path, help="CSV: date,europe_usd_mmbtu,us_usd_mmbtu")
    ap.add_argument("--storage", type=Path, help="Optional GIE AGSI CSV: date,fill_pct")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--start", default="2015-04", help="TTF-based European benchmark begins April 2015")
    ap.add_argument("--min-train", type=int, default=48)
    args = ap.parse_args()
    prices = load_worldbank(args.worldbank) if args.worldbank else load_prices_csv(args.prices)
    storage = load_storage(args.storage) if args.storage else None
    panel, columns = make_panel(prices, storage, args.start)
    predictions = walk_forward(panel, columns, args.min_train)
    stats = metrics(predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "walk_forward_forecasts.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
