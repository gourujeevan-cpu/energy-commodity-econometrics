"""
Data acquisition and preparation for energy commodity price series.

The module is deliberately commodity-agnostic: every routine works on a single
price column, so the same pipeline runs on Henry Hub natural gas, Brent, WTI or
any other series you point it at.

Two data paths are supported:

1.  ``load_local``  -- read a bundled CSV snapshot (works fully offline; this is
    the default so the repository is reproducible without network access).
2.  ``fetch_fred``  -- pull a fresh series from the St. Louis Fed (FRED) via its
    public CSV endpoint. No API key required.

FRED series IDs for the headline energy benchmarks:

    Henry Hub natural gas spot ($/MMBtu) ...... DHHNGSP
    Brent crude spot ($/bbl) .................. DCOILBRENTEU
    WTI crude spot ($/bbl) .................... DCOILWTICO

Source of the bundled ``henry_hub_spot.csv`` snapshot: U.S. Energy Information
Administration (EIA), Henry Hub Natural Gas Spot Price, daily, public domain.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Friendly name -> FRED series id, for the live-fetch path.
COMMODITIES: dict[str, str] = {
    "henry_hub": "DHHNGSP",
    "brent": "DCOILBRENTEU",
    "wti": "DCOILWTICO",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class PriceSeries:
    """A cleaned commodity price series with returns attached.

    Attributes
    ----------
    name : str
        Human-readable label used in plots and table captions.
    frame : pandas.DataFrame
        Business-day indexed frame with columns ``price``, ``log_price`` and
        ``ret`` (percentage log returns).
    """

    name: str
    frame: pd.DataFrame

    @property
    def price(self) -> pd.Series:
        return self.frame["price"]

    @property
    def ret(self) -> pd.Series:
        return self.frame["ret"]

    @property
    def log_price(self) -> pd.Series:
        return self.frame["log_price"]

    def describe(self) -> pd.Series:
        """Return a compact set of summary statistics for the return series."""
        r = self.ret
        return pd.Series(
            {
                "start": self.frame.index.min().date().isoformat(),
                "end": self.frame.index.max().date().isoformat(),
                "n_obs": int(len(self.frame)),
                "price_min": round(float(self.price.min()), 3),
                "price_max": round(float(self.price.max()), 3),
                "ret_mean_ann_%": round(float(r.mean()) * 252, 3),
                "ret_vol_ann_%": round(float(r.std()) * np.sqrt(252), 2),
                "ret_skew": round(float(r.skew()), 3),
                "ret_exkurt": round(float(r.kurtosis()), 2),
            }
        )


def _prepare(raw: pd.DataFrame, name: str) -> PriceSeries:
    """Clean a two-column (date, price) frame into a ``PriceSeries``.

    Steps: parse dates, sort ascending, forward-fill sparse gaps, reindex onto a
    business-day calendar, then build log prices and percentage log returns.
    Percentage returns (x100) are used because GARCH optimisers are far better
    conditioned when the data are O(1) rather than O(0.01).
    """
    df = raw.copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").ffill()
    df = df.asfreq("B").ffill()
    df["log_price"] = np.log(df["price"])
    df["ret"] = 100.0 * df["log_price"].diff()
    df = df.dropna()
    return PriceSeries(name=name, frame=df)


def load_local(filename: str = "henry_hub_spot.csv", name: str = "Henry Hub") -> PriceSeries:
    """Load a bundled CSV snapshot from the ``data/`` directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Use fetch_fred(...) to download, or check the path."
        )
    return _prepare(pd.read_csv(path), name=name)


def fetch_fred(commodity: str = "henry_hub", start: str = "1997-01-01") -> PriceSeries:
    """Download a daily series from FRED's public CSV endpoint.

    Requires internet access. ``commodity`` may be a key of ``COMMODITIES``
    (``henry_hub``, ``brent``, ``wti``) or a raw FRED series id.
    """
    series_id = COMMODITIES.get(commodity, commodity)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urlopen(url, timeout=30) as resp:  # noqa: S310 - trusted, fixed host
        raw = pd.read_csv(io.BytesIO(resp.read()))
    raw.columns = ["date", "price"]
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw[raw["date"] >= start]
    label = {"henry_hub": "Henry Hub", "brent": "Brent", "wti": "WTI"}.get(
        commodity, series_id
    )
    return _prepare(raw, name=label)


def load(commodity: str = "henry_hub", prefer_live: bool = False) -> PriceSeries:
    """Convenience entry point.

    With ``prefer_live=True`` the loader tries FRED first and silently falls back
    to the bundled snapshot if the network is unavailable, so notebooks remain
    runnable in either environment.
    """
    if prefer_live:
        try:
            return fetch_fred(commodity)
        except Exception:  # pragma: no cover - network dependent
            pass
    return load_local()


if __name__ == "__main__":
    ps = load_local()
    print(ps.describe().to_string())
