# European vs US natural-gas basis: walk-forward research

**Question.** Can monthly EU-US natural gas price-basis changes be forecast better than a no-change and seasonal-average baseline using only lagged price and seasonal signals? Does adding European storage fullness improve out-of-sample performance?

This is an **exploratory, reproducible research framework**, not a profitable strategy claim. It has **not** been run on the current official World Bank and GIE datasets as part of this release. Run it on the official source data and report the outcome, including negative results.

## Why the gas basis?

Europe competes for LNG cargoes with other regions, while US gas is shaped by domestic production, storage and liquefaction constraints. A high European relative price can reflect LNG demand, shipping constraints, weather or geopolitics; that does **not** automatically imply arbitrage or predictable mean reversion. The code deliberately poses a falsifiable forecasting question, not a position recommendation.

## Data: verified, attributable sources

1. **World Bank Pink Sheet, monthly gas prices.** Download `CMO-Historical-Data-Monthly.xlsx` from https://www.worldbank.org/en/research/commodity-markets . The script reads the **Monthly Prices** sheet, `NGAS_EUR` / `NGAS_US` (or labelled equivalents). Prices are $/MMBtu. The Europe series measures TTF from April 2015; **do not interpret the earlier import-border series as continuous TTF**. Default sample starts April 2015.
2. **Optional GIE AGSI European storage fullness.** Download historical data from https://agsi.gie.eu/ or register for their free API, which requires an API key. Prepare `date,fill_pct` daily or monthly with numeric fill percentage (0 to 100). The parser takes the **last observation of each month**. Credit GIE AGSI when publishing results.
3. **Next research extensions (not included):** weather-degree days, ENTSOG pipeline nominations, LNG terminal send-out (GIE ALSI), TTF and Henry Hub futures curves, shipping and regasification economics, carbon and power prices. Do not call the price-only prototype a physical-flow or live-futures trading model.

## Method and falsification

At each month end, observe EU and US monthly price averages and compute `log(EU/US)`; predict **next month's change** in that log basis. Features: 12-month rolling basis z-score, current EU and US returns, winter flag, annual Fourier seasonal terms, and optional storage deficit. Train **Ridge regression** on an expanding window (minimum 48 feature-complete months), with no shuffle or future data. Compare against the zero-change and **training-only** same-calendar-month mean. Write every out-of-sample forecast to CSV.

**Reject the research thesis** if the model does not reduce out-of-sample MAE by at least 5% versus the *better* baseline, or if the improvement disappears in pre/post-crisis holdouts or after data-revision and release-lag checks. Thresholds are predeclared research criteria, **not established results**. Inspect residuals, regime shifts and structural breaks before interpreting any correlation. A negative result is a useful outcome.

**Crucial limitations:** the World Bank is a monthly retrospective **spot-price proxy**, not a tradeable TTF-Henry Hub futures spread; it cannot validate P&L, executable signals, LNG shipping arbitrage, storage optionality or same-day decision-making. Follow-up on ICE TTF & Henry Hub contracts requires point-in-time futures curves, transaction costs, liquidity, FX/unit conversions, and contract roll conventions. The World Bank release lag must be modelled before any live-signal claim. This script gives predictive diagnostics only.

## Run

```bash
python -m pip install -r requirements.txt
python gas_basis.py --worldbank /path/to/CMO-Historical-Data-Monthly.xlsx --out outputs/
python gas_basis.py --worldbank /path/to/CMO-Historical-Data-Monthly.xlsx --storage path/to/gie_agsi.csv --out outputs_with_storage/
pytest -q
```

Alternative prepared-price input: `python gas_basis.py --prices prices.csv` where CSV columns are `date,europe_usd_mmbtu,us_usd_mmbtu`.

Outputs are `walk_forward_forecasts.csv` and `metrics.json`. Do not copy forward the output from a simulated test as empirical evidence. The tests use random **synthetic data only** to verify implementation, input validation and absence of obvious look-ahead.

**Author:** Gouru Jeevan Reddy | https://github.com/gourujeevan-cpu  
**Source attribution:** World Bank Commodity Price Data (Pink Sheet); GIE AGSI (optional). No proprietary market data included.
