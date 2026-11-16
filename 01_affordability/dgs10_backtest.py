"""Backtest of central-case forecasts for the 10-year Treasury yield (DGS10).

Question: does any trend model beat a no-change (random walk) forecast of the
10-year yield at 6- and 12-month horizons? Supports the Driver 1 claim that the
"no change" central case is the right one.

Design (mirrors the spirit of the mortgage-rate interval backtest in
rate_prediction_interval.py):
- DGS10 daily, averaged to monthly, 1962-2026.
- Expanding window: at each month-end origin from 1990-01 on, fit each model on
  data available to that date only, forecast 6 and 12 months ahead.
- Models: random walk (no change), random walk with drift, linear trend on the
  last 10 years, Holt damped trend, ARIMA(1,1,1).
- Error: MAE and RMSE of the central forecast vs the actual monthly average.
- Overlap note: 12-month windows overlap, so the number of independent
  observations is roughly origins/12; comparisons are indicative.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

RAW = "data/raw/DGS10_2026-11-09.csv"  # run from 01_affordability

df = pd.read_csv(RAW, parse_dates=["observation_date"])
df = df.dropna()
m = df.set_index("observation_date")["DGS10"].resample("MS").mean().dropna()
print(f"Monthly obs: {len(m)}, {m.index[0]:%Y-%m} to {m.index[-1]:%Y-%m}")

HORIZONS = [6, 12]
START = "1990-01-01"
origins = m.index[(m.index >= START) & (m.index <= m.index[-1] - pd.DateOffset(months=max(HORIZONS)))]
print(f"Origins: {len(origins)} ({origins[0]:%Y-%m} to {origins[-1]:%Y-%m})")


def forecasts(hist, hs):
    """Return dict model -> {h: forecast}."""
    out = {}
    last = hist.iloc[-1]
    n = len(hist)
    out["Random walk (no change)"] = {h: last for h in hs}
    # RW with drift: mean monthly change over full history
    drift = (hist.iloc[-1] - hist.iloc[0]) / (n - 1)
    out["Random walk with drift"] = {h: last + drift * h for h in hs}
    # Linear trend, last 10 years
    win = hist.iloc[-120:]
    t = np.arange(len(win))
    b, a = np.polyfit(t, win.values, 1)
    out["Linear trend (10y)"] = {h: a + b * (len(win) - 1 + h) for h in hs}
    # Holt damped trend
    try:
        fit = ExponentialSmoothing(hist.values, trend="add", damped_trend=True).fit()
        f = fit.forecast(max(hs))
        out["Damped trend (Holt)"] = {h: f[h - 1] for h in hs}
    except Exception:
        out["Damped trend (Holt)"] = {h: np.nan for h in hs}
    # ARIMA(1,1,1)
    try:
        fit = ARIMA(hist.values, order=(1, 1, 1)).fit()
        f = fit.forecast(max(hs))
        out["ARIMA(1,1,1)"] = {h: f[h - 1] for h in hs}
    except Exception:
        out["ARIMA(1,1,1)"] = {h: np.nan for h in hs}
    return out


rows = []
for i, orig in enumerate(origins):
    hist = m.loc[:orig]
    preds = forecasts(hist, HORIZONS)
    for h in HORIZONS:
        target = orig + pd.DateOffset(months=h)
        if target not in m.index:
            continue
        actual = m.loc[target]
        for model, p in preds.items():
            rows.append({"origin": orig, "horizon_months": h, "model": model,
                         "forecast": p[h], "actual": actual, "error": p[h] - actual})
    if i % 50 == 0:
        print(f"  {i}/{len(origins)} origins done")

res = pd.DataFrame(rows).dropna(subset=["forecast"])

summary = (res.groupby(["horizon_months", "model"])
           .agg(origins=("error", "size"),
                mae=("error", lambda e: np.abs(e).mean()),
                rmse=("error", lambda e: np.sqrt((e ** 2).mean())))
           .reset_index())

# ratio vs random walk
rw = summary[summary.model == "Random walk (no change)"].set_index("horizon_months")
summary["mae_vs_rw"] = summary.apply(lambda r: r["mae"] / rw.loc[r["horizon_months"], "mae"], axis=1)
summary["rmse_vs_rw"] = summary.apply(lambda r: r["rmse"] / rw.loc[r["horizon_months"], "rmse"], axis=1)
summary["eff_indep_obs"] = (summary["origins"] / summary["horizon_months"]).round(1)

summary = summary.sort_values(["horizon_months", "mae"])
pd.set_option("display.width", 160)
print(summary.round(3).to_string(index=False))

summary.round(4).to_csv("outputs/dgs10_backtest.csv", index=False)
print("saved outputs/dgs10_backtest.csv")
