"""Driver #1 add-on: what a 1-point mortgage-rate drop does to payments, and whether it
is likely to revive new-home purchasing.

Run after affordability.py (it reuses that module's cached FRED pulls and assumptions):
    .venv/bin/python rate_demand.py

Three parts:
  1. Rate band. Maps the DGS10 random-walk band (dgs10_trend_forecast.ipynb) to mortgage
     rates using the FULL current spread, held constant (stated once, below).
  2. Payment scenario. 7.18% (latest PMMS week) vs 6.18% on the Sep-2026 median new home.
  3. Demand response. (a) Every past episode since 1972 in which the 30-yr rate fell
     1 point or more from its 12-month high, and what new-home sales did afterwards.
     (b) A regression of 12-month changes in new-home sales on 12-month changes in the
     mortgage rate, controlling for the change in unemployment.

Everything in parts 1-2 is [Scenario]. Part 3 describes historical association, not a
forecast: rates usually fall when the economy weakens, which is why unemployment is controlled.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import affordability as a

OUT = a.OUT_DIR
RATE_NOW = 7.18                 # MORTGAGE30US, week of 2026-11-05 [Observed]
RATE_DROP = 1.0                 # scenario size, percentage points
RATE_LOW = round(RATE_NOW - RATE_DROP, 2)
PRICE = a.REFERENCE_PRICE       # MSPNHSUS Sep-2026 [Observed]
DROP_THRESHOLD = 1.0            # episode = rate at least 1pp below its trailing 12-month high
RECESSION_UNRATE_RISE = 1.0     # tag an episode "recession" if UNRATE rose >= 1pp over it
HAC_LAGS = 12                   # overlapping 12-month changes -> Newey-West with 12 lags
REGRESSION_START = "1976-01-01"


def load(sid: str) -> pd.Series:
    path = a._latest_raw(sid)
    if path is None:
        path = a.fetch_series(sid)[1]
    return a.parse_fred_csv(path, sid)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# --------------------------------------------------------------------------
def rate_band(dgs10: pd.Series, mort_w: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Map the DGS10 random-walk band to mortgage rates.

    ASSUMPTION (stated once): the mortgage-Treasury spread stays at its FULL current level,
    MORTGAGE30US on 2026-11-05 minus DGS10 on the same day, for the whole horizon. This is
    wider than the Oct-2026 monthly average (2.20pp) and the 2018-19 norm, so any narrowing
    would lower mortgage rates further than shown.
    """
    d = mort_w.index[-1]
    spread = mort_w.iloc[-1] - dgs10.loc[d]
    ytd = dgs10.loc["2026-01-01":]
    sigma = ytd.diff().dropna().std()          # same estimator as the notebook's random walk
    last = ytd.iloc[-1]
    rows = []
    for label, h in [("6 months", 126), ("12 months", 252)]:
        s = sigma * math.sqrt(h)
        rows.append(dict(
            horizon=label, trading_days=h, dgs10_central=last,
            mortgage_central=last + spread,
            mortgage_lo80=last + spread - 1.2816 * s, mortgage_hi80=last + spread + 1.2816 * s,
            mortgage_lo95=last + spread - 1.96 * s, mortgage_hi95=last + spread + 1.96 * s,
            prob_mortgage_at_or_below_6_18=norm_cdf(-RATE_DROP / s)))
    meta = dict(spread_date=str(d.date()), spread_pp=spread, dgs10_last=last, sigma_daily=sigma)
    return pd.DataFrame(rows), meta


# --------------------------------------------------------------------------
def payment_scenario(income_hh: float, income_nowcast: float, hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate in (RATE_NOW, RATE_LOW):
        pi = a.monthly_pi(PRICE, rate)
        total = a.total_at(PRICE, rate)
        gap_to_6 = pi - a.monthly_pi(PRICE, a.BASE_RATE)
        rows.append(dict(
            rate_pct=rate, pi_payment=pi, total_payment=total,
            total_to_income_hh=total / (income_hh / 12),
            total_to_income_nowcast=total / (income_nowcast / 12),
            price_supported_at_7_18_payment=a.price_supported(a.monthly_pi(PRICE, RATE_NOW), rate),
            buydown_gap_to_6pct_per_month=gap_to_6,
            buydown_cost_to_6pct_pct_price=a.pv_gap(gap_to_6, rate, 84) / PRICE))
    df = pd.DataFrame(rows)
    # When was total/income last at the 6.18% level? (observed history, our measure)
    lvl = df.loc[1, "total_to_income_hh"]
    below = hist[hist["total_to_income"] <= lvl]
    df["last_month_ratio_this_low"] = [None, f"{below.index[-1]:%Y-%m}" if len(below) else None]
    return df


# --------------------------------------------------------------------------
def find_episodes(rate_m: pd.Series, sales3: pd.Series, unrate: pd.Series) -> pd.DataFrame:
    """Months where the rate first sits >= 1pp below its trailing 12-month high.
    A new episode needs 12 months without a qualifying month."""
    high12 = rate_m.rolling(13, min_periods=13).max()   # t-12..t
    hits = rate_m[(rate_m <= high12 - DROP_THRESHOLD)].index
    rows, last_hit = [], None
    for t in hits:
        if last_hit is not None and (t - last_hit).days < 365:
            last_hit = t
            continue
        last_hit = t
        window = rate_m.loc[t - pd.DateOffset(months=12):t]
        peak = window.idxmax()
        def sales_at(d):
            return sales3.get(d, np.nan)
        s0 = sales_at(peak)
        rows.append(dict(
            crossing_month=f"{t:%Y-%m}", rate_peak_month=f"{peak:%Y-%m}",
            rate_at_peak=rate_m[peak], rate_at_crossing=rate_m[t],
            sales3_at_peak=s0,
            sales_chg_peak_to_cross=sales_at(t) / s0 - 1,
            sales_chg_peak_to_cross_plus6=sales_at(t + pd.DateOffset(months=6)) / s0 - 1,
            sales_chg_peak_to_cross_plus12=sales_at(t + pd.DateOffset(months=12)) / s0 - 1,
            unrate_chg_peak_to_cross_plus12=unrate.get(t + pd.DateOffset(months=12), np.nan)
                                             - unrate.get(peak, np.nan)))
    ep = pd.DataFrame(rows)
    ep["recession_tag"] = ep["unrate_chg_peak_to_cross_plus12"] >= RECESSION_UNRATE_RISE
    return ep


def near_miss_2025(rate_m: pd.Series, sales3: pd.Series) -> dict:
    """The most recent decline (Jan-2025 to Feb-2026, -0.91pp) misses the 1pp cut-off but is
    the closest analogue to today, so report it separately rather than drop it silently."""
    p, t = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-02-01")
    return dict(rate_peak_month="2025-01", trough_month="2026-02",
                rate_at_peak=rate_m[p], rate_at_trough=rate_m[t],
                sales_chg=sales3[t] / sales3[p] - 1,
                sales_chg_plus6=sales3[t + pd.DateOffset(months=6)] / sales3[p] - 1)


# --------------------------------------------------------------------------
def regression(rate_m: pd.Series, sales: pd.Series, unrate: pd.Series) -> pd.DataFrame:
    """dlog(sales, 12m) on d(rate, 12m, lagged k months) + d(UNRATE, 12m). HAC(12) errors."""
    out = []
    ls = np.log(sales)
    for start, label in [(REGRESSION_START, "1976-2026"), ("2000-01-01", "2000-2026")]:
        for lag in (0, 3, 6):
            df = pd.DataFrame({
                "dlog_sales": ls - ls.shift(12),
                "d_rate": (rate_m - rate_m.shift(12)).shift(lag),
                "d_unrate": unrate - unrate.shift(12),
            }).loc[start:].dropna()
            X = sm.add_constant(df[["d_rate", "d_unrate"]])
            m = sm.OLS(df["dlog_sales"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            b, se = m.params["d_rate"], m.bse["d_rate"]
            eff = math.exp(-b) - 1                      # sales change for a -1pp rate move
            out.append(dict(sample=label, rate_lag_months=lag, n=int(m.nobs),
                            coef_d_rate=b, se_hac=se, t_stat=b / se,
                            coef_d_unrate=m.params["d_unrate"], r2=m.rsquared,
                            sales_effect_of_minus_1pp=eff,
                            effect_lo95=math.exp(-(b + 1.96 * se)) - 1,
                            effect_hi95=math.exp(-(b - 1.96 * se)) - 1))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def charts(rate_m: pd.Series, sales: pd.Series, ep: pd.DataFrame, nm: dict) -> None:
    # 1. Two stacked panels on a shared time axis (no dual y-axis).
    r, s = rate_m.loc["2018-01-01":], sales.loc["2018-01-01":]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(r.index, r, color=a.GOLD, lw=2)
    a._style(ax1, "30-yr rate, %")
    ax1.set_title("Rates fell twice since 2023; new-home sales did not respond", loc="left")
    ax2.plot(s.index, s, color=a.NAVY, lw=1.2, alpha=0.45)
    ax2.plot(s.index, s.rolling(3).mean(), color=a.NAVY, lw=2, label="3-month average")
    a._style(ax2, "New-home sales, k SAAR")
    ax2.legend(frameon=False, loc="upper right", fontsize=8)
    for p, t in [("2023-10-01", "2024-09-01"), ("2025-01-01", "2026-02-01")]:
        for ax in (ax1, ax2):
            ax.axvspan(pd.Timestamp(p), pd.Timestamp(t), color=a.GREY, alpha=0.18, lw=0)
    ax1.text(pd.Timestamp("2023-10-15"), r.max() - 0.3, "−1.4pp", fontsize=8, color="#333333")
    ax1.text(pd.Timestamp("2025-01-15"), r.max() - 0.3, "−0.9pp", fontsize=8, color="#333333")
    fig.text(0.01, 0.005, "Sources: Freddie Mac PMMS via FRED (MORTGAGE30US, monthly mean); "
             "Census via FRED (HSN1F). Shaded: rate peak to trough.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig_sales_vs_rate.png", dpi=160); plt.close(fig)

    # 2. Past >=1pp drops: sales change 12 months after the crossing, from the rate peak.
    e = ep.dropna(subset=["sales_chg_peak_to_cross_plus12"]).copy()
    e["label"] = [f"{p} → {c}  ({rp:.1f}% → {rc:.1f}%)" for p, c, rp, rc in
                  zip(e.rate_peak_month, e.crossing_month, e.rate_at_peak, e.rate_at_crossing)]
    e = e.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(e) + 1.6))
    colors = [a.GREY if rec else a.NAVY for rec in e.recession_tag]
    ax.barh(e["label"], e["sales_chg_peak_to_cross_plus12"] * 100, color=colors, height=0.62)
    ax.axvline(0, color="#333333", lw=0.8)
    lo = (e["sales_chg_peak_to_cross_plus12"].min() * 100)
    ax.set_xlim(lo - 8, e["sales_chg_peak_to_cross_plus12"].max() * 100 + 8)
    for y, v in enumerate(e["sales_chg_peak_to_cross_plus12"] * 100):
        ax.annotate(f"{v:+.1f}%" if abs(v) < 1 else f"{v:+.0f}%", (v, y), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color="#333333")
    ax.set_xlabel("New-home sales (3-mo avg), % change from rate peak to 12 months after the 1pp drop")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=a.NAVY, label="Unemployment stable"),
                       Patch(color=a.GREY, label=f"Unemployment up ≥{RECESSION_UNRATE_RISE:.0f}pp over the window")],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Every 1-point mortgage-rate drop since 1972, and what new-home sales did", loc="left")
    fig.text(0.01, 0.005, "Sources: MORTGAGE30US, HSN1F, UNRATE via FRED. Episode = monthly rate ≥1pp below its "
             "12-month high. The 2018-19 episode is grey because of the 2020 COVID job losses.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig_rate_drop_events.png", dpi=160); plt.close(fig)



def rate_charts(mort_w: pd.Series, dgs10: pd.Series, meta: dict) -> dict:
    """Mortgage-rate forecast fan (DGS10 random walk + full spread) and spread history."""
    out = {}
    # --- Fan chart ---
    hist = mort_w.loc["2024-01-01":]
    last_d, last_v = hist.index[-1], hist.iloc[-1]
    sig = meta["sigma_daily"]
    days = np.arange(0, 253, 5)                       # trading days ahead, weekly steps
    dates = pd.bdate_range(last_d, periods=len(days) * 5, freq="B")[::5][:len(days)]
    s = sig * np.sqrt(days)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(dates, last_v - 1.96 * s, last_v + 1.96 * s, color=a.NAVY, alpha=0.12, lw=0, label="95% range")
    ax.fill_between(dates, last_v - 1.2816 * s, last_v + 1.2816 * s, color=a.NAVY, alpha=0.22, lw=0, label="80% range")
    ax.plot(hist.index, hist, color=a.NAVY, lw=2, label="30-yr mortgage rate, weekly (MORTGAGE30US)")
    ax.plot(dates, np.full(len(dates), last_v), color=a.NAVY, lw=1.5, ls="--", label="Central case: unchanged")
    ax.axhline(RATE_LOW, color=a.GOLD, lw=1.5)
    p12 = norm_cdf(-RATE_DROP / (sig * math.sqrt(252)))
    ax.text(hist.index[0], RATE_LOW - 0.17, f"{RATE_LOW:.2f}%: the 1-point drop. About {p12:.0%} odds within 12 months",
            fontsize=8, color="#333333")
    a._style(ax, "%")
    ax.set_title("The base case is no change: mortgage-rate range for the next 12 months", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.text(0.01, 0.005, f"Sources: MORTGAGE30US, DGS10 via FRED. Random walk on DGS10 plus the full spread "
             f"({meta['spread_pp']:.2f} pts) held constant. Backtest says ranges are likely too narrow.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_mortgage_rate_fan.png", dpi=160); plt.close(fig)

    # --- Spread history ---
    mm = mort_w.resample("MS").mean(); tm = dgs10.resample("MS").mean()
    spread = (mm - tm).loc["2000-01-01":].dropna()
    norm = spread.loc["2010-01-01":"2019-12-01"].mean()
    out.update(spread_norm_2010_19=norm, spread_oct=spread.iloc[-2], spread_latest_month=spread.index[-1])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(spread.index, spread, color=a.NAVY, lw=2, label="30-yr mortgage minus 10-yr Treasury, monthly")
    ax.axhline(norm, color=a.GREY, lw=1, ls="--", label=f"2010-19 average ({norm:.2f} pts)")
    a._style(ax, "percentage points")
    ax.set_title("The mortgage-Treasury spread is still wide of its pre-2020 norm", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.005, "Sources: MORTGAGE30US, DGS10 via FRED (monthly means).", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_mortgage_spread.png", dpi=160); plt.close(fig)
    return out

# --------------------------------------------------------------------------
def main() -> None:
    mort_w, dgs10 = load("MORTGAGE30US"), load("DGS10")
    sales, unrate = load("HSN1F"), load("UNRATE")
    rate_m = mort_w.resample("MS").mean()
    sales3 = sales.rolling(3).mean()      # new-home sales are noisy month to month

    band, meta = rate_band(dgs10, mort_w)

    monthly = pd.read_csv(OUT / "affordability_monthly.csv", index_col=0, parse_dates=True)
    hist = monthly.dropna(subset=["total_to_income"])
    # Income for the latest complete month (Oct-2026), from affordability.py's current estimate
    kn = pd.read_csv(OUT / "key_numbers.csv").set_index("metric")["value"]
    hh_income = float(kn["median_hh_income_used"])
    nowcast = float(kn["current_hh_income_nowcast"])
    scen = payment_scenario(hh_income, nowcast, hist)

    ep = find_episodes(rate_m.loc[:"2026-10-01"], sales3, unrate)
    nm = near_miss_2025(rate_m, sales3)
    reg = regression(rate_m.loc[:"2026-10-01"], sales, unrate)

    band.to_csv(OUT / "mortgage_rate_band.csv", index=False)
    scen.to_csv(OUT / "rate_drop_scenario.csv", index=False)
    ep.to_csv(OUT / "rate_drop_episodes.csv", index=False)
    reg.to_csv(OUT / "rate_sales_regression.csv", index=False)
    charts(rate_m, sales, ep, nm)
    sp = rate_charts(mort_w, dgs10, meta)
    print("Spread norm 2010-19:", round(sp["spread_norm_2010_19"], 3), " Oct-2026:", round(sp["spread_oct"], 3))

    pd.DataFrame([
        ("spread_full_pp", meta["spread_pp"], "MORTGAGE30US − DGS10", meta["spread_date"]),
        ("dgs10_sigma_daily", meta["sigma_daily"], "DGS10 2026 YTD daily changes", "2026-11-05"),
        ("near_miss_2025_sales_chg", nm["sales_chg"], "HSN1F 3-mo avg, 2025-01 to 2026-02", "2026-02"),
        ("near_miss_2025_sales_chg_plus6", nm["sales_chg_plus6"], "HSN1F 3-mo avg, 2025-01 to 2026-08", "2026-08"),
        ("hsn1f_latest", sales.iloc[-1], "HSN1F", f"{sales.index[-1]:%Y-%m}"),
        ("hsn1f_3mo_latest", sales3.iloc[-1], "HSN1F 3-mo avg", f"{sales.index[-1]:%Y-%m}"),
        ("unrate_latest", unrate.iloc[-1], "UNRATE", f"{unrate.index[-1]:%Y-%m}"),
        ("spread_norm_2010_19", sp["spread_norm_2010_19"], "MORTGAGE30US − DGS10, monthly means", "2010-01..2019-12"),
    ], columns=["metric", "value", "source", "obs_date"]).to_csv(
        OUT / "rate_demand_key_numbers.csv", index=False)

    # Append the two new series to the source log (affordability.py rewrites it each run,
    # so run this script after it).
    log = pd.read_csv(OUT / "source_log.csv")
    log = log[~log.series_id.isin(["HSN1F", "UNRATE"])]
    for sid, desc, s in [("HSN1F", "New one-family houses sold (thousands, SAAR)", sales),
                         ("UNRATE", "Civilian unemployment rate (%, SA)", unrate)]:
        p = a._latest_raw(sid)
        log.loc[len(log)] = dict(
            source="FRED", series_id=sid, description=desc, frequency="monthly",
            first_obs=s.index.min().date(), last_obs=s.index.max().date(), last_value=s.iloc[-1],
            vintage="FRED current vintage as of access date (not ALFRED real-time)",
            accessed=p.stem.split("_", 1)[1][:10], raw_file=str(p.relative_to(a.ROOT)),
            url=f"https://fred.stlouisfed.org/series/{sid}", used_as="data (rate_demand.py)")
    log.to_csv(OUT / "source_log.csv", index=False)

    pd.set_option("display.width", 200)
    print("Spread:", meta)
    print(band.round(3).to_string(index=False))
    print(scen.round(4).to_string(index=False))
    print(ep.round(3).to_string(index=False))
    print("2025 near miss:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in nm.items()})
    print(reg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
