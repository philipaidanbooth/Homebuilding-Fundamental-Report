"""
Homebuilding case study, driver #1: mortgage affordability (public data only).

What it does
  1. Pulls public series from FRED (CSV endpoint, no API key).
  2. Caches each raw pull to data/raw/<SERIES>_<access-date>.csv (one file per pull
     date; existing raw files are never overwritten) and writes outputs/source_log.csv.
  3. Builds a monthly table: rate, median new-home price, payment, and payment-to-income
     under three income measures.
  4. Builds a payment-vs-rate sensitivity table and the 6.00% vs 7.18% payment gap.
  5. Cross-checks payment-to-income against the Atlanta Fed HOAM (public xlsx).
  6. Loads the NAHB/Wells Fargo HMI headline history (public xls) and the manual
     price-cut / incentive CSV (data/manual/), since those shares have no
     machine-readable public source.
  7. Saves charts, CSVs and outputs/key_numbers.csv (every headline number with its
     series ID and observation date) to outputs/.

Run:
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python affordability.py            # use today's raw pull if it exists, else download
    .venv/bin/python affordability.py --refresh  # force a new download (still a new dated file)

Everything below marked ASSUMPTION must also appear in the methodology section.
Observed data and scenario assumptions are kept in separate columns/tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# ASSUMPTIONS (state each one in the methodology section)
# --------------------------------------------------------------------------
START_DATE = "2018-01-01"
DOWN_PAYMENT_PCT = 0.20          # ASSUMPTION: 20% down, no PMI. Many new-home buyers put
                                 # down less (FHA/VA), which raises the payment.
TERM_MONTHS = 360                # ASSUMPTION: 30-year fixed, Freddie Mac PMMS rate, no points.

# ASSUMPTION (sourced): national effective property tax rate on single-family homes, 2025.
#   ATTOM "2025 Property Tax Analysis", released 2026-04-09: "the effective tax rate for
#   single-family homes in 2025 was 0.9 percent, up from 0.86 percent in 2024".
#   Method: average tax bill ($4,427) / average AVM value ($494,231), ~89.6M homes.
#   https://www.attomdata.com/news/market-trends/home-sales-prices/2025-annual-tax-report
#   Accessed 2026-11-09. Caveats: (a) ATTOM covers the whole single-family stock, not new
#   homes (new homes often carry MUD/PID/CDD special assessments in TX/FL, which this
#   misses); (b) we hold 0.90% constant across 2018-2026, while ATTOM reported 1.1% for 2020,
#   so historical tax costs are slightly understated.
PROPERTY_TAX_RATE = 0.009

# ASSUMPTION (PLACEHOLDER, FLAGGED): homeowners insurance as % of price per year.
#   No credible public figure for *paid* premiums as a share of value is current: the
#   NAIC Homeowners report lags ~3 years (latest covers 2022). The best public current
#   figure is Bankrate/Quadrant ($2,470/yr on a $300k-dwelling profile, released
#   2025-08-25), but those are quoted sample rates, not premiums paid, and dwelling
#   coverage excludes land. On the Sep-2026 median new home ($400,400) that would be
#   ~0.62% of price. Kept at the original 0.50% placeholder; findings show both.
INSURANCE_RATE = 0.005
INSURANCE_RATE_ALT = 2470 / 400_400   # Bankrate-implied alternative, for sensitivity only

HOURS_PER_YEAR = 2080            # ASSUMPTION: wage-proxy income = AHE x 40h x 52wk
SENSITIVITY_RATES = [5.5, 6.0, 6.5, 7.0, 7.18, 7.5, 8.0]   # scenario rates (assumptions)
BASE_RATE, STRESS_RATE = 6.0, 7.18
# Sept-2026 median new-home price (Census, FRED MSPNHSUS, monthly). The script checks
# this constant against the pulled data and warns if they differ.
REFERENCE_PRICE = 400_400
REFERENCE_PRICE_MONTH = "2026-09-01"

# FRED series: id -> (description, native frequency)
SERIES = {
    "MORTGAGE30US": ("Freddie Mac PMMS 30-yr fixed mortgage rate (%)", "weekly"),
    "DGS10": ("10-year Treasury constant maturity yield (%)", "daily"),
    # MSPNHSUS added: the original script used quarterly MSPUS, which cannot produce the
    # monthly Sept-2026 reference price and made the monthly table stale. MSPUS is kept
    # as a cross-check column.
    "MSPNHSUS": ("Median sales price of new houses sold, US ($, monthly, NSA)", "monthly"),
    "MSPUS": ("Median sales price of houses sold, US ($, quarterly, NSA)", "quarterly"),
    "MEHOINUSA646N": ("Median household income, US ($, annual, current dollars)", "annual"),
    "CES0500000003": ("Avg hourly earnings, all private employees ($, SA)", "monthly"),
    "PSAVERT": ("Personal saving rate (%)", "monthly"),
    "MSACSR": ("Monthly supply of new houses (months, SA)", "monthly"),
}

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
EXT_DIR = RAW_DIR / "external"
MANUAL_DIR = ROOT / "data" / "manual"
OUT_DIR = ROOT / "outputs"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
TODAY = dt.date.today().isoformat()

HOAM_URL = ("https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
            "research-and-data/data/hoam/HOAM_US_Affordability_Index.xlsx")
NAHB_T2_URL = ("https://www.nahb.org/-/media/NAHB/news-and-economics/docs/housing-economics/"
               "hmi/{vintage}/t2-national-hmi-history-{vintage_compact}.xls")
NAHB_VINTAGE = "2026-10"         # update monthly when NAHB posts a new release

NAVY, GOLD, GREY = "#1b3550", "#d39b2f", "#8a8f98"
DATE_COLS = ("observation_date", "DATE", "date")


# --------------------------------------------------------------------------
# Data pull
# --------------------------------------------------------------------------
def _download(url: str, dest: Path, browser_ua: bool = False) -> None:
    # FRED stalls requests carrying a browser-style User-Agent (observed 2026-11-09), so
    # FRED pulls use Python's default UA; the Atlanta Fed and NAHB sites want a browser UA.
    headers = {"User-Agent": "Mozilla/5.0"} if browser_ua else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def _latest_raw(sid: str) -> Path | None:
    files = sorted(RAW_DIR.glob(f"{sid}_*.csv"))
    return files[-1] if files else None


def parse_fred_csv(path: Path, sid: str) -> pd.Series:
    """Read a FRED CSV. The date column has been both 'DATE' and 'observation_date'."""
    df = pd.read_csv(path)
    date_col = next((c for c in DATE_COLS if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"{path.name}: no date column; got {list(df.columns)}")
    val_col = sid if sid in df.columns else next(c for c in df.columns if c != date_col)
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,  # '.' = missing
                  index=pd.to_datetime(df[date_col]), name=sid)
    return s.dropna().sort_index()


def fetch_series(sid: str, refresh: bool = False) -> tuple[pd.Series, Path]:
    """Return (series, raw file used). Raw pulls are written once per access date and
    never overwritten; --refresh on a day that already has a pull adds a timestamped file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{sid}_{TODAY}.csv"
    if dest.exists() and refresh:
        dest = RAW_DIR / f"{sid}_{TODAY}T{dt.datetime.now():%H%M%S}.csv"
    if refresh or not dest.exists():
        _download(FRED_CSV.format(sid=sid), dest)
    return parse_fred_csv(dest, sid), dest


def fetch_external(url: str, name: str) -> Path:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXT_DIR / name
    if not dest.exists():
        _download(url, dest, browser_ua=True)
    return dest


def pull_all(refresh: bool = False) -> tuple[dict[str, pd.Series], list[dict]]:
    data, log_rows = {}, []
    for sid, (desc, freq) in SERIES.items():
        try:
            s, path = fetch_series(sid, refresh=refresh)
        except Exception as exc:  # keep going; report which series failed
            print(f"[warn] could not fetch {sid}: {exc}")
            fallback = _latest_raw(sid)
            if fallback is None:
                continue
            print(f"[warn] using older cached pull {fallback.name}")
            s, path = parse_fred_csv(fallback, sid), fallback
        data[sid] = s
        log_rows.append(dict(
            source="FRED", series_id=sid, description=desc, frequency=freq,
            first_obs=s.index.min().date(), last_obs=s.index.max().date(),
            last_value=s.iloc[-1],
            vintage="FRED current vintage as of access date (not ALFRED real-time)",
            accessed=path.stem.split("_", 1)[1][:10], raw_file=str(path.relative_to(ROOT)),
            url=f"https://fred.stlouisfed.org/series/{sid}", used_as="data"))
    return data, log_rows


STATIC_SOURCES = [
    dict(source="ATTOM", series_id="ATTOM 2025 Property Tax Analysis",
         description="National effective property tax rate, single-family, 2025 = 0.90% "
                     "(2024 = 0.86%); avg bill $4,427 / avg AVM value $494,231",
         frequency="annual", last_obs="2025", last_value=0.009,
         vintage="released 2026-04-09", accessed="2026-11-09", raw_file="",
         url="https://www.attomdata.com/news/market-trends/home-sales-prices/2025-annual-tax-report",
         used_as="assumption PROPERTY_TAX_RATE"),
    dict(source="Bankrate / Quadrant Information Services",
         series_id="Bankrate True Cost of Home Insurance 2025",
         description="Avg annual premium $2,470 on $300k dwelling profile (quoted sample "
                     "rates, July 2025, not premiums paid)",
         frequency="annual", last_obs="2025-07", last_value=2470,
         vintage="released 2025-08-25", accessed="2026-11-09", raw_file="",
         url="https://www.bankrate.com/press-releases/national-average-cost-of-home-insurance-in-the-united-states-is-2470/",
         used_as="sensitivity only; INSURANCE_RATE remains a 0.50% placeholder (FLAG)"),
    dict(source="NAIC", series_id="NAIC Homeowners Insurance Report (2022 data)",
         description="Considered for INSURANCE_RATE; rejected as too stale (data year 2022)",
         frequency="annual", last_obs="2022", last_value="", vintage="released 2025-05-21",
         accessed="2026-11-09", raw_file="", url="https://content.naic.org/",
         used_as="not used"),
]


# --------------------------------------------------------------------------
# Calculations
# --------------------------------------------------------------------------
def monthly_pi(price, rate_pct, down_pct: float = DOWN_PAYMENT_PCT, n: int = TERM_MONTHS):
    """Fixed-rate principal & interest: L * r(1+r)^n / ((1+r)^n - 1)."""
    loan = price * (1 - down_pct)
    r = rate_pct / 100 / 12
    factor = (1 + r) ** n
    return loan * r * factor / (factor - 1)


def price_supported(payment: float, rate_pct: float,
                    down_pct: float = DOWN_PAYMENT_PCT, n: int = TERM_MONTHS) -> float:
    """Invert the payment formula: the home price a fixed P&I payment supports."""
    r = rate_pct / 100 / 12
    factor = (1 + r) ** n
    loan = payment * (factor - 1) / (r * factor)
    return loan / (1 - down_pct)


def build_monthly_table(data: dict[str, pd.Series]) -> pd.DataFrame:
    """Monthly table, observed data only (plus the fixed tax/insurance assumptions).

    Fixes vs. the original: resample(...).ffill() only fills up to a series' last
    observation, so annual income (dated Jan-1) was missing for every month after
    Jan of its last year and the 'latest' row was stuck at 2025-01. Now the table
    spans to the latest rate month and only income is carried forward (with its
    vintage year shown). Price is NOT carried forward: months without a published
    price have no payment.
    """
    rate = data["MORTGAGE30US"].resample("MS").mean()      # monthly avg of weekly PMMS
    idx = pd.date_range(START_DATE, rate.index.max(), freq="MS")
    df = pd.DataFrame(index=idx)
    df["mortgage_rate"] = rate
    df["mortgage_rate_n_weeks"] = data["MORTGAGE30US"].resample("MS").count()
    df["ust10"] = data["DGS10"].resample("MS").mean()
    df["median_new_price"] = data["MSPNHSUS"].resample("MS").last()
    # quarterly value repeated for the 3 months of its quarter (limit=2 stops it running past)
    df["median_new_price_q_MSPUS"] = data["MSPUS"].reindex(idx).ffill(limit=2)
    df["avg_hourly_earnings"] = data["CES0500000003"].resample("MS").last()
    df["saving_rate"] = data["PSAVERT"].resample("MS").last()
    df["months_supply"] = data["MSACSR"].resample("MS").last()

    # --- Income measure A: median household income (Census CPS ASEC, annual) -------
    # The value for year Y is applied to Jan-Dec of Y, and the latest year is carried
    # forward. Census publishes year Y in September of Y+1, so this is (i) look-ahead for
    # historical months and (ii) stale for months after the last published year.
    hh = data["MEHOINUSA646N"]
    hh_by_year = pd.Series(hh.values, index=hh.index.year)
    last_hh_year = int(hh_by_year.index.max())
    df["hh_income_year"] = np.minimum(df.index.year, last_hh_year)
    df["median_hh_income"] = df["hh_income_year"].map(hh_by_year)

    # --- Income measure B: wage proxy = AHE x 2080 -------------------------------
    # Current (monthly) but an individual full-time-equivalent wage, not household
    # income. It sits below household income because (1) households average more than
    # one earner, (2) it excludes non-wage income (transfers, self-employment, capital),
    # (3) AHE is a mean for private payroll workers, while household income is a median
    # across all households incl. retirees. Its LEVEL is therefore not comparable; use
    # it for direction and recent momentum.
    df["wage_proxy_income"] = df["avg_hourly_earnings"] * HOURS_PER_YEAR

    # --- Income measure C: household income nowcast (ASSUMPTION) -----------------
    # Latest published household income grown by AHE growth since that year's average.
    # Equals measure A through the last published year; diverges only afterwards.
    ahe_year_avg = df["avg_hourly_earnings"].groupby(df.index.year).mean()
    base_ahe = ahe_year_avg.loc[last_hh_year]
    nowcast = hh_by_year.loc[last_hh_year] * df["avg_hourly_earnings"] / base_ahe
    df["hh_income_nowcast"] = np.where(df.index.year <= last_hh_year,
                                       df["median_hh_income"], nowcast)

    # --- Payments ----------------------------------------------------------------
    df["mortgage_spread"] = df["mortgage_rate"] - df["ust10"]
    df["pi_payment"] = monthly_pi(df["median_new_price"], df["mortgage_rate"])
    df["tax_payment"] = df["median_new_price"] * PROPERTY_TAX_RATE / 12
    df["ins_payment"] = df["median_new_price"] * INSURANCE_RATE / 12
    df["total_payment"] = df["pi_payment"] + df["tax_payment"] + df["ins_payment"]

    for tag, col in [("", "median_hh_income"), ("_wage", "wage_proxy_income"),
                     ("_nowcast", "hh_income_nowcast")]:
        df[f"pi_to_income{tag}"] = df["pi_payment"] / (df[col] / 12)
        df[f"total_to_income{tag}"] = df["total_payment"] / (df[col] / 12)
    return df


def pv_gap(monthly_gap: float, rate_pct: float, months: int) -> float:
    """Present value of a level monthly amount, discounted at the mortgage rate."""
    r = rate_pct / 1200
    return monthly_gap * (1 - (1 + r) ** -months) / r


def total_at(price: float, rate_pct: float) -> float:
    """P&I + property tax + insurance under the module assumptions."""
    return monthly_pi(price, rate_pct) + price * (PROPERTY_TAX_RATE + INSURANCE_RATE) / 12


def build_sensitivity(ref_price: float) -> pd.DataFrame:
    """Scenario table (assumptions, not observed data)."""
    base_pay = monthly_pi(ref_price, BASE_RATE)
    rows = []
    for rt in SENSITIVITY_RATES:
        pay = monthly_pi(ref_price, rt)
        supported = price_supported(base_pay, rt)
        rows.append(dict(rate_pct=rt, pi_payment=round(pay, 0),
                         change_vs_6pct=round(pay - base_pay, 0),
                         pct_change_vs_6pct=round(pay / base_pay - 1, 4),
                         price_supported_at_6pct_payment=round(supported, 0),
                         purchasing_power_vs_6pct=round(supported / ref_price - 1, 4)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# External cross-checks
# --------------------------------------------------------------------------
def load_hoam(path: Path) -> pd.Series:
    x = pd.read_excel(path)
    x = x.rename(columns=lambda c: str(c).strip())
    col = "Annual Payment Share of Income"
    if col not in x.columns:
        raise ValueError(f"HOAM layout changed; columns: {list(x.columns)}")
    return pd.Series(x[col].values, index=pd.to_datetime(x["Month"]), name="hoam_share")


def turning_points(s: pd.Series, window: int = 6, min_move: float = 0.01) -> pd.DataFrame:
    """Local peaks/troughs of a 3-month centred average: the extreme within +/-window
    months, and at least `min_move` (in ratio points) away from the opposite extreme on
    each side. Simple and transparent, not a formal Bry-Boschan dating."""
    sm = s.rolling(3, center=True).mean().dropna()
    out = []
    for i in range(window, len(sm) - window):
        seg = sm.iloc[i - window:i + window + 1]
        v = sm.iloc[i]
        left, right = sm.iloc[:i + 1], sm.iloc[i:]
        if v == seg.max() and v - left.iloc[-window - 1:].min() >= min_move \
                and v - right.iloc[:window + 1].min() >= min_move:
            out.append((sm.index[i], "peak", v))
        elif v == seg.min() and left.iloc[-window - 1:].max() - v >= min_move \
                and right.iloc[:window + 1].max() - v >= min_move:
            out.append((sm.index[i], "trough", v))
    return pd.DataFrame(out, columns=["date", "type", "value_3mma"])


def compare_hoam(df: pd.DataFrame, hoam: pd.Series) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    both = pd.concat([df["total_to_income"].rename("ours_total_to_income"), hoam],
                     axis=1).dropna()
    d12 = both.diff(12).dropna()
    d1 = both.diff(1).dropna()
    stats = dict(
        overlap=f"{both.index.min():%Y-%m} to {both.index.max():%Y-%m}",
        n_months=len(both),
        corr_levels=both.corr().iloc[0, 1],
        corr_12m_change=d12.corr().iloc[0, 1],
        sign_agree_12m=(np.sign(d12.iloc[:, 0]) == np.sign(d12.iloc[:, 1])).mean(),
        sign_agree_1m=(np.sign(d1.iloc[:, 0]) == np.sign(d1.iloc[:, 1])).mean(),
    )
    tp = pd.concat([turning_points(both.iloc[:, 0]).assign(series="ours (new homes, Census income)"),
                    turning_points(both.iloc[:, 1]).assign(series="Atlanta Fed HOAM")])
    return both, stats, tp.sort_values("date")


def load_nahb_hmi(path: Path) -> pd.Series:
    """Parse NAHB Table 2 (rows = years, cols = Jan..Dec)."""
    raw = pd.read_excel(path, header=None)
    hdr_row = raw.index[raw.iloc[:, 1].astype(str).str.strip() == "Jan"][0]
    body = raw.iloc[hdr_row + 1:, :13].dropna(subset=[0])
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    recs = []
    for _, row in body.iterrows():
        for m in range(1, 13):
            v = pd.to_numeric(row[m], errors="coerce")
            if pd.notna(v):
                recs.append((pd.Timestamp(int(row[0]), m, 1), v))
    return pd.Series(dict(recs), name="nahb_hmi").sort_index()


def load_manual_nahb() -> pd.DataFrame:
    path = MANUAL_DIR / "nahb_hmi_price_cuts_incentives.csv"
    m = pd.read_csv(path, comment="#", parse_dates=["month"])
    return m.set_index("month").sort_index()


# --------------------------------------------------------------------------
# Charts (one y-axis each; <= 2 coloured series + neutral reference)
# --------------------------------------------------------------------------
def _style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _label_end(ax, s, text, color):
    s = s.dropna()
    ax.annotate(text, (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color="#333333")


def current_estimate(df: pd.DataFrame) -> pd.Series:
    """Latest COMPLETE rate month (Oct-2026 on 2026-11-09), not the latest month with a price.

    Rates, Treasuries and hourly earnings are published faster than Census new-home prices,
    so the newest month has October rates and earnings but no October price.
    ASSUMPTION: the latest observed MSPNHSUS price is carried forward into that month. The
    observed monthly table is left untouched; this row only feeds key numbers and is labelled
    as an estimate.
    """
    this_month = pd.Timestamp(dt.date.today().replace(day=1))
    complete = df[(df.index < this_month) & df["mortgage_rate"].notna()]
    m = complete.index[-1]
    row = df.loc[m].copy()
    priced = df["median_new_price"].dropna()
    row["price_month"] = priced.index[-1]
    row["median_new_price"] = priced.iloc[-1]
    for col in ("avg_hourly_earnings", "wage_proxy_income", "hh_income_nowcast"):
        if pd.isna(row[col]):          # earnings not out yet: use the latest available
            row[col] = df[col].dropna().iloc[-1]
    row["pi_payment"] = monthly_pi(row["median_new_price"], row["mortgage_rate"])
    row["total_payment"] = row["pi_payment"] + row["median_new_price"] * (PROPERTY_TAX_RATE + INSURANCE_RATE) / 12
    for tag, col in [("", "median_hh_income"), ("_wage", "wage_proxy_income"),
                     ("_nowcast", "hh_income_nowcast")]:
        row[f"pi_to_income{tag}"] = row["pi_payment"] / (row[col] / 12)
        row[f"total_to_income{tag}"] = row["total_payment"] / (row[col] / 12)
    row.name = m
    return row


def make_charts(df, sens, hoam_both, nahb_hmi, manual, cur=None) -> None:
    obs = df.dropna(subset=["pi_payment"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df.index, df["mortgage_rate"], color=GOLD, lw=2, label="30-yr mortgage (MORTGAGE30US)")
    ax.plot(df.index, df["ust10"], color=NAVY, lw=2, label="10-yr Treasury (DGS10)")
    _style(ax, "%")
    ax.set_title("Mortgage rate vs. 10-year Treasury, monthly average")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_rate_vs_treasury.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(obs.index, obs["pi_to_income"] * 100, color=NAVY, lw=2, label="P&I / median household income")
    ax.plot(obs.index, obs["total_to_income"] * 100, color=GOLD, lw=2,
            label="P&I + tax + insurance / median household income")
    if cur is not None:
        ax.plot([cur.name], [cur["total_to_income"] * 100], marker="o", ms=8, mfc="white",
                mec=GOLD, mew=2, ls="none",
                label=f"{cur.name:%b-%Y} estimate: {cur.name:%b} rate on {cur.price_month:%b} price")
    ax.axhline(30, color=GREY, lw=1, ls="--")
    ax.text(obs.index[0], 30.4, "30% rule of thumb", fontsize=8, color="#555555")
    _style(ax, "% of income")
    ax.set_title("Payment-to-income, median new home (MSPNHSUS), 20% down")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_payment_to_income.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rec = obs.loc["2022-01-01":]
    ax.plot(rec.index, rec["total_to_income"] * 100, color=NAVY, lw=2,
            label="Median HH income (MEHOINUSA646N), latest year carried forward")
    ax.plot(rec.index, rec["total_to_income_nowcast"] * 100, color=GREY, lw=2, ls="--",
            label="HH income nowcast (grown with AHE after last Census year)")
    ax.plot(rec.index, rec["total_to_income_wage"] * 100, color=GOLD, lw=2,
            label="Wage proxy: AHE x 2080 (CES0500000003)")
    _style(ax, "% of income")
    ax.set_title("Total payment-to-income under three income measures")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_income_measures.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hoam_both.index, hoam_both["ours_total_to_income"] * 100, color=NAVY, lw=2,
            label="Ours: median NEW home, P&I+tax+ins / median HH income")
    ax.plot(hoam_both.index, hoam_both["hoam_share"] * 100, color=GOLD, lw=2,
            label="Atlanta Fed HOAM: annual payment share of income")
    _style(ax, "% of income")
    ax.set_title("Cross-check: our payment-to-income vs. Atlanta Fed HOAM")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_hoam_crosscheck.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sens["rate_pct"], sens["pi_payment"], marker="o", ms=6, lw=2, color=NAVY)
    for _, r in sens.iterrows():
        ax.annotate(f"${r.pi_payment:,.0f}", (r.rate_pct, r.pi_payment),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("30-yr fixed rate (%), scenario")
    _style(ax, "Monthly P&I ($)")
    ax.set_title(f"Payment sensitivity: ${REFERENCE_PRICE:,.0f} home, {DOWN_PAYMENT_PCT:.0%} down")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_payment_sensitivity.png", dpi=160); plt.close(fig)

    if manual is not None and len(manual):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(manual.index, manual["share_cut_prices_pct"], marker="o", ms=6, lw=2, color=NAVY,
                label="Share of builders cutting prices")
        ax.plot(manual.index, manual["share_using_incentives_pct"], marker="o", ms=6, lw=2,
                color=GOLD, label="Share using sales incentives")
        _label_end(ax, manual["share_cut_prices_pct"], f"{manual['share_cut_prices_pct'].iloc[-1]:.0f}%", NAVY)
        _label_end(ax, manual["share_using_incentives_pct"], f"{manual['share_using_incentives_pct'].iloc[-1]:.0f}%", GOLD)
        _style(ax, "% of builders surveyed")
        ax.set_ylim(0, 80)
        ax.set_title("NAHB/Wells Fargo HMI: price cuts and incentives (manual entry from releases)")
        ax.legend(frameon=False, loc="lower left")
        fig.tight_layout(); fig.savefig(OUT_DIR / "fig_nahb_price_cuts.png", dpi=160); plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force a new FRED download")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flags: list[str] = []

    data, log_rows = pull_all(refresh=args.refresh)
    missing = [s for s in SERIES if s not in data]
    if missing:
        raise SystemExit(f"Missing series, cannot continue: {missing}")

    df = build_monthly_table(data)

    # Check the hard-coded reference price against the data (flag, don't fix silently)
    ref_obs = data["MSPNHSUS"].get(pd.Timestamp(REFERENCE_PRICE_MONTH))
    if ref_obs is None or abs(ref_obs - REFERENCE_PRICE) > 0.5:
        flags.append(f"REFERENCE_PRICE {REFERENCE_PRICE:,} != MSPNHSUS {REFERENCE_PRICE_MONTH} "
                     f"({ref_obs}); Census revises new-home prices, update the constant.")
    sens = build_sensitivity(REFERENCE_PRICE)

    # External sources
    hoam_path = fetch_external(HOAM_URL, f"HOAM_US_Affordability_Index_{TODAY}.xlsx") \
        if not list(EXT_DIR.glob("HOAM_US_Affordability_Index_*.xlsx")) \
        else sorted(EXT_DIR.glob("HOAM_US_Affordability_Index_*.xlsx"))[-1]
    hoam = load_hoam(hoam_path)
    hoam_both, hoam_stats, hoam_tp = compare_hoam(df, hoam)
    vc = NAHB_VINTAGE.replace("-", "")
    nahb_path = fetch_external(NAHB_T2_URL.format(vintage=NAHB_VINTAGE, vintage_compact=vc),
                               f"NAHB_t2_national_hmi_history_{vc}.xls")
    nahb_hmi = load_nahb_hmi(nahb_path)
    manual = load_manual_nahb()

    log_rows += [
        dict(source="Federal Reserve Bank of Atlanta", series_id="HOAM US Affordability Index",
             description="Annual Payment Share of Income (national, monthly)", frequency="monthly",
             first_obs=hoam.index.min().date(), last_obs=hoam.index.max().date(),
             last_value=hoam.iloc[-1], vintage="file as downloaded",
             accessed=hoam_path.stem.rsplit("_", 1)[-1], raw_file=str(hoam_path.relative_to(ROOT)),
             url=HOAM_URL, used_as="cross-check"),
        dict(source="NAHB", series_id="NAHB/Wells Fargo HMI Table 2 (national, SA)",
             description="Housing Market Index headline history", frequency="monthly",
             first_obs=nahb_hmi.index.min().date(), last_obs=nahb_hmi.index.max().date(),
             last_value=nahb_hmi.iloc[-1], vintage=f"{NAHB_VINTAGE} release",
             accessed="2026-11-09", raw_file=str(nahb_path.relative_to(ROOT)),
             url=NAHB_T2_URL.format(vintage=NAHB_VINTAGE, vintage_compact=vc), used_as="data"),
        dict(source="NAHB (manual entry)", series_id="HMI price-cut & incentive shares",
             description="Transcribed from monthly HMI releases on eyeonhousing.org; see "
                         "data/manual/nahb_hmi_price_cuts_incentives.csv for per-row URLs",
             frequency="monthly", first_obs=manual.index.min().date(),
             last_obs=manual.index.max().date(), last_value=manual["share_cut_prices_pct"].iloc[-1],
             vintage="as first published (not revised)", accessed="2026-11-09",
             raw_file="data/manual/nahb_hmi_price_cuts_incentives.csv",
             url="https://eyeonhousing.org/", used_as="data (manual)"),
    ] + STATIC_SOURCES
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "source_log.csv", index=False)

    # Consistency flags: NAHB release text vs Table 2 (HMI gets revised)
    for d, row in manual.iterrows():
        t2 = nahb_hmi.get(d)
        if t2 is not None and pd.notna(row.get("hmi_as_published")) and t2 != row["hmi_as_published"]:
            flags.append(f"NAHB HMI {d:%Y-%m}: release text {row['hmi_as_published']:.0f}, "
                         f"Table 2 ({NAHB_VINTAGE}) {t2:.0f} (revision)")

    # Outputs
    df.to_csv(OUT_DIR / "affordability_monthly.csv", index_label="month")
    sens.to_csv(OUT_DIR / "payment_sensitivity.csv", index=False)
    hoam_both.to_csv(OUT_DIR / "hoam_crosscheck.csv", index_label="month")
    hoam_tp.to_csv(OUT_DIR / "hoam_turning_points.csv", index=False)
    pd.Series(hoam_stats).to_csv(OUT_DIR / "hoam_crosscheck_stats.csv", header=["value"])
    cur = current_estimate(df)
    make_charts(df, sens, hoam_both, nahb_hmi, manual, cur)

    # Key numbers, each tagged with source series and observation date
    latest = df.dropna(subset=["pi_payment"]).iloc[-1]
    ld = f"{latest.name:%Y-%m}"
    last_rate_wk = data["MORTGAGE30US"].index[-1]
    pay6, pay718 = monthly_pi(REFERENCE_PRICE, BASE_RATE), monthly_pi(REFERENCE_PRICE, STRESS_RATE)
    hist = df.dropna(subset=["total_to_income"])
    pre = hist.loc["2018-01-01":"2019-12-01", "total_to_income"]
    peak = hist["total_to_income"].idxmax()
    yago = hist.loc[latest.name - pd.DateOffset(years=1)]
    alt_total = latest.pi_payment + latest.tax_payment + latest.median_new_price * INSURANCE_RATE_ALT / 12
    cd = f"{cur.name:%Y-%m}"
    cyago = hist.loc[cur.name - pd.DateOffset(years=1)]
    cur_note = f"est.: {cd} rate, {cur.price_month:%Y-%m} MSPNHSUS price carried forward"
    k = [
        ("current_month", cd, "latest complete MORTGAGE30US month", cd),
        ("current_mortgage_rate_monthly_avg", cur.mortgage_rate, "MORTGAGE30US (monthly mean)", cd),
        ("current_ust10_monthly_avg", cur.ust10, "DGS10 (monthly mean)", cd),
        ("current_mortgage_spread", cur.mortgage_spread, "MORTGAGE30US - DGS10", cd),
        ("current_price_used", cur.median_new_price, "MSPNHSUS (latest published)", f"{cur.price_month:%Y-%m}"),
        ("current_pi_payment", cur.pi_payment, cur_note, cd),
        ("current_total_payment", cur.total_payment, cur_note, cd),
        ("current_hh_income_nowcast", cur.hh_income_nowcast, "calc: MEHOINUSA646N x CES0500000003 growth", cd),
        ("current_wage_proxy_income", cur.wage_proxy_income, "CES0500000003 x 2080", cd),
        ("current_total_to_income", cur.total_to_income, cur_note, cd),
        ("current_total_to_income_nowcast", cur.total_to_income_nowcast, cur_note, cd),
        ("current_total_to_income_wage", cur.total_to_income_wage, cur_note, cd),
        ("current_total_to_income_1y_ago", cyago.total_to_income, "calc (observed)", f"{cyago.name:%Y-%m}"),
        ("current_mortgage_rate_1y_ago", cyago.mortgage_rate, "MORTGAGE30US (monthly mean)", f"{cyago.name:%Y-%m}"),
        ("latest_month_with_price", ld, "MSPNHSUS", ld),
        ("mortgage_rate_monthly_avg", latest.mortgage_rate, "MORTGAGE30US (monthly mean)", ld),
        ("mortgage_rate_latest_week", data["MORTGAGE30US"].iloc[-1], "MORTGAGE30US", f"{last_rate_wk:%Y-%m-%d}"),
        ("ust10_monthly_avg", latest.ust10, "DGS10 (monthly mean)", ld),
        ("mortgage_spread", latest.mortgage_spread, "MORTGAGE30US - DGS10", ld),
        ("median_new_price", latest.median_new_price, "MSPNHSUS", ld),
        ("median_new_price_quarterly", latest.median_new_price_q_MSPUS, "MSPUS (quarter containing month)", ld),
        ("pi_payment", latest.pi_payment, "calc: MSPNHSUS, MORTGAGE30US, 20% down", ld),
        ("total_payment", latest.total_payment, "calc: + tax 0.90% (ATTOM) + ins 0.50% (placeholder)", ld),
        ("total_payment_alt_insurance", alt_total, "calc: insurance at Bankrate-implied 0.62%", ld),
        ("median_hh_income_used", latest.median_hh_income, f"MEHOINUSA646N ({int(latest.hh_income_year)})", ld),
        ("hh_income_nowcast", latest.hh_income_nowcast, "calc: MEHOINUSA646N x CES0500000003 growth", ld),
        ("wage_proxy_income", latest.wage_proxy_income, "CES0500000003 x 2080", ld),
        ("pi_to_income", latest.pi_to_income, "calc", ld),
        ("total_to_income", latest.total_to_income, "calc", ld),
        ("total_to_income_nowcast", latest.total_to_income_nowcast, "calc", ld),
        ("total_to_income_wage", latest.total_to_income_wage, "calc", ld),
        ("total_to_income_1y_ago", yago.total_to_income, "calc", f"{yago.name:%Y-%m}"),
        ("total_to_income_2018_19_avg", pre.mean(), "calc", "2018-01..2019-12"),
        ("total_to_income_peak", hist["total_to_income"].max(), "calc", f"{peak:%Y-%m}"),
        ("months_supply", latest.months_supply, "MSACSR", ld),
        ("saving_rate", latest.saving_rate, "PSAVERT", ld),
        ("ref_price", REFERENCE_PRICE, "MSPNHSUS", REFERENCE_PRICE_MONTH[:7]),
        ("pi_at_6.00", pay6, "scenario calc on ref_price", "assumption"),
        ("pi_at_7.18", pay718, "scenario calc on ref_price", "assumption"),
        ("pi_gap_7.18_vs_6.00", pay718 - pay6, "scenario calc", "assumption"),
        ("pi_gap_pct", pay718 / pay6 - 1, "scenario calc", "assumption"),
        ("price_supported_at_7.18_with_6pct_payment", price_supported(pay6, STRESS_RATE), "scenario calc", "assumption"),
        # Cost to a builder of a permanent 7.18%->6.00% buydown = PV (at 7.18%) of the
        # monthly payment gap. ASSUMPTION: 84-month expected loan life (refi/sale); the
        # 360-month figure is the upper bound and equals the loan-size cut in the line above.
        ("buydown_cost_pv_84m", pv_gap(pay718 - pay6, STRESS_RATE, 84), "scenario calc", "assumption"),
        ("buydown_cost_pv_84m_pct_price", pv_gap(pay718 - pay6, STRESS_RATE, 84) / REFERENCE_PRICE, "scenario calc", "assumption"),
        ("buydown_cost_pv_360m", pv_gap(pay718 - pay6, STRESS_RATE, 360), "scenario calc", "assumption"),
        ("total_to_income_at_7.18_sep_price", total_at(REFERENCE_PRICE, STRESS_RATE) / (latest.median_hh_income / 12), "scenario calc, MEHOINUSA646N 2025", "assumption"),
        ("total_to_income_at_7.18_sep_price_nowcast", total_at(REFERENCE_PRICE, STRESS_RATE) / (cur.hh_income_nowcast / 12), f"scenario calc, nowcast income {cd}", "assumption"),
        ("hoam_latest", hoam.iloc[-1], "Atlanta Fed HOAM", f"{hoam.index[-1]:%Y-%m}"),
        ("nahb_hmi_latest", nahb_hmi.iloc[-1], "NAHB HMI Table 2", f"{nahb_hmi.index[-1]:%Y-%m}"),
        ("nahb_share_cut_prices", manual["share_cut_prices_pct"].iloc[-1], "NAHB HMI release (manual)", f"{manual.index[-1]:%Y-%m}"),
        ("nahb_avg_price_cut", manual["avg_price_cut_pct"].iloc[-1], "NAHB HMI release (manual)", f"{manual.index[-1]:%Y-%m}"),
        ("nahb_share_incentives", manual["share_using_incentives_pct"].iloc[-1], "NAHB HMI release (manual)", f"{manual.index[-1]:%Y-%m}"),
    ] + [(f"hoam_{key}", v, "calc", "") for key, v in hoam_stats.items()]
    pd.DataFrame(k, columns=["metric", "value", "source", "obs_date"]).to_csv(
        OUT_DIR / "key_numbers.csv", index=False)

    print(f"\nCurrent month {cd} (estimate, {cur.price_month:%b} price): rate {cur.mortgage_rate:.2f}%, "
          f"10y {cur.ust10:.2f}%, spread {cur.mortgage_spread:.2f}pp, total/income {cur.total_to_income:.1%} "
          f"(nowcast {cur.total_to_income_nowcast:.1%}, wage {cur.total_to_income_wage:.1%}); "
          f"a year earlier {cyago.total_to_income:.1%}")
    print(f"\nLatest month with an observed price: {ld}")
    print(f"  Mortgage rate (month avg): {latest.mortgage_rate:.2f}%   latest week "
          f"{last_rate_wk:%Y-%m-%d}: {data['MORTGAGE30US'].iloc[-1]:.2f}%")
    print(f"  Median new price:          ${latest.median_new_price:,.0f}")
    print(f"  P&I / P&I+tax+ins:         ${latest.pi_payment:,.0f} / ${latest.total_payment:,.0f}")
    print(f"  Total / income  (HH {int(latest.hh_income_year)}): {latest.total_to_income:.1%}   "
          f"nowcast: {latest.total_to_income_nowcast:.1%}   wage proxy: {latest.total_to_income_wage:.1%}")
    print(f"  6.00% vs 7.18% P&I on ${REFERENCE_PRICE:,}: ${pay6:,.0f} vs ${pay718:,.0f} "
          f"(+${pay718 - pay6:,.0f}, {pay718 / pay6 - 1:+.1%})")
    print("\nHOAM cross-check:", {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in hoam_stats.items()})
    print("\nSensitivity table (scenario):\n", sens.to_string(index=False))
    if flags:
        print("\nFLAGS:")
        for f in flags:
            print("  -", f)
    (OUT_DIR / "flags.txt").write_text("\n".join(flags) + "\n")
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
