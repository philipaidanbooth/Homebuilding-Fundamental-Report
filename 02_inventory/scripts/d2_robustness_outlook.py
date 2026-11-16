"""Driver 2 add-ons (2026-11-10): robustness checks for the price-cut test and a finished-inventory
scenario calculation. Reads outputs/d2_*.csv, data/raw/NHSDPC_*.csv and Driver 1's MORTGAGE30US raw file
(read-only). Writes outputs/d2_robustness.csv, outputs/d2_inventory_outlook.csv, outputs/d2_outlook_key.csv.
Run from the 02_inventory folder: python3 scripts/d2_robustness_outlook.py
"""
import glob, math, os
import numpy as np, pandas as pd

OUT = "outputs"

# ---- t-distribution two-sided p (no scipy needed) ----
def _betacf(a, b, x):
    TINY = 1e-300; qab = a + b; qap = a + 1; qam = a - 1; c = 1.0; d = 1 - qab * x / qap
    d = TINY if abs(d) < TINY else d; d = 1 / d; h = d
    for m in range(1, 400):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = TINY if abs(d) < TINY else d; c = 1 + aa / c; c = TINY if abs(c) < TINY else c; d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = TINY if abs(d) < TINY else d; c = 1 + aa / c; c = TINY if abs(c) < TINY else c; d = 1 / d
        de = d * c; h *= de
        if abs(de - 1) < 1e-14: break
    return h
def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b
def p_two_sided(t, df): return _ibeta(df / 2, 0.5, df / (df + t * t))

def ols_nw(y, X):
    """OLS with Newey-West (Bartlett) errors; lag rule floor(4*(n/100)^(2/9)), same as d2_build_data.py."""
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1)); y = y[ok]; X = X[ok]; n, k = X.shape
    Z = np.column_stack([np.ones(n), X]); XtXi = np.linalg.inv(Z.T @ Z); b = XtXi @ Z.T @ y; e = y - Z @ b
    L = int(math.floor(4 * (n / 100) ** (2 / 9))); u = Z * e[:, None]; S = u.T @ u
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l]; S = S + (1 - l / (L + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi)); t = b / se
    p = [p_two_sided(abs(x), n - k - 1) for x in t]
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return dict(n=n, b=b, se=se, t=t, p=p, r2=r2, L=L)

# ---- data ----
cuts = pd.read_csv(os.path.join(OUT, "d2_cuts_vs_completed.csv"), parse_dates=["date"])
stg = pd.read_csv(os.path.join(OUT, "d2_inventory_by_stage.csv"), parse_dates=["date"]).set_index("date")
mfile = sorted(glob.glob(os.path.join("..", "01_affordability", "data", "raw", "MORTGAGE30US_*.csv")))[-1]
mr = pd.read_csv(mfile, parse_dates=["observation_date"]).set_index("observation_date")["MORTGAGE30US"]
mr_m = mr.resample("MS").mean()                              # monthly average of the weekly PMMS rate
cms = stg["completed_months_supply"]
d = cuts[["date", "share_cut_pct"]].copy()
d["cms"] = d.date.map(cms)
for k in (1, 2, 3):                                          # lag by CALENDAR month
    d[f"cms_l{k}"] = d.date.map(lambda x, k=k: cms.get(x - pd.DateOffset(months=k), np.nan))
d["rate"] = d.date.map(mr_m)
y = d.share_cut_pct.to_numpy(float)

rows = []
def add(label, cols, note):
    r = ols_nw(y, d[cols].to_numpy(float))
    row = dict(model=label, n=r["n"], slope=r["b"][1], nw_se=r["se"][1], t=r["t"][1], p=r["p"][1], r2=r["r2"],
               rate_coef=np.nan, rate_p=np.nan, nw_lag=r["L"], note=note)
    if len(cols) == 2: row.update(rate_coef=r["b"][2], rate_p=r["p"][2])
    rows.append(row); return r
add("(b) % cutting ~ completed months' supply (baseline)", ["cms"], "same month; reproduces model (b)")
for k in (1, 2, 3):
    add(f"(f{k}) % cutting ~ completed months' supply, {k} month{'s' if k > 1 else ''} earlier", [f"cms_l{k}"],
        "supply lagged by calendar month; Oct-2026 survey month enters because lagged supply exists")
add("(g) % cutting ~ completed months' supply + 30-year mortgage rate", ["cms", "rate"],
    "rate = monthly average of weekly Freddie Mac PMMS; slope shown is on completed months' supply")
add("(h) % cutting ~ 30-year mortgage rate only", ["rate"], "slope shown is on the mortgage rate")
rows[-1].update(rate_coef=rows[-1]["slope"], rate_p=rows[-1]["p"])
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "d2_robustness.csv"), index=False, float_format="%.10g")
print(R[["model", "n", "slope", "nw_se", "t", "p", "r2", "rate_coef", "rate_p"]].to_string())
print("corr(cms, rate) in sample:", d[["cms", "rate"]].dropna().corr().iloc[0, 1])

# ---- finished-inventory scenario calculation ----
sfile = sorted(glob.glob(os.path.join("data", "raw", "NHSDPC_*.csv")))[-1]
sold = pd.read_csv(sfile, parse_dates=["observation_date"]).set_index("observation_date")["NHSDPC"]  # thousands, monthly, NSA
end = stg.index.max(); beg = end - pd.DateOffset(months=12)
win = sold.loc[beg + pd.DateOffset(months=1): end]; assert len(win) == 12
S12 = float(win.sum())                                       # completed homes sold, trailing 12 months
c_end = float(stg.completed_k[end]); c_beg = float(stg.completed_k[beg])
I12 = S12 + (c_end - c_beg)                                  # implied completions entering unsold inventory
uc_end = float(stg.under_construction_k[end]); uc_beg = float(stg.under_construction_k[beg])
pipe = uc_end / uc_beg
sales_yoy = float(stg.sales_saar_k[end] / stg.sales_saar_k[beg] - 1)
avg1819 = float(stg.completed_k["2018-01-01":"2019-12-01"].mean())
sc = [("Last 12 months (actual)", c_beg, I12, S12, "Observed and implied"),
      ("Base: last year's pace repeats", c_end, I12, S12, "Scenario"),
      ("Base: completions scaled to the smaller pipeline", c_end, I12 * pipe, S12, "Scenario"),
      ("Stress: finished-home sales fall as much as total sales did", c_end, I12, S12 * (1 + sales_yoy), "Scenario")]
O = pd.DataFrame([dict(case=a, beginning_k=b, completions_in_k=i, completed_sold_k=s, ending_k=b + i - s,
                       ending_vs_2018_19_avg_pct=100 * ((b + i - s) / avg1819 - 1), kind=k) for a, b, i, s, k in sc])
O.to_csv(os.path.join(OUT, "d2_inventory_outlook.csv"), index=False, float_format="%.6g")
K = pd.DataFrame([("window_start", (beg + pd.DateOffset(months=1)).strftime("%Y-%m-01"), "first month of trailing 12-month window"),
    ("window_end", end.strftime("%Y-%m-01"), "last Census month"),
    ("completed_sold_12m_k", S12, "NHSDPC sum, NSA, thousands"),
    ("implied_completions_in_12m_k", I12, "completed sold + change in completed for sale"),
    ("under_construction_now_k", uc_end, "NHFSEPUCS"), ("under_construction_year_ago_k", uc_beg, "NHFSEPUCS"),
    ("pipeline_ratio", pipe, "under construction now / a year ago"),
    ("pipeline_change_pct", 100 * (pipe - 1), ""),
    ("total_sales_yoy_pct", 100 * sales_yoy, "HSN1F latest month vs a year earlier"),
    ("breakeven_sales_drop_pct", 100 * (c_end - c_beg) / S12, "change in finished-home sales that holds the stock flat with completions unchanged"),
    ("completed_sold_share_of_sales_pct", 100 * S12 / float(pd.read_csv(sorted(glob.glob('data/raw/HSN1F_*.csv'))[-1], parse_dates=['observation_date']).set_index('observation_date')['HSN1F'].loc[win.index].mean()), "completed homes sold / average annual sales rate over the window")],
    columns=["name", "value", "note"])
K.to_csv(os.path.join(OUT, "d2_outlook_key.csv"), index=False)
print(O.to_string()); print(K.to_string())
