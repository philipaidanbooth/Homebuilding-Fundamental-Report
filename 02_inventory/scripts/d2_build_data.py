#!/usr/bin/env python3
# d2_build_data.py  (Driver 2: Inventory, data layer)
# Reads  (never modifies): data/raw/{NHFSEPCS,NHFSEPUCS,NHFSEPNTS,HNFSEPUSSA,MSACSR,HSN1F,MSPNHSUS,
#         ACTLISCOUUS,PERMIT1,HOUST1F}_2026-11-10.csv, data/raw/external/nmdb_*.zip,
#         data/manual/nahb_price_cuts_backfill_STAGING.csv, data/manual/builder_completed_specs_STAGING.csv,
#         ../01_affordability/data/manual/nahb_hmi_price_cuts_incentives.csv,
#         outputs/builder_specs_per_closing_ts.csv, outputs/test_*.csv (comparison only)
# Writes (new files only, d2_ prefix) in outputs/: d2_inventory_by_stage, d2_cuts_vs_completed,
#         d2_test_results, d2_starts_permits, d2_lockin, d2_builder_specs, d2_key_numbers,
#         d2_source_log (.csv) and d2_data_flags.txt.  Refuses to overwrite an existing d2_ file
#         unless run with --force-d2 (which only ever touches d2_ files).
# Run from anywhere: python3 02_inventory/scripts/d2_build_data.py
import csv, io, math, os, sys, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, EXT, MAN, OUT = [os.path.join(ROOT, p) for p in ("data/raw", "data/raw/external", "data/manual", "outputs")]
D1_NAHB = os.path.join(ROOT, "../01_affordability/data/manual/nahb_hmi_price_cuts_incentives.csv")
ACCESS = "2026-11-10"
FORCE = "--force-d2" in sys.argv
flags = []

def outpath(name):
    p = os.path.join(OUT, name)
    if os.path.exists(p) and not FORCE:
        sys.exit(f"{p} exists; not overwriting (use --force-d2 to replace d2_ files)")
    return p

# ---------- FRED ----------
SRC = []  # source log rows
def fred(sid, desc, freq, used_as):
    f = os.path.join(RAW, f"{sid}_{ACCESS}.csv")
    d = pd.read_csv(f, na_values=[".", ""])
    d.columns = ["date", sid]
    d["date"] = pd.to_datetime(d["date"]); d = d.dropna().sort_values("date")
    SRC.append(dict(source="FRED", series_id=sid, url=f"https://fred.stlouisfed.org/series/{sid}",
        description=desc, frequency=freq, first_obs=d.date.min().date(), last_obs=d.date.max().date(),
        last_value=d[sid].iloc[-1], vintage="FRED current vintage as of access date",
        access_date=ACCESS, raw_file=os.path.relpath(f, ROOT), used_as=used_as))
    return d.set_index("date")[sid]

S = {}
for sid, desc, freq, use in [
    ("NHFSEPCS", "New houses for sale, completed (thous., SA)", "monthly", "d2_inventory_by_stage; d2_cuts_vs_completed; d2_test_results"),
    ("NHFSEPUCS", "New houses for sale, under construction (thous., SA)", "monthly", "d2_inventory_by_stage"),
    ("NHFSEPNTS", "New houses for sale, not started (thous., SA)", "monthly", "d2_inventory_by_stage"),
    ("HNFSEPUSSA", "New houses for sale, total (thous., SA)", "monthly", "d2_inventory_by_stage total_k; stage-sum check"),
    ("MSACSR", "Monthly supply of new houses (months, SA)", "monthly", "d2_inventory_by_stage; d2_test_results"),
    ("HSN1F", "New one-family houses sold (thous., SAAR)", "monthly", "completed months' supply; d2_key_numbers"),
    ("MSPNHSUS", "Median sales price, new houses sold ($, NSA)", "monthly", "d2_test_results proxy models"),
    ("ACTLISCOUUS", "Realtor.com active listing count, U.S. (NSA)", "monthly", "d2_lockin"),
    ("PERMIT1", "New private housing units authorized by building permits, 1-unit (thous., SAAR)", "monthly", "d2_starts_permits"),
    ("HOUST1F", "Housing starts, new privately owned 1-unit structures (thous., SAAR)", "monthly", "d2_starts_permits")]:
    S[sid] = fred(sid, desc, freq, use)
m = pd.concat(S.values(), axis=1).sort_index()
m["stage_sum"] = m.NHFSEPCS + m.NHFSEPUCS + m.NHFSEPNTS
m["cms"] = m.NHFSEPCS / (m.HSN1F / 12)   # completed months' supply (Calc)
def last(s):
    s = s.dropna(); return s.index[-1], s.iloc[-1]
def iso(d): return pd.Timestamp(d).strftime("%Y-%m-%d")
def r(x, n=4): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)

# ---------- FHFA NMDB ----------
zf = os.path.join(EXT, f"nmdb_outstanding_national_quarterly_{ACCESS}.zip")
z = zipfile.ZipFile(zf)
nm = pd.read_csv(z.open(z.namelist()[0]))
ids = ["PCT_INTRATE_LT_3", "PCT_INTRATE_3_4", "PCT_INTRATE_4_5", "PCT_INTRATE_5_6", "PCT_INTRATE_GE_6"]
nm = nm[(nm.GEOID.astype(str) == "USA") & (nm.MARKET == "All Mortgages") & nm.SERIESID.isin(ids)].copy()
nm["date"] = pd.to_datetime(dict(year=nm.YEAR, month=nm.MONTH, day=1))
dups = nm.duplicated(["date", "SERIESID"]).sum()
if dups: flags.append(f"NMDB: {dups} duplicate (date, series) rows for USA / All Mortgages; pivot keeps the mean.")
nmw = nm.pivot_table(index="date", columns="SERIESID", values="VALUE1", aggfunc="mean").sort_index()
nmw["below5"] = nmw.PCT_INTRATE_LT_3 + nmw.PCT_INTRATE_3_4 + nmw.PCT_INTRATE_4_5
nmw["bsum"] = nmw.below5 + nmw.PCT_INTRATE_5_6 + nmw.PCT_INTRATE_GE_6
if (abs(nmw.bsum - 100) > 0.5).any():
    flags.append(f"NMDB rate buckets do not sum to 100 in some quarters (max gap {abs(nmw.bsum-100).max():.2f} pts).")
SRC.append(dict(source="FHFA", series_id="NMDB outstanding, national, All Mortgages: PCT_INTRATE_LT_3 + 3_4 + 4_5",
    url="https://www.fhfa.gov/document/d/nmdb/nmdb-outstanding-mortgage-statistics-national-census-areas-quarterly.zip",
    description="Share of outstanding first-lien mortgages with rate below 5% (%), GEOID USA, MARKET All Mortgages; dated at quarter-end month",
    frequency="quarterly", first_obs=nmw.index.min().date(), last_obs=nmw.index.max().date(),
    last_value=r(nmw.below5.iloc[-1], 2), vintage="FHFA zip file as downloaded on access date (not revision-tracked)",
    access_date=ACCESS, raw_file=os.path.relpath(zf, ROOT), used_as="d2_lockin pct_mortgages_below_5; d2_key_numbers"))

# ---------- NAHB (Driver 1 wins; backfill fills missing only) ----------
def read_nahb(p, origin):
    with open(p, encoding="utf-8") as fh:
        lines = [l for l in fh if not l.lstrip().startswith("#")]
    d = pd.read_csv(io.StringIO("".join(lines)))
    return pd.DataFrame({"date": pd.to_datetime(d["month"]),
        "share_cut": pd.to_numeric(d["share_cut_prices_pct"], errors="coerce"),
        "avg_cut": pd.to_numeric(d["avg_price_cut_pct"], errors="coerce"),
        "share_incent": pd.to_numeric(d["share_using_incentives_pct"], errors="coerce"), "origin": origin})
bf_path = os.path.join(MAN, "nahb_price_cuts_backfill_STAGING.csv")
d1 = read_nahb(D1_NAHB, "Driver 1 manual file"); bf = read_nahb(bf_path, os.path.basename(bf_path))
rows = []
allr = pd.concat([d1, bf]); allr = allr[allr.share_cut.notna() | allr.share_incent.notna()]
for dt, g in allr.groupby("date"):
    g = g.sort_values("origin", key=lambda s: s != "Driver 1 manual file", kind="stable")
    def pick(c):
        h = g[g[c].notna()]
        return (h[c].iloc[0], h.origin.iloc[0]) if len(h) else (np.nan, None)
    sc, so = pick("share_cut"); ac, _ = pick("avg_cut"); si, _ = pick("share_incent")
    rows.append(dict(date=dt, share_cut=sc, avg_cut=ac, share_incent=si, cut_origin=so,
                     files=" + ".join(dict.fromkeys(g.origin))))
nahb = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
j = d1.merge(bf, on="date")
j = j[j.share_cut_x.notna() & j.share_cut_y.notna() & (j.share_cut_x != j.share_cut_y)]
if len(j): flags.append("NAHB price-cut share differs between Driver 1 file and backfill for " + ", ".join(j.date.dt.strftime("%Y-%m")) + ". Driver 1 value used.")
else: flags.append("NAHB: Driver 1 file and backfill agree on every month where both have a price-cut share (Driver 1 value used anyway).")
cutm = nahb[nahb.share_cut.notna()]
gaps = [d for d in pd.date_range(cutm.date.min(), cutm.date.max(), freq="MS") if d not in set(cutm.date)]
if gaps: flags.append("NAHB price-cut share missing for months: " + ", ".join(g.strftime("%Y-%m") for g in gaps) + " (no row in either file; not interpolated).")
n_bf = (cutm.cut_origin != "Driver 1 manual file").sum()
flags.append(f"NAHB backfill file is a STAGING file (AI-transcribed, not spot-checked against verbatim_quote). {n_bf} of {len(cutm)} price-cut months come from it; the rest from the Driver 1 hand-entered file. NAHB values are as first published (e.g. 2022-09 kept at 24 although NAHB later restated 26).")
SRC.append(dict(source="NAHB", series_id="HMI special questions: % builders cutting prices, avg cut, % using incentives",
    url="https://eyeonhousing.org (per-row source_url in the manual files)",
    description="Hand-transcribed from NAHB/Wells Fargo HMI monthly posts; Driver 1 file wins, STAGING backfill fills missing months only",
    frequency="monthly", first_obs=cutm.date.min().date(), last_obs=cutm.date.max().date(),
    last_value=cutm.share_cut.iloc[-1], vintage="as first published (not revised)", access_date="2026-11-10 (per-row 'accessed' in source files)",
    raw_file=os.path.relpath(D1_NAHB, ROOT) + " ; " + os.path.relpath(bf_path, ROOT),
    used_as="d2_cuts_vs_completed; d2_test_results; d2_key_numbers"))

# ---------- Tests (replicates driver2_charts.Rmd) ----------
def tsf(t, df):
    x = df / (df + t * t)
    try:
        from scipy.special import betainc; return float(betainc(df / 2, 0.5, x))
    except Exception:
        pass
    a, b = df / 2, 0.5
    def cf(a, b, x):
        tiny = 1e-300; qab = a + b; qap = a + 1; qam = a - 1
        c = 1.0; d = 1 - qab * x / qap; d = tiny if abs(d) < tiny else d; d = 1 / d; h = d
        for mm in range(1, 500):
            m2 = 2 * mm; aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d; c = 1 + aa / c; c = tiny if abs(c) < tiny else c; d = 1 / d; h *= d * c
            aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d; c = 1 + aa / c; c = tiny if abs(c) < tiny else c; d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-14: break
        return h
    if x <= 0: return 0.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * cf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * cf(b, a, 1 - x) / b

def ols_nw(y, x, L=None):
    ok = ~(np.isnan(y) | np.isnan(x)); y = y[ok]; x = x[ok]; n = len(y)
    X = np.column_stack([np.ones(n), x]); beta = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ beta
    if L is None: L = int(math.floor(4 * (n / 100) ** (2 / 9)))
    XtXi = np.linalg.inv(X.T @ X); u = X * e[:, None]; Sm = u.T @ u
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l]; Sm = Sm + (1 - l / (L + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ Sm @ XtXi)); t = beta[1] / se[1]
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return dict(n=n, intercept=beta[0], slope=beta[1], nw_se=se[1], t=t, p=tsf(abs(t), n - 2), r2=r2, L=L)

# td: NAHB rows (all months in the NAHB table, incl. rows lacking share_cut) left-joined to Census; lags are by ROW as in the Rmd
td = nahb[["date", "share_cut"]].merge(m.reset_index(), on="date", how="left").sort_values("date").reset_index(drop=True)
td["ms_l1"] = td.MSACSR.shift(1)
gapok = (td.date.diff().dt.days <= 31)
for a, b in [("d_cut", "share_cut"), ("d_ms", "MSACSR"), ("d_cms", "cms")]:
    td[a] = (td[b] - td[b].shift(1)).where(gapok)
V = lambda df, c: df[c].to_numpy(float)
res = []
for lab, y, x in [("(a) % cutting ~ months' supply", "share_cut", "MSACSR"),
                  ("(b) % cutting ~ completed months' supply", "share_cut", "cms"),
                  ("(c) % cutting ~ months' supply, lagged 1m", "share_cut", "ms_l1"),
                  ("(d) change in % cutting ~ change in months' supply", "d_cut", "d_ms"),
                  ("(e) change in % cutting ~ change in completed months' supply", "d_cut", "d_cms")]:
    res.append(dict(model=lab, **ols_nw(V(td, y), V(td, x))))
px = m[["MSPNHSUS", "cms", "MSACSR"]].dropna(subset=["MSPNHSUS"]).copy()
px["p3"] = (px.MSPNHSUS + px.MSPNHSUS.shift(1) + px.MSPNHSUS.shift(2)) / 3
px["yoy"] = 100 * (px.p3 / px.p3.shift(12) - 1); px["fwd6"] = px.yoy.shift(-6)
px = px[px.index >= "1999-01-01"]; px12 = px[px.index >= "2012-01-01"]
for lab, df, x in [("Median price y/y, 6m ahead ~ completed months' supply", px, "cms"),
                   ("Median price y/y, 6m ahead ~ months' supply", px, "MSACSR"),
                   ("Same, completed months' supply, 2012+ only (excludes the 2007-09 bust)", px12, "cms")]:
    res.append(dict(model=lab, **ols_nw(V(df, "fwd6"), V(df, x), L=18)))
R = pd.DataFrame(res)
tr = R[["model", "n", "intercept", "slope", "nw_se", "t", "p", "r2"]].copy()
tr.to_csv(outpath("d2_test_results.csv"), index=False, float_format="%.10g")
# reconcile with existing R outputs
recon = []
for fn, sub in [("test_price_cuts_vs_supply.csv", R.iloc[:5]), ("test_proxy_price_vs_supply.csv", R.iloc[5:])]:
    ex = pd.read_csv(os.path.join(OUT, fn))
    for (_, a), (_, b) in zip(sub.iterrows(), ex.iterrows()):
        dif = max(abs(a[c] - b[c]) / max(abs(b[c]), 1e-12) for c in ["slope", "nw_se", "t", "p", "r2"])
        recon.append((a.model, int(a.n), int(b.n), dif))
print("RECONCILIATION (max relative diff vs existing R output; n py/R):")
for x in recon: print(f"  {x[0][:60]:60s} n={x[1]}/{x[2]} maxreldiff={x[3]:.2e}")
bad = [x for x in recon if x[3] > 1e-6 or x[1] != x[2]]
if bad: flags.append("MISMATCH between Python and R test results: " + "; ".join(x[0] for x in bad))
else: flags.append("Test results recomputed in Python match the R outputs (test_price_cuts_vs_supply.csv, test_proxy_price_vs_supply.csv) to better than 1e-6 relative; the R outputs do not store the intercept (added here).")
flags.append("Models (c)-(e) lag/difference by NAHB ROW as the Rmd does; across the missing 2022-10 and 2023-01 months, (c) lags 2 months and (d)/(e) are blanked by the <=31-day rule. n: (a),(b)=49, (c)=49 (includes Oct-2026 against Sep-2026 supply), (d),(e)=46.")

# ---------- 1. inventory by stage ----------
mi = m.loc["1999-01-01":].copy()
mi = mi[mi[["NHFSEPCS", "NHFSEPUCS", "NHFSEPNTS", "HNFSEPUSSA", "MSACSR", "HSN1F"]].notna().any(axis=1)]
inv = pd.DataFrame({"date": mi.index.strftime("%Y-%m-01"), "completed_k": mi.NHFSEPCS.values,
    "under_construction_k": mi.NHFSEPUCS.values, "not_started_k": mi.NHFSEPNTS.values, "total_k": mi.HNFSEPUSSA.values,
    "months_supply": mi.MSACSR.values, "sales_saar_k": mi.HSN1F.values, "completed_months_supply": np.round(mi.cms.values, 4)})
inv.to_csv(outpath("d2_inventory_by_stage.csv"), index=False)
chk = m[(m.stage_sum.notna()) & m.HNFSEPUSSA.notna() & ((m.stage_sum - m.HNFSEPUSSA).abs() > 2)].loc["1999-01-01":]
flags.append(f"Stage sum vs HNFSEPUSSA total: {len(chk)} months since 1999 differ by more than 2k" + (": " + ", ".join(chk.index.strftime('%Y-%m')) if len(chk) else " (rounding only)") + ". total_k is the Census total series, not the stage sum.")

# ---------- 2. cuts vs completed ----------
fitm = R.iloc[1]
b0, b1 = fitm.intercept, fitm.slope
cv = nahb[nahb.share_cut.notna()].merge(m[["cms", "MSACSR"]].reset_index(), on="date", how="left")
cv["has_census"] = cv.cms.notna().astype(int)
cv["fitted"] = np.where(cv.has_census == 1, b0 + b1 * cv.cms, np.nan)
cv["resid"] = cv.share_cut - cv.fitted
latest_d, latest_cms = last(m.cms)
fit_latest = b0 + b1 * latest_cms
no_census = cv.has_census == 0
note = ("NAHB survey month has no Census print yet (Census new-home inventory lags one month: latest Census month is "
        f"{latest_d.strftime('%b-%Y')}). Model fitted value is evaluated at {latest_d.strftime('%b-%Y')} completed months' supply "
        f"({latest_cms:.2f}); gap compares the {cv.date.iloc[-1].strftime('%b-%Y')} survey to a one-month-older inventory baseline, so it is a timing-mismatched comparison.")
cv_out = pd.DataFrame({"date": cv.date.dt.strftime("%Y-%m-01"), "share_cut_pct": cv.share_cut,
    "completed_months_supply": cv.cms.round(4), "months_supply": cv.MSACSR, "fitted_pct": cv.fitted.round(3),
    "residual_pts": cv.resid.round(3), "has_census": cv.has_census, "origin": cv.cut_origin,
    "fitted_at_latest_inventory": np.where(no_census, round(fit_latest, 3), np.nan),
    "gap_vs_latest_pts": np.where(no_census, (cv.share_cut - fit_latest).round(3), np.nan),
    "note": np.where(no_census, note, "")})
cv_out.to_csv(outpath("d2_cuts_vs_completed.csv"), index=False)
oct_gap = float(cv_out.gap_vs_latest_pts.dropna().iloc[-1]) if no_census.any() else None
if any(no_census & (cv.date != cv.date.max())): flags.append("Some NAHB months other than the latest lack Census data.")

# ---------- 4. starts / permits ----------
sp = pd.DataFrame({"permits_1f_k": m.PERMIT1, "starts_1f_k": m.HOUST1F}).dropna(how="all")
for c in ["permits_1f_k", "starts_1f_k"]:
    sp[c.replace("_k", "") + "_yoy_pct"] = (100 * (sp[c] / sp[c].shift(12) - 1)).round(3)
sp = sp.loc["2018-01-01":]; sp.insert(0, "date", sp.index.strftime("%Y-%m-01"))
sp = sp.rename(columns={"permits_1f_yoy_pct": "permits_yoy_pct", "starts_1f_yoy_pct": "starts_yoy_pct"})
sp.to_csv(outpath("d2_starts_permits.csv"), index=False)

# ---------- 5. lock-in ----------
a = S["ACTLISCOUUS"]
l12 = a.rolling(12).mean()  # needs 12 consecutive monthly obs
idx = a.loc["2016-07-01":].index
same19 = [a.get(pd.Timestamp(year=2019, month=d.month, day=1), np.nan) for d in idx]
lk = pd.DataFrame({"date": idx.strftime("%Y-%m-01"), "active_listings": a.loc[idx].values,
    "listings_12m_avg": l12.loc[idx].round(1).values, "listings_same_month_2019": same19,
    "pct_mortgages_below_5": [nmw.below5.get(d, np.nan) for d in idx]})
lk["pct_mortgages_below_5"] = lk.pct_mortgages_below_5.round(2)
lk.to_csv(outpath("d2_lockin.csv"), index=False)
missing_q = [d for d in nmw.index if d >= pd.Timestamp("2016-07-01") and d not in set(idx)]
if missing_q: flags.append("NMDB quarter dates with no matching listings month (not in d2_lockin): " + ", ".join(d.strftime('%Y-%m') for d in missing_q))
flags.append(f"FHFA NMDB last observation is {nmw.index.max().strftime('%Y-%m')} (quarter end; share {nmw.below5.iloc[-1]:.1f}%), about 4 months behind Census (Sep-2026) and 4 behind Realtor.com (Oct-2026). Lock-in share lags the other series; Q3-2026 not yet published.")
flags.append("Realtor.com ACTLISCOUUS is NOT seasonally adjusted: compare same calendar month (listings_same_month_2019, yoy), not adjacent months. listings_12m_avg is a trailing mean and is blank until 12 months exist. For 2019 rows the 2019 column equals the row's own value.")

# ---------- 6. builder specs ----------
bsrc = os.path.join(OUT, "builder_specs_per_closing_ts.csv")
with open(bsrc, newline="", encoding="utf-8") as fi, open(outpath("d2_builder_specs.csv"), "w", newline="", encoding="utf-8") as fo:
    rd = csv.reader(fi); w = csv.writer(fo, quoting=csv.QUOTE_MINIMAL)
    hdr = next(rd); w.writerow(hdr + ["data_status", "definition_note"])
    for row in rd:
        w.writerow(row + ["STAGING (AI-transcribed, not spot-checked)", "definitions differ by builder; compare each builder with itself"])
bs = pd.read_csv(os.path.join(MAN, "builder_completed_specs_STAGING.csv"))
bs["period_end"] = pd.to_datetime(bs.period_end)
bs = bs[pd.to_numeric(bs.completed_unsold_homes, errors="coerce").notna()]
lastb = bs.groupby("builder").period_end.max()
cut = lastb.max() - pd.DateOffset(months=7)
stale = lastb[lastb < cut]
flags.append("Builder spec counts (d2_builder_specs.csv) come from builder_completed_specs_STAGING.csv: AI-transcribed, NOT spot-checked against filings. Copied from outputs/builder_specs_per_closing_ts.csv without recalculation.")
if len(stale): flags.append("Builders that no longer publish a recent count (last count older than 7 months before latest): " + "; ".join(f"{b} (last {d.date()})" for b, d in stale.items()) + ". In the per-closing series only builders with 8+ quarters appear.")
flags.append("Builder spec definitions differ by builder (and wording changes within builders); compare each builder with itself only. NVR, KB Home and Century do not disclose a comparable count.")
ts = pd.read_csv(bsrc); flags.append("d2_builder_specs.csv builders and last period: " + "; ".join(f"{b} {g.period_end.max()}" for b, g in ts.groupby("builder")) + ".")

# ---------- 3 done; 7. key numbers ----------
K = []
def key(name, value, unit, kind, asof, note=""):
    if isinstance(value, (float, np.floating)): value = round(float(value), 4)
    K.append(dict(name=name, value=value, unit=unit, kind=kind, asof=asof, note=note))
d_tot, v_tot = last(m.HNFSEPUSSA); d_c, v_c = last(m.NHFSEPCS); d_ms, v_ms = last(m.MSACSR); d_h, v_h = last(m.HSN1F)
avg_c = m.loc["2018":"2019", "NHFSEPCS"].mean(); avg_ms = m.loc["2018":"2019", "MSACSR"].mean()
yr_c = m.NHFSEPCS.get(d_c - pd.DateOffset(years=1))
pk = m.loc["1999":, "NHFSEPCS"].idxmax(); pkv = m.NHFSEPCS[pk]
pk22 = m.loc["2022":, "NHFSEPCS"].idxmax(); pk22v = m.NHFSEPCS[pk22]
key("new_homes_for_sale_total_k", v_tot, "thousand", "Observed", iso(d_tot), "HNFSEPUSSA, SA")
key("completed_for_sale_k", v_c, "thousand", "Observed", iso(d_c), "NHFSEPCS, SA")
key("completed_2018_19_avg_k", avg_c, "thousand", "Calc", "2018-01 to 2019-12", "mean of 24 monthly values")
key("completed_pct_above_2018_19_avg", 100 * (v_c / avg_c - 1), "%", "Calc", iso(d_c), "completed / 2018-19 avg - 1")
key("completed_change_vs_year_ago_k", v_c - yr_c, "thousand", "Calc", f"{iso(d_c - pd.DateOffset(years=1))} to {iso(d_c)}", "")
key("completed_peak_since_1999_k", pkv, "thousand", "Observed", iso(pk), "peak month in asof")
key("completed_pct_below_peak_since_1999", 100 * (v_c / pkv - 1), "%", "Calc", iso(d_c), f"vs {iso(pk)} peak; negative = below")
key("completed_peak_since_2022_k", pk22v, "thousand", "Observed", iso(pk22), "peak month in asof; max of 2022-01 onward")
key("months_supply", v_ms, "months", "Observed", iso(d_ms), "MSACSR")
key("months_supply_2018_19_avg", avg_ms, "months", "Calc", "2018-01 to 2019-12", "")
key("completed_months_supply", latest_cms, "months", "Calc", iso(latest_d), "NHFSEPCS / (HSN1F/12)")
key("new_home_sales_saar_k", v_h, "thousand", "Observed", iso(d_h), "HSN1F, SAAR")
hy = m.HSN1F.get(d_h - pd.DateOffset(years=1))
key("new_home_sales_yoy_pct", 100 * (v_h / hy - 1), "%", "Calc", iso(d_h), f"vs {iso(d_h - pd.DateOffset(years=1))} ({hy}k); HSN1F is revised by Census")
dn, vn = cutm.date.iloc[-1], cutm.share_cut.iloc[-1]
key("nahb_share_cutting_prices_pct", vn, "%", "Observed", iso(dn), "NAHB HMI special question; origin: " + cutm.cut_origin.iloc[-1])
ni = nahb[nahb.share_incent.notna()]
key("nahb_share_using_incentives_pct", ni.share_incent.iloc[-1], "%", "Observed", iso(ni.date.iloc[-1]), "NAHB HMI special question")
key("fitted_share_cutting_at_current_inventory_pct", fit_latest, "%", "Calc", iso(latest_d), f"model (b) b0 {b0:.2f} + b1 {b1:.2f} x completed months' supply {latest_cms:.2f}; baseline is {latest_d.strftime('%b-%Y')} Census month")
key("nahb_oct_gap_vs_fitted_pts", oct_gap, "pts", "Calc", iso(dn), "NAHB share (Oct) minus fitted at Sep Census inventory; timing mismatch, see d2_cuts_vs_completed note")
dl, vl = last(a); vl19 = a.get(pd.Timestamp(2019, dl.month, 1)); vly = a.get(dl - pd.DateOffset(years=1))
key("existing_active_listings", vl, "count", "Observed", iso(dl), "Realtor.com, NSA")
key("existing_active_listings_same_month_2019", vl19, "count", "Observed", iso(pd.Timestamp(2019, dl.month, 1)), "NSA, same calendar month")
key("existing_active_listings_pct_vs_2019", 100 * (vl / vl19 - 1), "%", "Calc", iso(dl), "vs same month 2019")
key("existing_active_listings_yoy_pct", 100 * (vl / vly - 1), "%", "Calc", iso(dl), f"vs {iso(dl - pd.DateOffset(years=1))}")
dq, vq = nmw.index[-1], nmw.below5.iloc[-1]; dpk = nmw.below5.idxmax(); vpk = nmw.below5.max()
key("pct_mortgages_below_5_latest", vq, "%", "Observed", iso(dq), "FHFA NMDB, quarter end; lags other series")
key("pct_mortgages_below_5_peak", vpk, "%", "Observed", iso(dpk), "peak month in asof (series starts " + iso(nmw.index.min()) + ")")
key("pct_mortgages_below_5_change_from_peak_pts", vq - vpk, "pts", "Calc", f"{iso(dpk)} to {iso(dq)}", "")
dpm, vpm = last(m.PERMIT1); dst, vst = last(m.HOUST1F)
key("permits_1f_k", vpm, "thousand", "Observed", iso(dpm), "PERMIT1, SAAR")
key("permits_1f_yoy_pct", 100 * (vpm / m.PERMIT1.get(dpm - pd.DateOffset(years=1)) - 1), "%", "Calc", iso(dpm), "")
key("starts_1f_k", vst, "thousand", "Observed", iso(dst), "HOUST1F, SAAR")
key("starts_1f_yoy_pct", 100 * (vst / m.HOUST1F.get(dst - pd.DateOffset(years=1)) - 1), "%", "Calc", iso(dst), "")
key("permits_minus_starts_1f_k", vpm - vst, "thousand", "Calc", iso(dpm), "SAAR difference; permits and starts both SAAR")
if dpm != dst: flags.append(f"PERMIT1 last obs {dpm.date()} differs from HOUST1F {dst.date()}; permits-minus-starts uses each series' own date.")
pd.DataFrame(K).to_csv(outpath("d2_key_numbers.csv"), index=False)
flags.append("Not computed: single-family starts as % of for-sale inventory. Starts are an annualized flow (SAAR, 1-unit) while for-sale inventory is a stock by stage; a ratio of the two has no clean interpretation, so it was skipped rather than shown.")
flags.append("Observed vs Calc: Observed = value as published by source; Calc = derived here. 2018-19 averages are simple means of monthly values.")

# ---------- 8. source log ----------
SRC.append(dict(source="Company filings", series_id="Builder completed unsold homes / closings (staging)",
    url="per-row source_url in builder_completed_specs_STAGING.csv", description="Completed unsold homes and quarterly closings by builder; per-closing ratio uses trailing 4-quarter closings",
    frequency="quarterly (fiscal)", first_obs=bs.period_end.min().date(), last_obs=bs.period_end.max().date(), last_value="n/a (multi-builder)",
    vintage="as filed; AI-transcribed, not spot-checked", access_date=ACCESS, raw_file="data/manual/builder_completed_specs_STAGING.csv; outputs/builder_specs_per_closing_ts.csv",
    used_as="d2_builder_specs"))
cols = ["source", "series_id", "url", "description", "frequency", "first_obs", "last_obs", "last_value", "vintage", "access_date", "raw_file", "used_as"]
pd.DataFrame(SRC)[cols].to_csv(outpath("d2_source_log.csv"), index=False)

# ---------- 9. flags ----------
flags.append(f"NAHB Oct-2026 survey vs Census {latest_d.strftime('%b-%Y')}: Census new-home inventory for Oct-2026 is not yet published, so the latest NAHB reading has no same-month supply. The Oct gap vs the model uses Sep inventory (one month older) and is not a like-for-like residual.")
flags.append("Models are monthly OLS on a short sample (n=49), with strongly trending series; Newey-West errors help with autocorrelation and the level fit in (b) may partly reflect two trending series (the change-on-change models (d),(e) are not significant). Test, not a forecast. Models (a),(c),(d),(e) show no significant relationship.")
flags.append("Proxy models: median new-home price is mix-sensitive and MSPNHSUS is NSA; yoy uses a 3-month average; Newey-West lag 18. These are not price-cut measures.")
flags.append("FRED series are current vintage (revised); HSN1F and inventory series are revised by Census, so recomputing later may change numbers slightly. PERMIT1 and HOUST1F downloaded 2026-11-10 with default curl user agent.")
with open(outpath("d2_data_flags.txt"), "w", encoding="utf-8") as f:
    f.write(f"Driver 2 data flags, built {ACCESS} by scripts/d2_build_data.py\n")
    for x in flags: f.write("- " + x + "\n")
print("OK")
print(R[["model", "n", "intercept", "slope", "nw_se", "t", "p", "r2"]].to_string())
print(pd.DataFrame(K)[["name", "value", "unit", "kind", "asof"]].to_string())
