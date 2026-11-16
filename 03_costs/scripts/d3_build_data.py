"""Driver 3 data build: pulls cost and labor series from FRED, reuses Driver 1 raw files,
writes CSVs to outputs/ and a source log. Run from the 03_costs folder."""
import os, sys, io, urllib.request
import pandas as pd

ACCESS = "2026-11-10"
RAW, OUT = "data/raw", "outputs"
D1RAW = "../01_affordability/data/raw"   # reused, not re-pulled
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)

NEW = {  # series_id: (description, frequency)
 "WPUIP2311101": ("PPI net goods inputs to single-family residential construction (index, Dec 2014=100)", "monthly"),
 "WPUIP2311001": ("PPI inputs to residential construction, goods/services/trade (index)", "monthly"),
 "WPUID612":     ("PPI materials and components for construction (index)", "monthly"),
 "WPU0811":      ("PPI softwood lumber (index)", "monthly"),
 "WPU1017":      ("PPI steel mill products (index)", "monthly"),
 "WPU057303":    ("PPI No. 2 diesel fuel (index)", "monthly"),
 "CES2000000003":("Average hourly earnings, all employees, construction ($)", "monthly"),
 "CES2000000008":("Average hourly earnings, production and nonsupervisory, construction ($)", "monthly"),
 "USCONS":       ("All employees, construction (thousands, SA)", "monthly"),
 "JTS2300JOL":   ("JOLTS construction job openings (thousands, SA)", "monthly"),
 "JTS2300HIR":   ("JOLTS construction hires rate (%, SA)", "monthly"),
}
REUSED = {
 "MSPNHSUS": ("Median sales price of new houses sold, US ($)", "monthly", "MSPNHSUS_2026-11-09.csv"),
 "CES0500000003": ("Average hourly earnings, all employees, total private ($)", "monthly", "CES0500000003_2026-11-09.csv"),
 "UNRATE": ("Unemployment rate (%)", "monthly", "UNRATE_2026-11-09.csv"),
}

def load(path):
    d = pd.read_csv(path, na_values=["."]); d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"]); return d.dropna().set_index("date")["v"]

log, series = [], {}
for sid, (desc, freq) in NEW.items():
    f = f"{RAW}/{sid}_{ACCESS}.csv"
    if not os.path.exists(f):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        with urllib.request.urlopen(url, timeout=60) as r: open(f, "wb").write(r.read())
    s = load(f); series[sid] = s
    log.append(dict(source="FRED", series_id=sid, description=desc, frequency=freq, first_obs=s.index[0].date(),
        last_obs=s.index[-1].date(), last_value=s.iloc[-1], vintage="FRED current vintage as of access date (not ALFRED real-time)",
        accessed=ACCESS, raw_file=f, url=f"https://fred.stlouisfed.org/series/{sid}", used_as="data (new, driver 3)"))
for sid, (desc, freq, fn) in REUSED.items():
    s = load(f"{D1RAW}/{fn}"); series[sid] = s
    log.append(dict(source="FRED", series_id=sid, description=desc, frequency=freq, first_obs=s.index[0].date(),
        last_obs=s.index[-1].date(), last_value=s.iloc[-1], vintage="FRED current vintage as of Driver 1 pull (2026-11-09)",
        accessed="2026-11-09", raw_file=f"{D1RAW}/{fn}", url=f"https://fred.stlouisfed.org/series/{sid}", used_as="data (reused from driver 1)"))

def yoy(s): return (s / s.shift(12) - 1) * 100
m = lambda sid: series[sid].resample("MS").last()

# 1. PPI vs new-home price, monthly from 2018
ppi = pd.DataFrame({"goods_input_ppi": m("WPUIP2311101"), "median_new_home_price": m("MSPNHSUS")})
ppi["goods_input_ppi_yoy_pct"] = yoy(ppi["goods_input_ppi"])
ppi["median_new_home_price_yoy_pct"] = yoy(ppi["median_new_home_price"])
ppi["gap_pts_input_minus_price"] = ppi["goods_input_ppi_yoy_pct"] - ppi["median_new_home_price_yoy_pct"]
ppi = ppi.loc["2018-01-01":].round(3); ppi.index.name = "date"; ppi.to_csv(f"{OUT}/d3_ppi_vs_price.csv")

# 2. Input components snapshot (latest and y/y)
names = {"WPUIP2311101": "Goods inputs to single-family construction (report series)",
         "WPUIP2311001": "Inputs to residential construction (broader PPI series)",
         "WPUID612": "Materials and components for construction",
         "WPU0811": "Softwood lumber", "WPU1017": "Steel mill products", "WPU057303": "No. 2 diesel fuel"}
rows = []
for sid, nm in names.items():
    s = m(sid); y = yoy(s)
    rows.append(dict(series_id=sid, name=nm, last_obs=s.index[-1].date(), index_level=round(s.iloc[-1], 3),
        yoy_pct=round(y.iloc[-1], 1), vs_dec2024_pct=round((s.iloc[-1] / s.loc["2024-12-01"] - 1) * 100, 1),
        vs_dec2019_pct=round((s.iloc[-1] / s.loc["2019-12-01"] - 1) * 100, 1)))
pd.DataFrame(rows).to_csv(f"{OUT}/d3_input_components.csv", index=False)

# 3. Input components monthly from 2018, indexed Jan 2020 = 100 (for a shared chart)
comp = pd.DataFrame({sid: m(sid) for sid in names}).loc["2018-01-01":]
comp = (comp / comp.loc["2020-01-01"] * 100).round(2); comp.index.name = "date"
comp.to_csv(f"{OUT}/d3_input_components_index_2020.csv")

# 4. Labor
lab = pd.DataFrame({"constr_ahe": m("CES2000000003"), "constr_ahe_prod": m("CES2000000008"),
                    "private_ahe": m("CES0500000003"), "constr_employment_k": m("USCONS"),
                    "constr_job_openings_k": m("JTS2300JOL"), "constr_hires_rate_pct": m("JTS2300HIR")})
lab["constr_ahe_yoy_pct"] = yoy(lab["constr_ahe"]); lab["private_ahe_yoy_pct"] = yoy(lab["private_ahe"])
lab["constr_employment_yoy_pct"] = yoy(lab["constr_employment_k"]); lab["constr_openings_yoy_pct"] = yoy(lab["constr_job_openings_k"])
lab = lab.loc["2018-01-01":].round(3); lab.index.name = "date"; lab.to_csv(f"{OUT}/d3_labor_monthly.csv")

# 5. Lennar figures carried from the earlier driver 3 draft (Lennar Q3 FY26 8-K); NOT re-verified here
asp_now, asp_prev, gm_now, gm_prev = 372000, 383000, 0.158, 0.175
cost_now, cost_prev = asp_now * (1 - gm_now), asp_prev * (1 - gm_prev)
ln = [("asp_q3fy26", asp_now, "$", "Observed (draft, Lennar 8-K)"), ("asp_q3fy25", asp_prev, "$", "Observed (draft, Lennar 8-K)"),
      ("gross_margin_q3fy26", gm_now * 100, "%", "Observed (draft, Lennar 8-K)"), ("gross_margin_q3fy25", gm_prev * 100, "%", "Observed (draft, Lennar 8-K)"),
      ("q4_gm_guide_low", 15.5, "%", "Observed (draft, Lennar 8-K)"), ("q4_gm_guide_high", 16.0, "%", "Observed (draft, Lennar 8-K)"),
      ("construction_cost_per_sqft_yoy", -6.0, "%", "Observed (draft, Lennar 8-K)"),
      ("homesites_owned_or_controlled", 488000, "count", "Observed (draft, Lennar 8-K)"), ("homesites_owned_share_max", 2.5, "% (under)", "Observed (draft, Lennar 8-K)"),
      ("implied_cost_of_sales_q3fy26", round(cost_now), "$", "Calc: ASP x (1 - gross margin)"),
      ("implied_cost_of_sales_q3fy25", round(cost_prev), "$", "Calc: ASP x (1 - gross margin)"),
      ("implied_cost_yoy_pct", round((cost_now / cost_prev - 1) * 100, 1), "%", "Calc"),
      ("asp_yoy_pct", round((asp_now / asp_prev - 1) * 100, 1), "%", "Calc")]
pd.DataFrame(ln, columns=["metric", "value", "unit", "basis"]).to_csv(f"{OUT}/d3_lennar_price_vs_cost.csv", index=False)

# 6. Key numbers
last = ppi.dropna().iloc[-1]
c = pd.DataFrame(rows).set_index("series_id")
kn = [("goods_input_ppi_yoy_pct", c.loc["WPUIP2311101", "yoy_pct"]), ("goods_input_ppi_vs_dec2024_pct", c.loc["WPUIP2311101", "vs_dec2024_pct"]),
      ("goods_input_ppi_last_obs", str(c.loc["WPUIP2311101", "last_obs"])), ("median_new_home_price_yoy_pct", round(last["median_new_home_price_yoy_pct"], 1)),
      ("input_minus_price_gap_pts", round(last["gap_pts_input_minus_price"], 1)),
      ("diesel_ppi_yoy_pct", c.loc["WPU057303", "yoy_pct"]), ("steel_mill_ppi_yoy_pct", c.loc["WPU1017", "yoy_pct"]),
      ("softwood_lumber_ppi_yoy_pct", c.loc["WPU0811", "yoy_pct"]),
      ("constr_ahe_yoy_pct", round(lab["constr_ahe_yoy_pct"].dropna().iloc[-1], 1)), ("private_ahe_yoy_pct", round(lab["private_ahe_yoy_pct"].dropna().iloc[-1], 1)),
      ("constr_employment_yoy_pct", round(lab["constr_employment_yoy_pct"].dropna().iloc[-1], 1)),
      ("constr_job_openings_yoy_pct", round(lab["constr_openings_yoy_pct"].dropna().iloc[-1], 1))]
pd.DataFrame(kn, columns=["metric", "value"]).to_csv(f"{OUT}/d3_key_numbers.csv", index=False)

pd.DataFrame(log).to_csv(f"{OUT}/d3_source_log.csv", index=False)
print(pd.DataFrame(kn).to_string(index=False))
