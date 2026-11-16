"""Driver 3 exhibit data: reads existing d3_*.csv (outputs/) and ../data_ready/, writes
outputs/d3_key_numbers_v2.csv, d3_land_model.csv, d3_labor_summary.csv. Run from the 03_costs folder.
No downloading. All inputs are CSVs logged in outputs/d3_source_log.csv."""
import pandas as pd, numpy as np

O, DR = "outputs", "../data_ready"
kn = pd.read_csv(f"{O}/d3_key_numbers.csv").set_index("metric")["value"]
kn = kn.apply(lambda v: pd.to_numeric(v, errors="coerce") if str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else v)
comp = pd.read_csv(f"{O}/d3_input_components.csv").set_index("series_id")
b = pd.read_csv(f"{DR}/builders_latest_quarter.csv")
sens = pd.read_csv(f"{DR}/margin_sensitivity_grid.csv")
fb = pd.read_csv(f"{DR}/labor_foreign_born_cps_2025.csv").set_index("metric")["value"]
lab = pd.read_csv(f"{O}/d3_labor_monthly.csv", parse_dates=["date"]).set_index("date")

# JOLTS 3-month average, latest 3 months with data vs same months a year earlier
jo = lab["constr_job_openings_k"].dropna()
last3 = jo.iloc[-3:]; prev3 = jo.loc[last3.index - pd.DateOffset(years=1)]
jolts_3m = last3.mean(); jolts_3m_prev = prev3.mean()
jolts_yoy = (jolts_3m / jolts_3m_prev - 1) * 100
lab_out = pd.DataFrame([
    dict(name="constr_ahe_yoy_pct", value=kn["constr_ahe_yoy_pct"], asof="2026-10-01", basis="Calc from CES2000000003"),
    dict(name="private_ahe_yoy_pct", value=kn["private_ahe_yoy_pct"], asof="2026-10-01", basis="Calc from CES0500000003"),
    dict(name="constr_employment_yoy_pct", value=kn["constr_employment_yoy_pct"], asof="2026-10-01", basis="Calc from USCONS"),
    dict(name="jolts_openings_3m_avg_k", value=round(jolts_3m, 1), asof=str(last3.index[-1].date()), basis="Calc: mean of last 3 months, JTS2300JOL"),
    dict(name="jolts_openings_3m_avg_prev_k", value=round(jolts_3m_prev, 1), asof=str((last3.index[-1] - pd.DateOffset(years=1)).date()), basis="Calc: same 3 months a year earlier"),
    dict(name="jolts_openings_3m_yoy_pct", value=round(jolts_yoy, 1), asof=str(last3.index[-1].date()), basis="Calc"),
    dict(name="jolts_openings_single_month_yoy_pct", value=kn["constr_job_openings_yoy_pct"], asof="2026-09-01", basis="Calc; one volatile month"),
    dict(name="foreign_born_share_constr_extraction_occ_pct", value=fb["Foreign-born share of construction and extraction occupation workers"], asof="2025", basis="Calc from BLS CPS annual 2025; occupation basis; see data_ready flags"),
])
lab_out.to_csv(f"{O}/d3_labor_summary.csv", index=False)

# land model: owned share of owned+controlled lots, vs gross margin change
land = pd.DataFrame([
    dict(builder="Lennar", owned_share_pct=2.4, basis="Observed/Calc: owned homesites / total homesites (10-Q, Aug 31 2026)"),
    dict(builder="D.R. Horton", owned_share_pct=23.0, basis="Observed: owned lots 130,500 (23%) of owned + controlled (FY4Q26 release)"),
    dict(builder="Toll Brothers", owned_share_pct=round(31817 / 75486 * 100, 1), basis="Calc: owned 31,817 / (owned + optioned 75,486) (Q3 FY26 release)"),
    dict(builder="PulteGroup", owned_share_pct=45.0, basis="Observed: owned share of controlled lots (10-Q, Sep 30 2026)"),
    dict(builder="KB Home", owned_share_pct=60.0, basis="Observed: about 60% of 61,581 lots owned or under contract are owned (Q3 release)"),
    dict(builder="NVR", owned_share_pct=np.nan, basis="Asset-light: lots controlled through purchase agreements (185,300); owned share not split in the release"),
    dict(builder="Meritage Homes", owned_share_pct=np.nan, basis="Not stated: about 71,800 owned and controlled lots, no split"),
])
land = land.merge(b[["builder", "gm_change_bps"]], on="builder", how="left")
land.to_csv(f"{O}/d3_land_model.csv", index=False)
ok = land.dropna(subset=["owned_share_pct"])
r = float(np.corrcoef(ok.owned_share_pct, ok.gm_change_bps)[0, 1])

cost, asp = b["implied_cost_yoy_pct"], b["asp_yoy_pct"]
s22 = sens[(sens.starting_gm_pct == 22.0) & (sens.shock_type == "price")]["margin_change_bps"].iloc[0]
kv = [
    ("goods_input_ppi_yoy_pct", kn["goods_input_ppi_yoy_pct"], "2026-09-01", "Observed (WPUIP2311101)"),
    ("diesel_ppi_yoy_pct", kn["diesel_ppi_yoy_pct"], "2026-09-01", "Observed (WPU057303)"),
    ("steel_ppi_yoy_pct", kn["steel_mill_ppi_yoy_pct"], "2026-09-01", "Observed (WPU1017)"),
    ("lumber_ppi_yoy_pct", comp.loc["WPU0811", "yoy_pct"], "2026-09-01", "Observed (WPU0811)"),
    ("materials_ppi_yoy_pct", comp.loc["WPUID612", "yoy_pct"], "2026-09-01", "Observed (WPUID612)"),
    ("broad_residential_inputs_ppi_yoy_pct", comp.loc["WPUIP2311001", "yoy_pct"], "2026-09-01", "Observed (WPUIP2311001)"),
    ("median_new_home_price_yoy_pct", kn["median_new_home_price_yoy_pct"], "2026-09-01", "Observed (MSPNHSUS)"),
    ("cost_per_home_yoy_median_7_builders_pct", round(float(cost.median()), 1), "latest quarter", "Calc: median of 7 builders, ASP x (1 - GM)"),
    ("cost_per_home_yoy_min_pct", round(float(cost.min()), 1), "latest quarter", "Calc"),
    ("cost_per_home_yoy_max_pct", round(float(cost.max()), 1), "latest quarter", "Calc"),
    ("asp_yoy_median_7_builders_pct", round(float(asp.median()), 1), "latest quarter", "Calc: median of 7 builders"),
    ("n_builders_cost_per_home_below_ppi", int((cost < kn["goods_input_ppi_yoy_pct"]).sum()), "latest quarter", "Calc"),
    ("margin_per_1pct_price_bps_at_22pct_gm", round(float(s22), 1), "n/a", "Calc: (1 - GM) x 1%"),
    ("owned_share_vs_gm_change_corr_5_builders", round(r, 2), "latest quarter", "Calc: Pearson r, n = 5; descriptive only"),
    ("lennar_construction_cost_per_sqft_yoy_pct", -6.0, "Q3 FY26", "Observed (10-Q MD&A); see d3_lennar_10q_extract.csv"),
    ("lennar_cycle_time_days", 116, "Q3 FY26", "Observed (10-Q MD&A)"),
    ("dhi_land_balance_per_owned_lot_yoy_pct", 15.8, "Jun-2026", "Calc, rough proxy (flag 10)"),
]
pd.DataFrame(kv, columns=["name", "value", "asof", "basis"]).to_csv(f"{O}/d3_key_numbers_v2.csv", index=False)
print(pd.DataFrame(kv, columns=["name", "value", "asof", "basis"]).to_string())
print(lab_out.to_string()); print(land.to_string())
