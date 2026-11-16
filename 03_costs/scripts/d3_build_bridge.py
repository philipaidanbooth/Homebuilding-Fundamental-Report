"""Driver 3 cost bridge data. Run from the 03_costs folder: python3 scripts/d3_build_bridge.py
Reads ../data_ready/builders_latest_quarter.csv (read-only), data/raw/*.csv and outputs/d3_input_components.csv.
Writes outputs/d3_cost_vs_price_by_builder.csv, outputs/d3_bridge_numbers.csv, outputs/d3_filing_commentary.csv.
No downloading. Filing facts typed below were each checked against the filing on 2026-11-10; see VERIFICATION_LOG.csv."""
import pandas as pd

O, DR, RAW = "outputs", "../data_ready", "data/raw"

# ---- 1. builder cost per home vs price (company evidence) ---------------------------
b = pd.read_csv(f"{DR}/builders_latest_quarter.csv")
cp = b[["builder", "ticker", "period", "asp_yoy_pct", "implied_cost_yoy_pct", "gm_change_bps"]].copy()
cp.to_csv(f"{O}/d3_cost_vs_price_by_builder.csv", index=False)
cost, asp = b["implied_cost_yoy_pct"], b["asp_yoy_pct"]

# ---- 2. PPI series from raw FRED files ----------------------------------------------
def load(sid):
    d = pd.read_csv(f"{RAW}/{sid}_2026-11-10.csv")
    d.columns = ["date", "v"]; d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["v"].astype(float)

def yoy(s):            # latest month vs same month a year earlier
    return (s.iloc[-1] / s.loc[s.index[-1] - pd.DateOffset(years=1)] - 1) * 100

def yoy3(s):           # latest 3-month average vs same 3 months a year earlier
    last = s.iloc[-3:]; prev = s.loc[last.index - pd.DateOffset(years=1)]
    return (last.mean() / prev.mean() - 1) * 100

goods, lumber, steel, diesel = load("WPUIP2311101"), load("WPU0811"), load("WPU1017"), load("WPU057303")
asof = str(goods.index[-1].date())
lum_win = lumber.loc["2020-01-01":"2022-12-01"]
lum_peak, lum_peak_date = lum_win.max(), lum_win.idxmax()

rows = [
    ("cost_per_home_yoy_median_pct", round(float(cost.median()), 1), "latest quarter", "Calc: median of 7 builders; implied cost per home = ASP x (1 - gross margin)"),
    ("cost_per_home_yoy_min_pct", round(float(cost.min()), 1), "latest quarter", "Calc"),
    ("cost_per_home_yoy_max_pct", round(float(cost.max()), 1), "latest quarter", "Calc"),
    ("asp_yoy_median_pct", round(float(asp.median()), 1), "latest quarter", "Calc: median of 7 builders"),
    ("n_builders", int(len(b)), "latest quarter", "Count"),
    ("n_builders_cost_per_home_up", int((cost > 0).sum()), "latest quarter", "Count: implied cost per home y/y above zero"),
    ("n_builders_cost_per_home_below_ppi", int((cost < yoy(goods)).sum()), "latest quarter", "Count: implied cost per home y/y below goods-input PPI y/y"),
    ("goods_input_ppi_yoy_pct", round(yoy(goods), 1), asof, "Calc from WPUIP2311101: latest month vs same month a year earlier"),
    ("goods_input_ppi_3m_yoy_pct", round(yoy3(goods), 1), asof, "Calc from WPUIP2311101: 3-month average vs same 3 months a year earlier"),
    ("steel_ppi_yoy_pct", round(yoy(steel), 1), asof, "Calc from WPU1017, single month"),
    ("steel_ppi_3m_yoy_pct", round(yoy3(steel), 1), asof, "Calc from WPU1017, 3-month average"),
    ("lumber_ppi_yoy_pct", round(yoy(lumber), 1), asof, "Calc from WPU0811, single month"),
    ("lumber_ppi_3m_yoy_pct", round(yoy3(lumber), 1), asof, "Calc from WPU0811, 3-month average"),
    ("lumber_vs_2020_22_peak_pct", round((lumber.iloc[-1] / lum_peak - 1) * 100, 1), asof, f"Calc from WPU0811: latest index vs peak month of 2020-22 ({lum_peak_date:%b %Y})"),
    ("diesel_ppi_yoy_pct", round(yoy(diesel), 1), asof, "Calc from WPU057303, single month; general fuel index, not in section body"),
    ("diesel_ppi_3m_yoy_pct", round(yoy3(diesel), 1), asof, "Calc from WPU057303, 3-month average; not in section body"),
    # Lennar 10-Q, quarter ended Aug 31 2026, MD&A Outlook (checked 2026-11-10)
    ("lennar_construction_cost_per_sqft_yoy_pct", -6.0, "Q3 FY26", "Lennar 10-Q MD&A Outlook: 'approximately $80, down 6% from a year ago'"),
    ("lennar_construction_cost_per_sqft_usd", 80.0, "Q3 FY26", "Lennar 10-Q MD&A Outlook (approximate)"),
    ("lennar_cycle_time_days", 116.0, "Q3 FY26", "Lennar 10-Q MD&A Outlook; 126 days a year ago"),
    ("lennar_cycle_time_days_year_ago", 126.0, "Q3 FY25", "Lennar 10-Q MD&A Outlook"),
]
pd.DataFrame(rows, columns=["name", "value", "asof", "basis"]).to_csv(f"{O}/d3_bridge_numbers.csv", index=False)

# ---- 3. filing commentary on cost (each quote checked against the document) ---------
com = [
    ("Lennar", "10-Q, quarter ended Aug 31 2026, MD&A", True, "lower revenue per square foot and higher land costs year over year", "decrease in construction costs"),
    ("D.R. Horton", "10-Q, quarter ended Jun 30 2026, MD&A", False, "the average cost of our homes closed increasing along with a decrease in the average selling price of those homes", ""),
    ("PulteGroup", "10-Q, quarter ended Sep 30 2026, MD&A", True, "increased land acquisition and development costs", ""),
    ("NVR", "8-K Ex. 99.1, quarter ended Sep 30 2026", True, "higher lot costs, pricing pressure due to continued affordability challenges", ""),
    ("KB Home", "Earnings release, quarter ended Aug 31 2026", True, "continued pricing pressure, higher relative land costs and reduced operating leverage", ""),
    ("Meritage Homes", "Earnings release, quarter ended Sep 30 2026", True, "higher lot and incentive costs coupled with lost leverage on lower home closing revenue", "direct cost savings"),
    ("Toll Brothers", "Earnings release, quarter ended Jul 31 2026", False, "", ""),
]
pd.DataFrame(com, columns=["builder", "document", "cites_land_or_lot_cost", "margin_reason_quote", "cost_offset_quote"]).to_csv(f"{O}/d3_filing_commentary.csv", index=False)
print(pd.DataFrame(rows, columns=["name", "value", "asof", "basis"]).to_string())
