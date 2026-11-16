"""Driver 4 starts model: fit, out-of-sample test by horizon, scenario paths, horizon band.
Run from the 04_margins folder:  python3 scripts/d4_starts_model.py
Inputs : ../02_inventory/data/raw/PERMIT1_2026-11-10.csv, HOUST1F_2026-11-10.csv (Census via FRED)
         data/d4_permit_assumptions.csv (this report's permit assumption for each scenario)
Outputs: outputs/d4_model_estimates.csv, d4_oos_by_horizon.csv, d4_oos_errors.csv,
         d4_scenario_paths.csv, d4_history.csv
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path

RAW = Path("../02_inventory/data/raw"); OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
START, END = "2018-01-01", "2026-09-01"
OOS_FIRST_TARGET = "2024-01-01"; HORIZONS = range(1, 13); Z80 = 1.2816

p = pd.read_csv(RAW / "PERMIT1_2026-11-10.csv", parse_dates=["observation_date"]).set_index("observation_date")["PERMIT1"]
s = pd.read_csv(RAW / "HOUST1F_2026-11-10.csv", parse_dates=["observation_date"]).set_index("observation_date")["HOUST1F"]
d = pd.DataFrame({"starts": s, "permits": p}).loc[START:END]
d["starts_3m"] = d["starts"].rolling(3).mean(); d["permits_3m"] = d["permits"].rolling(3).mean()
d["starts_3m_lag1"] = d["starts_3m"].shift(1)

def fit(df):
    m = df.dropna()
    X = sm.add_constant(m[["permits_3m", "starts_3m_lag1"]])
    return sm.OLS(m["starts_3m"], X).fit(cov_type="HC3"), m

res, m = fit(d)
c, b, r = res.params["const"], res.params["permits_3m"], res.params["starts_3m_lag1"]
resid = m["starts_3m"] - res.fittedvalues
rmse_in = float(np.sqrt((resid ** 2).sum() / (len(m) - 3)))
est = pd.DataFrame({"term": ["const", "permits_3m", "starts_3m_lag1"], "coef": res.params.values,
                    "hc3_se": res.bse.values, "p_value": res.pvalues.values})
extra = pd.DataFrame({"term": ["n_obs", "r_squared", "resid_rmse_k", "sample_start", "sample_end"],
                      "coef": [len(m), res.rsquared, rmse_in, np.nan, np.nan]})
extra["note"] = ["", "", "", m.index[0].strftime("%Y-%m"), m.index[-1].strftime("%Y-%m")]
pd.concat([est, extra]).to_csv(OUT / "d4_model_estimates.csv", index=False)
d.assign(fitted=res.fittedvalues, resid=resid).round(3).to_csv(OUT / "d4_history.csv", index_label="date")

# ---- out-of-sample, expanding window. Same targets for every horizon (Jan 2024 - Sep 2026).
# For target month T and horizon h: fit on data through T-h, then iterate the equation h steps.
#   known  = realized permits fed in (tests the equation, given a correct permit path)
#   flat   = permits held at their last known 3M average (a forecast with no knowledge of future permits)
#   nochg  = benchmark: starts 3M average stays at its last known value
rows = []
targets = d.loc[OOS_FIRST_TARGET:].index
for h in HORIZONS:
    for T in targets:
        o = T - pd.DateOffset(months=h)
        fr, _ = fit(d.loc[:o]); cc, bb, rr = fr.params
        last_s, last_p = d.at[o, "starts_3m"], d.at[o, "permits_3m"]
        sk = sf = last_s
        for j in range(1, h + 1):
            t = o + pd.DateOffset(months=j)
            sk = cc + bb * d.at[t, "permits_3m"] + rr * sk
            sf = cc + bb * last_p + rr * sf
        rows.append(dict(horizon=h, target=T.strftime("%Y-%m"), actual=d.at[T, "starts_3m"],
                         model_known_permits=sk, model_flat_permits=sf, no_change=last_s))
e = pd.DataFrame(rows); e.round(2).to_csv(OUT / "d4_oos_errors.csv", index=False)
def stats(g):
    out = {"n_forecasts": len(g)}
    for k in ["model_known_permits", "model_flat_permits", "no_change"]:
        err = g[k] - g["actual"]
        out[f"{k}_mae_k"] = err.abs().mean(); out[f"{k}_rmse_k"] = np.sqrt((err ** 2).mean())
        out[f"{k}_mape_pct"] = (err.abs() / g["actual"]).mean() * 100
    return pd.Series(out)
oos = e.groupby("horizon").apply(stats).reset_index()
# band = model error once a permit path is given (excludes uncertainty about permits themselves)
oos["band80_half_width_k"] = Z80 * oos["model_known_permits_rmse_k"]
oos.round(3).to_csv(OUT / "d4_oos_by_horizon.csv", index=False)

# ---- scenario paths from this report's permit assumptions
pa = pd.read_csv("data/d4_permit_assumptions.csv")
out = []
hist_p = d["permits"]
for scen in ["upside", "base", "downside"]:
    mp = pa.set_index("month")[f"{scen}_permits_k"]
    full = pd.concat([hist_p, pd.Series(mp.values, index=pd.to_datetime(mp.index + "-01"))])
    p3 = full.rolling(3).mean(); prev = d["starts_3m"].iloc[-1]
    for i, t in enumerate(pd.to_datetime(mp.index + "-01"), start=1):
        prev = c + b * p3[t] + r * prev
        hw = float(oos.loc[oos.horizon == i, "band80_half_width_k"].iloc[0])
        out.append(dict(scenario=scen, month=t.strftime("%Y-%m"), horizon=i, permits_k=full[t], permits_3m_k=p3[t],
                        starts_3m_k=prev, band80_low=prev - hw, band80_high=prev + hw))
pd.DataFrame(out).round(1).to_csv(OUT / "d4_scenario_paths.csv", index=False)
print(est.round(3).to_string()); print("n", len(m), "R2", round(res.rsquared, 3), "rmse", round(rmse_in, 2))
print(oos.round(2).to_string())
