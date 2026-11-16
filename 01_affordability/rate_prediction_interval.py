import numpy as np, pandas as pd, sys
from math import erf, sqrt
src = sys.argv[1] if len(sys.argv)>1 else "data/raw/MORTGAGE30US_2026-11-09.csv"
OUTDIR = sys.argv[2] if len(sys.argv)>2 else "outputs"
s = pd.read_csv(src, parse_dates=["observation_date"]).set_index("observation_date")["MORTGAGE30US"].dropna()
s = s.resample("W-THU").last().dropna()          # weekly, Thursday
v = s.values; n = len(v); dates = s.index
Z80 = 1.2815515655
def ncdf(x): return 0.5*(1+erf(x/sqrt(2)))

def interval(i, h, method):
    """Return (lo10, hi10) for rate at i+h using only info up to i."""
    d1 = np.diff(v[:i+1])
    dh = v[h:i+1] - v[:i+1-h]                      # h-week changes fully observed by i
    if method == "Normal, 2y vol":
        sd = d1[-104:].std(ddof=1)*sqrt(h); return v[i]-Z80*sd, v[i]+Z80*sd
    if method == "Normal, all-history vol":
        sd = d1.std(ddof=1)*sqrt(h); return v[i]-Z80*sd, v[i]+Z80*sd
    if method == "Empirical, all history":
        q = np.quantile(dh, [.1,.9]); return v[i]+q[0], v[i]+q[1]
    if method == "Empirical, last 20y":
        q = np.quantile(dh[-1040:], [.1,.9]); return v[i]+q[0], v[i]+q[1]

methods = ["Normal, 2y vol","Normal, all-history vol","Empirical, all history","Empirical, last 20y"]
rows=[]
for h in (26,52):
    for start,label in (("1990-01-01","1990-2026"),("2010-01-01","2010-2026")):
        i0 = np.searchsorted(dates, pd.Timestamp(start))
        for m in methods:
            hit=below=above=0; N=0; width=[]
            for i in range(max(i0,1100 if m=="Empirical, last 20y" else i0), n-h):
                lo,hi = interval(i,h,m); a=v[i+h]; N+=1; width.append(hi-lo)
                hit += lo<=a<=hi; below += a<lo; above += a>hi
            neff = N/h
            cov = hit/N
            z = (cov-0.8)/sqrt(0.8*0.2/neff)
            p = 2*(1-ncdf(abs(z)))
            rows.append(dict(horizon_weeks=h, sample=label, method=m, origins=N, eff_indep_obs=round(neff,1),
                 coverage_80=cov, share_below=below/N, share_above=above/N, avg_width_pp=np.mean(width), z_vs_80=z, p_value=p))
bt = pd.DataFrame(rows)
pd.set_option("display.width",220); pd.set_option("display.float_format","{:.3f}".format)
print(bt.to_string(index=False)); bt.to_csv(f"{OUTDIR}/pi_backtest.csv", index=False)

# latest-origin intervals
i = n-1; cur = v[i]; print("\nlatest", dates[i].date(), cur)
out=[]
for h,lab in ((26,"6 months"),(52,"12 months")):
    dh = v[h:i+1]-v[:i+1-h]
    for m,arr in (("Empirical, all history",dh),("Empirical, last 20y",dh[-1040:]),("Empirical, since 2010",v[h:i+1][dates[h:i+1]>='2010-01-01']-v[:i+1-h][dates[h:i+1]>='2010-01-01'])):
        qs = np.quantile(arr,[.05,.1,.25,.5,.75,.9,.95])+cur
        r=dict(horizon=lab, method=m, n=len(arr), p05=qs[0],p10=qs[1],p25=qs[2],p50=qs[3],p75=qs[4],p90=qs[5],p95=qs[6],
               prob_le_6_18=np.mean(cur+arr<=6.18), prob_le_6_50=np.mean(cur+arr<=6.5), prob_ge_7_50=np.mean(cur+arr>=7.5), prob_ge_8_00=np.mean(cur+arr>=8.0), sd_change=arr.std(ddof=1))
        out.append(r)
    d1=np.diff(v); sd=d1[-104:].std(ddof=1)*sqrt(h)
    out.append(dict(horizon=lab, method="Normal, 2y vol", n=104, p05=cur-1.645*sd,p10=cur-Z80*sd,p25=cur-.6745*sd,p50=cur,p75=cur+.6745*sd,p90=cur+Z80*sd,p95=cur+1.645*sd,
      prob_le_6_18=ncdf((6.18-cur)/sd), prob_le_6_50=ncdf((6.5-cur)/sd), prob_ge_7_50=1-ncdf((7.5-cur)/sd), prob_ge_8_00=1-ncdf((8.0-cur)/sd), sd_change=sd))
    sd=d1.std(ddof=1)*sqrt(h)
    out.append(dict(horizon=lab, method="Normal, all-history vol", n=len(d1), p05=cur-1.645*sd,p10=cur-Z80*sd,p25=cur-.6745*sd,p50=cur,p75=cur+.6745*sd,p90=cur+Z80*sd,p95=cur+1.645*sd,
      prob_le_6_18=ncdf((6.18-cur)/sd), prob_le_6_50=ncdf((6.5-cur)/sd), prob_ge_7_50=1-ncdf((7.5-cur)/sd), prob_ge_8_00=1-ncdf((8.0-cur)/sd), sd_change=sd))
lt = pd.DataFrame(out); print(lt.to_string(index=False)); lt.to_csv(f"{OUTDIR}/pi_latest.csv", index=False)
# scenario bins from the selected method (normal, all-history vol, h=52 and 26)
edges=[-9,6.25,6.75,7.5,8.0,99]; labs=["<6.25","6.25-6.75","6.75-7.50","7.50-8.00",">8.00"]
d1=np.diff(v); rows=[]
for h in (26,52):
    sd=d1.std(ddof=1)*sqrt(h); cdf=[ncdf((e-cur)/sd) if abs(e)<50 else (0 if e<0 else 1) for e in edges]
    rows.append(dict(horizon_weeks=h, sd_pp=sd, **{l:cdf[k+1]-cdf[k] for k,l in enumerate(labs)}))
sb=pd.DataFrame(rows); print("\nScenario-bin probabilities (normal, all-history vol):\n", sb.to_string(index=False)); sb.to_csv(f"{OUTDIR}/pi_scenario_bins.csv", index=False)
print("\nweekly change sd: all", np.diff(v).std(ddof=1), " 2y", np.diff(v)[-104:].std(ddof=1), " since2010", np.diff(v[dates>='2010-01-01']).std(ddof=1))
