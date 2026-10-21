"""Does SI add information beyond RV20? Full-sample (labelled rows only) logit with HAC (Newey-West, 10 lags) SEs,
plus per-period coefficients and SI effect within RV20 terciles. Primary target: Future_Return_5D < -0.03."""
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
import sys; sys.argv = [sys.argv[0], "--drop-unlabeled"]
sys.path.insert(0, str(Path(__file__).parent))
import walkforward as wf
rows, tert = [], []
for w in wf.WAVELETS:
    d = wf.build(w); d["lrv"] = np.log(d.RV20)
    z = lambda s: (s - s.mean()) / s.std()
    d["zSI"], d["zRV"] = z(d.SI), z(d.lrv)
    d["period"] = pd.qcut(np.arange(len(d)), 5, labels=False) + 1
    for name, s in [("ALL", d)] + [(f"P{k}", d[d.period == k]) for k in range(1, 6)]:
        for spec, X in [("RV20", ["zRV"]), ("SI", ["zSI"]), ("RV20+SI", ["zRV", "zSI"])]:
            m = sm.Logit(s.Y_Return5D, sm.add_constant(s[X])).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": 10})
            r = dict(wavelet=w, scope=name, spec=spec, n=len(s), events=int(s.Y_Return5D.sum()), llf=m.llf)
            for v in X: r[f"b_{v}"], r[f"se_{v}"], r[f"p_{v}"] = m.params[v], m.bse[v], m.pvalues[v]
            rows.append(r)
    d["rvT"] = pd.qcut(d.RV20, 3, labels=["low", "mid", "high"])
    for t, s in d.groupby("rvT", observed=True):
        hi, lo = s.SI.quantile(.75), s.SI.quantile(.25)
        a, b = s[s.SI >= hi].Y_Return5D, s[s.SI <= lo].Y_Return5D
        tert.append(dict(wavelet=w, rv_tercile=t, n=len(s), base=s.Y_Return5D.mean(), f_topSI=a.mean(), n_top=len(a), f_botSI=b.mean(), n_bot=len(b)))
R = pd.DataFrame(rows); R.to_csv(wf.ROOT / "validation/incremental_logit.csv", index=False)
T = pd.DataFrame(tert); T.to_csv(wf.ROOT / "validation/si_within_rv20_terciles.csv", index=False)
pd.set_option("display.width", 220)
x = R[R.spec == "RV20+SI"][["wavelet", "scope", "n", "events", "b_zRV", "p_zRV", "b_zSI", "se_zSI", "p_zSI"]]
print(x.round(4).to_string(index=False))
print((T.assign(**{c: (T[c]*100).round(1) for c in ["base","f_topSI","f_botSI"]})).to_string(index=False))
# LR test of SI beyond RV20 (ALL)
for w in wf.WAVELETS:
    a = R[(R.wavelet==w)&(R.scope=="ALL")].set_index("spec").llf
    print(w, "LR stat SI|RV20 =", round(2*(a["RV20+SI"]-a["RV20"]),2))
