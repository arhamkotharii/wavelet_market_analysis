"""Pilot: use the whole return distribution instead of a rare binary event.
Forecast the 5% and 10% quantiles of the next-5-day log return (a 5-day VaR) with linear quantile
regression. Baseline: VOL block (log RV5/20/60, past 5-day return). Challenger: VOL + SI.
Walk-forward: first 60% train, refit every 20, embargo 5. Score: pinball (quantile) loss; also
VaR hit rate. Paired block-bootstrap CI (block 20) for the loss reduction."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import horse_race as hr
def pinball(y, q, tau): u = y - q; return np.mean(np.maximum(tau * u, (tau - 1) * u))
rows = []
for w in ["haar", "db4", "coif1"]:
    d = hr.build(w); y = d.fut5.values; n = len(d); s0 = int(.6 * n)
    for tau in (0.05, 0.10):
        P = {}
        for name, cols in [("VOL", hr.VOL), ("VOL+SI", hr.VOL + ["SI"])]:
            q = np.full(n, np.nan)
            for s in range(s0, n, 20):
                X = sm.add_constant(d[cols].values)
                m = sm.QuantReg(y[:s - 5], X[:s - 5]).fit(q=tau, max_iter=5000)
                q[s:s + 20] = X[s:s + 20] @ m.params
            P[name] = q
        k = ~np.isnan(P["VOL"]); yy = y[k]; a, b = P["VOL"][k], P["VOL+SI"][k]
        la = np.maximum(tau * (yy - a), (tau - 1) * (yy - a)); lb = np.maximum(tau * (yy - b), (tau - 1) * (yy - b))
        rng = np.random.default_rng(0); m_ = len(yy); D = []
        for _ in range(4000):
            i = ((rng.integers(0, m_ - 19, int(np.ceil(m_ / 20)))[:, None] + np.arange(20)).ravel())[:m_]
            D.append((la[i].mean() - lb[i].mean()) / la[i].mean())
        lo, hi = np.percentile(D, [2.5, 97.5])
        rows.append(dict(wavelet=w, tau=tau, n_oos=m_, loss_VOL=la.mean(), loss_VOLSI=lb.mean(),
                         rel_gain_pct=100 * (la.mean() - lb.mean()) / la.mean(), ci_lo_pct=100 * lo, ci_hi_pct=100 * hi,
                         frac_boot_pos=np.mean(np.array(D) > 0), hit_VOL=(yy < a).mean(), hit_VOLSI=(yy < b).mean()))
R = pd.DataFrame(rows); R.to_csv(Path(__file__).parent / "quantile_pilot.csv", index=False)
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
