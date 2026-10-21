"""Period-by-period SI vs 5-day negative-return stress (Future_Return_5D < -0.03).
Run from repo root:  python si_robustness.py
Quartile cutoffs: (A) causal expanding (past-only, min 250 obs) -> prospective rule
                  (B) within-block quartiles -> descriptive only
CIs: moving-block bootstrap (block=20, 2000 reps) inside each period."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(42)
WAVELETS = ["haar", "db4", "coif1"]
NBLK, BLOCK, REPS, MINPAST = 5, 20, 2000, 250

out = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "outcomes/nifty50_future_outcomes.csv"), parse_dates=["Date"])
out["Date"] = out["Date"].dt.normalize()
out["Y"] = (out["Future_Return_5D"] < -0.03).astype(int)

def load(w):
    p = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parent.parent / f"percentiles/{w}_swt_percentiles.csv"), parse_dates=["Date"])
    p["Date"] = p["Date"].dt.normalize()
    p["SI"] = (p.Q_D1 + p.Q_D2 - p.Q_D3 - p.Q_A3) / 2
    d = p.merge(out[["Date", "Y", "Future_Return_5D"]], on="Date").dropna(subset=["SI", "Future_Return_5D"]).reset_index(drop=True)
    # causal expanding cutoffs: use only SI values strictly before t
    q25 = d.SI.expanding(MINPAST).quantile(0.25).shift(1)
    q75 = d.SI.expanding(MINPAST).quantile(0.75).shift(1)
    d["grp_causal"] = np.where(d.SI >= q75, "Top", np.where(d.SI <= q25, "Bot", "Mid"))
    d.loc[q25.isna(), "grp_causal"] = "NA"
    return d

def stats(y, g):
    t, b = y[g == "Top"], y[g == "Bot"]
    ft = t.mean() if len(t) else np.nan; fb = b.mean() if len(b) else np.nan
    return len(t), int(t.sum()), ft, len(b), int(b.sum()), fb

def boot(y, g, reps=REPS):
    n = len(y); nb = int(np.ceil(n / BLOCK)); diffs = []
    for _ in range(reps):
        st = rng.integers(0, n - BLOCK + 1, nb)
        idx = (st[:, None] + np.arange(BLOCK)).ravel()[:n]
        yy, gg = y[idx], g[idx]
        t, b = yy[gg == "Top"], yy[gg == "Bot"]
        if len(t) and len(b): diffs.append(t.mean() - b.mean())
    return np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)

rows = []
for w in WAVELETS:
    d = load(w)
    d["period"] = pd.qcut(np.arange(len(d)), NBLK, labels=False)
    scopes = [("ALL", d)] + [(f"P{k+1}", d[d.period == k]) for k in range(NBLK)] + \
             [("ALL_ex2020", d[d.Date.dt.year != 2020])]
    for name, s in scopes:
        s = s.reset_index(drop=True)
        for rule in ["causal", "within"]:
            if rule == "causal":
                g = s.grp_causal.values
            else:
                lo, hi = s.SI.quantile([.25, .75])
                g = np.where(s.SI >= hi, "Top", np.where(s.SI <= lo, "Bot", "Mid"))
            y = s.Y.values
            nt, et, ft, nbm, eb, fb = stats(y, g)
            lo_ci, hi_ci = boot(y, g) if (nt and nbm) else (np.nan, np.nan)
            rows.append(dict(wavelet=w, scope=name, rule=rule,
                start=s.Date.min().date(), end=s.Date.max().date(), n=len(s),
                base_rate=y.mean(), n_top=nt, ev_top=et, f_top=ft, n_bot=nbm, ev_bot=eb, f_bot=fb,
                RR_top_vs_bot=(ft / fb if fb and fb > 0 else np.nan), diff=ft - fb,
                diff_lo=lo_ci, diff_hi=hi_ci))
res = pd.DataFrame(rows)
res.to_csv(str(__import__("pathlib").Path(__file__).resolve().parent / "si_robustness_results.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
fmt = res.copy()
for c in ["base_rate", "f_top", "f_bot", "diff", "diff_lo", "diff_hi"]: fmt[c] = (fmt[c] * 100).round(2)
fmt["RR_top_vs_bot"] = fmt.RR_top_vs_bot.round(2)
for w in WAVELETS:
    for rule in ["causal", "within"]:
        print(f"\n=== {w.upper()} | {rule} quartiles | diff = Top - Bottom stress freq (pct pts) ===")
        print(fmt[(fmt.wavelet == w) & (fmt.rule == rule)].drop(columns=["wavelet", "rule"]).to_string(index=False))
