"""Which ingredient of SI carries information about 5-day stress (Future_Return_5D < -3%)?
Feature blocks (all causal, computed on data up to and including day t):
  VOL  : log RV5, log RV20, log RV60 (HAR-style multi-horizon volatility) + 5-day return (momentum/leverage)
  ACF  : circular autocorrelations rho1..rho7 of the same 256-day window SI uses
  SI   : DB4 scale imbalance;  SIr: residual of SI after projecting on ACF (expanding, causal)
Tests: (1) full-sample logit with HAC(10) SEs, LR tests; (2) walk-forward (60% initial, refit 20, embargo 5)
with paired block bootstrap of ROC-AUC / log-loss differences vs VOL."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).parent)); import spectral_pipeline as sp
ROOT = Path(__file__).resolve().parent.parent
N = sp.WINDOW

def build(wavelet="db4"):
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv", parse_dates=["Date"])
    r = df.Log_Return.values; s = pd.Series(r)
    d = pd.DataFrame({"Date": df.Date.dt.normalize()})
    for k in (5, 20, 60): d[f"lrv{k}"] = np.log(np.sqrt((s ** 2).rolling(k).mean()))
    d["ret5"] = s.rolling(5).sum()
    X = swv(r, N); den = (X ** 2).sum(1)
    for l in range(1, 8):
        a = np.full(len(r), np.nan); a[N - 1:] = (X * np.roll(X, l, axis=1)).sum(1) / den; d[f"rho{l}"] = a
    d["SI"] = sp.si_from_returns(r, wavelet)[:, 0]
    d["fut5"] = sum(s.shift(-i) for i in range(1, 6))
    d = d.dropna().reset_index(drop=True)
    d["Y"] = (d.fut5 < -0.03).astype(int)
    # causal residual of SI on ACF: expanding OLS fit using data up to t-1 (min 250 obs)
    A = d[[f"rho{l}" for l in range(1, 8)]].values; res = np.full(len(d), np.nan)
    for t in range(250, len(d)):
        b = np.linalg.lstsq(np.c_[np.ones(t), A[:t]], d.SI.values[:t], rcond=None)[0]
        res[t] = d.SI.values[t] - np.r_[1, A[t]] @ b
    d["SIr"] = res
    return d.dropna().reset_index(drop=True)

VOL = ["lrv5", "lrv20", "lrv60", "ret5"]; ACF = [f"rho{l}" for l in range(1, 8)]
SPECS = {"VOL": VOL, "VOL+SI": VOL + ["SI"], "VOL+ACF": VOL + ACF, "VOL+ACF+SI": VOL + ACF + ["SI"],
         "VOL+SIr": VOL + ["SIr"]}

def insample(d):
    z = (d - d.mean()) / d.std(); out = {}
    for k, cols in SPECS.items():
        m = sm.Logit(d.Y, sm.add_constant(z[cols])).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": 10})
        out[k] = m
    return out

def walk(d, cols, start_frac=.6, step=20, emb=5):
    n = len(d); s0 = int(start_frac * n); p = np.full(n, np.nan)
    for s in range(s0, n, step):
        m = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=1000, C=1.0))
        m.fit(d[cols].values[:s - emb], d.Y.values[:s - emb])
        p[s:s + step] = m.predict_proba(d[cols].values[s:s + step])[:, 1]
    return p

def boot_diff(y, pa, pb, bl=20, reps=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); nb = int(np.ceil(n / bl)); D = []
    for _ in range(reps):
        i = ((rng.integers(0, n - bl + 1, nb)[:, None] + np.arange(bl)).ravel())[:n]
        if y[i].sum() == 0: continue
        D.append([roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]),
                  log_loss(y[i], pb[i], labels=[0, 1]) - log_loss(y[i], pa[i], labels=[0, 1])])
    return np.percentile(np.array(D), [2.5, 97.5], axis=0)

if __name__ == "__main__":
    rows = []
    for w in ["haar", "db4", "coif1"]:
        d = build(w)
        M = insample(d)
        base = M["VOL"].llf
        print(f"\n=== {w}: n={len(d)}, events={d.Y.sum()}, {d.Date.min().date()}..{d.Date.max().date()}")
        for k, m in M.items():
            extra = {c: (round(m.tvalues[c], 2)) for c in ["SI", "SIr"] if c in m.params}
            print(f"  {k:12s} LR vs VOL={2*(m.llf-base):6.2f}  df={len(m.params)-len(M['VOL'].params)}  t: {extra}")
        P = {k: walk(d, c) for k, c in SPECS.items()}
        msk = ~np.isnan(P["VOL"]); y = d.Y.values[msk]
        for k in SPECS:
            pa = P[k][msk]; auc = roc_auc_score(y, pa); ll = log_loss(y, pa)
            ci = boot_diff(y, pa, P["VOL"][msk]) if k != "VOL" else np.full((2, 2), np.nan)
            rows.append(dict(wavelet=w, model=k, n_oos=msk.sum(), events=int(y.sum()), AUC=auc, LogLoss=ll,
                             dAUC=auc - roc_auc_score(y, P["VOL"][msk]), dAUC_lo=ci[0, 0], dAUC_hi=ci[1, 0],
                             dLL_gain=log_loss(y, P["VOL"][msk]) - ll, dLL_lo=ci[0, 1], dLL_hi=ci[1, 1]))
    R = pd.DataFrame(rows); R.to_csv(ROOT / "research/horse_race_walkforward.csv", index=False)
    pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
