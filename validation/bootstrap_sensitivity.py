"""Paired moving-block bootstrap of metric differences, with block-length sensitivity.
Delta = metric(model_A) - metric(RV20_only) on identical OOS rows. Uses validation/walkforward_predictions*.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
ROOT = Path(__file__).resolve().parent.parent
sfx = "_dropunlabeled" if "--drop-unlabeled" in sys.argv else ""
P = pd.read_csv(ROOT / f"validation/walkforward_predictions{sfx}.csv", parse_dates=["Date"])
REPS, BLOCKS, rng = 1000, [5, 10, 20, 40], np.random.default_rng(7)

def auc(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(p); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def mets(y, p):
    if y.sum() == 0: return [np.nan] * 4
    return [auc(y, p), average_precision_score(y, p), brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])]
NAMES = ["ROC_AUC", "PR_AUC", "Brier", "LogLoss"]

rows = []
for (w, t), g in P.groupby(["wavelet", "target"], sort=False):
    g = g.sort_values("Date").reset_index(drop=True); y = g.y.values; n = len(g)
    base = np.array(mets(y, g.RV20_only.values))
    for model in ["RV20_plus_SI", "SI_only"]:
        obs = np.array(mets(y, g[model].values)) - base
        for bl in BLOCKS:
            nb = int(np.ceil(n / bl)); D = []
            for _ in range(REPS):
                idx = ((rng.integers(0, n - bl + 1, nb)[:, None] + np.arange(bl)).ravel())[:n]
                yy = y[idx]
                if yy.sum() == 0: continue
                D.append(np.array(mets(yy, g[model].values[idx])) - np.array(mets(yy, g.RV20_only.values[idx])))
            D = np.array(D)
            for j, m in enumerate(NAMES):
                lo, hi = np.nanpercentile(D[:, j], [2.5, 97.5])
                rows.append(dict(wavelet=w, target=t, model=model, block=bl, metric=m, observed_diff=obs[j],
                                 lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0),
                                 frac_pos=float((D[:, j] > 0).mean())))
R = pd.DataFrame(rows); R.to_csv(ROOT / f"validation/bootstrap_sensitivity{sfx}.csv", index=False)
pd.set_option("display.width", 200)
print(R[(R.model == "RV20_plus_SI") & (R.metric.isin(["ROC_AUC", "PR_AUC"]))].round(3).to_string(index=False))
