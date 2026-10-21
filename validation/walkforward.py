"""Rebuild of the expanding-window walk-forward from the project handover (Sec. 13).
RV20 only / SI only / RV20+SI ; standardized logistic regression (liblinear, max_iter=500);
first 60% initial train, refit every 20 obs, 5-obs embargo.
Run from repo root: python validation/walkforward.py [--drop-unlabeled]
--drop-unlabeled removes the last 5 rows whose 5D label is undefined (original code treats them as 0)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
DROP = "--drop-unlabeled" in sys.argv
EMBARGO, STEP, TRAIN_FRAC = 5, 20, 0.60
WAVELETS = ["haar", "db4", "coif1"]
TARGETS = {"Return5D": ("Future_Return_5D", "ret"), "MDD5D": ("Future_MDD_5D", "mdd")}

out = pd.read_csv(ROOT / "outcomes/nifty50_future_outcomes.csv", parse_dates=["Date"])
out["Date"] = out.Date.dt.normalize()
out["RV20"] = np.sqrt((out.Log_Return ** 2).rolling(20).mean())
out["Y_Return5D"] = (out.Future_Return_5D < -0.03).astype(int)
out["Y_MDD5D"] = (out.Future_MDD_5D < -0.03).astype(int)

def build(w):
    p = pd.read_csv(ROOT / f"percentiles/{w}_swt_percentiles.csv", parse_dates=["Date"])
    p["Date"] = p.Date.dt.normalize()
    p["SI"] = (p.Q_D1 + p.Q_D2 - p.Q_D3 - p.Q_A3) / 2
    d = p.dropna(subset=["SI"]).merge(out, on="Date", how="inner").dropna(subset=["RV20"])
    if DROP: d = d.dropna(subset=["Future_Return_5D"])
    return d.reset_index(drop=True)

FEATS = {"RV20_only": ["RV20"], "SI_only": ["SI"], "RV20_plus_SI": ["RV20", "SI"]}

def walk(d, ycol):
    n = len(d); start = int(TRAIN_FRAC * n); preds = {k: [] for k in FEATS}; idx = []
    for s in range(start, n, STEP):
        e = min(s + STEP, n); tr = slice(0, s - EMBARGO)
        ytr = d[ycol].values[tr]
        for k, cols in FEATS.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=500))
            m.fit(d[cols].values[tr], ytr)
            preds[k].append(m.predict_proba(d[cols].values[s:e])[:, 1])
        idx.extend(range(s, e))
    res = d.iloc[idx][["Date", ycol]].rename(columns={ycol: "y"}).reset_index(drop=True)
    for k in FEATS: res[k] = np.concatenate(preds[k])
    return res

def metrics(y, p):
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                Brier=brier_score_loss(y, p), LogLoss=log_loss(y, p, labels=[0, 1]), MeanP=p.mean())

if __name__ == "__main__":
    allp, rows = [], []
    for w in WAVELETS:
        d = build(w)
        for tname, (_, _) in TARGETS.items():
            r = walk(d, f"Y_{tname}")
            r.insert(0, "target", tname); r.insert(0, "wavelet", w); allp.append(r)
            for k in FEATS:
                rows.append(dict(wavelet=w, target=tname, model=k, n=len(r), event_rate=r.y.mean(), **metrics(r.y.values, r[k].values)))
    sfx = "_dropunlabeled" if DROP else ""
    pd.concat(allp).to_csv(ROOT / f"validation/walkforward_predictions{sfx}.csv", index=False)
    M = pd.DataFrame(rows); M.to_csv(ROOT / f"validation/walkforward_model_comparison{sfx}.csv", index=False)
    pd.set_option("display.width", 200)
    print(M.round(4).to_string(index=False))
