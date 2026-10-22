"""Runs the tests pre-registered in research/PREREG_tilt.md (commit c7b5d93). Outputs:
  research/tilt_tests_log.txt        human-readable log (all numbers quoted in the write-up)
  research/tilt_models.csv           every logit fitted on real data (sample, spec, target, wavelet, t-values)
  research/tilt_surrogates.csv       per-surrogate t-values of Z_tilt on S_SI and S_hold
Usage: python research/tilt_tests.py [n_surrogates=500]
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import spectral_pipeline as sp, tilt, surrogates as su
ROOT = Path(__file__).resolve().parent.parent
HOLD_END = pd.Timestamp("2015-01-30")           # first SI-valid date; S_hold is strictly before it
VOL = ["lrv5", "lrv20", "lrv60", "ret5"]


def frame(r, dates, wavelet="db4", with_si=False):
    s = pd.Series(r)
    d = pd.DataFrame({"Date": dates})
    for k in (5, 20, 60): d[f"lrv{k}"] = np.log(np.sqrt((s ** 2).rolling(k).mean()))
    d["ret5"] = s.rolling(5).sum()
    F = tilt.features(r, wavelet)
    for c in F: d[c] = F[c].values
    d["fut5"] = sum(s.shift(-i) for i in range(1, 6))
    d["Y"] = (d.fut5 < -0.03).astype(float); d.loc[d.fut5.isna(), "Y"] = np.nan
    if with_si: d["SI"] = sp.si_from_returns(r, wavelet)[:, 0]
    d["S_hold"] = d[VOL + ["Z_tilt", "Y"]].notna().all(1) & (d.Date < HOLD_END)
    d["S_SI"] = d[VOL + ["Z_tilt", "Y"]].notna().all(1) & (d.Date >= HOLD_END)
    d["S_all"] = d.S_hold | d.S_SI
    return d


def logit_t(d, y, cols):
    """HAC(10) logit on standardized regressors; returns (dict of t-values, llf, fitted model)."""
    X = d[cols]; X = (X - X.mean()) / X.std()
    m = sm.Logit(d[y], sm.add_constant(X)).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": 10})
    return m


def block_boot_gap(y, hi, lo, bl=20, reps=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); nb = int(np.ceil(n / bl)); g = []
    for _ in range(reps):
        i = ((rng.integers(0, n - bl + 1, nb)[:, None] + np.arange(bl)).ravel())[:n]
        if hi[i].sum() and lo[i].sum(): g.append(y[i][hi[i]].mean() - y[i][lo[i]].mean())
    return np.percentile(g, [2.5, 97.5])


def surrogate_t(r, dates):
    d = frame(r, dates)
    out = {}
    for S in ("S_SI", "S_hold"):
        try: out[S] = logit_t(d[d[S]], "Y", VOL + ["Z_tilt"]).tvalues["Z_tilt"]
        except Exception: out[S] = np.nan
    return out


if __name__ == "__main__":
    nsim = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    log = open(ROOT / "research/tilt_tests_log.txt", "w")
    def P(*a):
        print(*a); print(*a, file=log); log.flush()
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv", parse_dates=["Date"])
    oc = pd.read_csv(ROOT / "outcomes/nifty50_future_outcomes.csv")
    r = df.Log_Return.values; dates = df.Date.dt.normalize()
    rows = []

    real = {}
    for w in ["db4", "haar", "coif1"]:
        d = frame(r, dates, w, with_si=True)
        assert np.allclose(d.fut5.dropna(), oc.Future_Return_5D.iloc[:len(d)][d.fut5.notna()])
        d["Ymdd"] = (oc.Future_MDD_5D < -0.03).astype(float).where(d.Y.notna())
        real[w] = d
        specs = {"VOL": VOL, "VOL+Z_tilt": VOL + ["Z_tilt"], "VOL+tau": VOL + ["tau"],
                 "VOL+Z_rho1": VOL + ["Z_rho1"], "VOL+Z_VR5": VOL + ["Z_VR5"],
                 "VOL+Z_rho1+Z_tilt": VOL + ["Z_rho1", "Z_tilt"], "VOL+Z_VR5+Z_tilt": VOL + ["Z_VR5", "Z_tilt"],
                 "VOL+SI": VOL + ["SI"], "VOL+SI+Z_tilt": VOL + ["SI", "Z_tilt"]}
        for S in ("S_SI", "S_hold", "S_all"):
            dd = d[d[S]]
            for y in ("Y", "Ymdd"):
                base = logit_t(dd, y, VOL).llf
                for k, cols in specs.items():
                    if "SI" in cols and S != "S_SI": continue
                    m = logit_t(dd, y, cols)
                    rows.append(dict(wavelet=w, sample=S, target=y, spec=k, n=len(dd), events=int(dd[y].sum()),
                                     LR_vs_VOL=2 * (m.llf - base),
                                     **{f"t_{c}": m.tvalues[c] for c in cols if c not in VOL},
                                     **{f"b_{c}": m.params[c] for c in cols if c not in VOL}))
    M = pd.DataFrame(rows); M.to_csv(ROOT / "research/tilt_models.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = ["wavelet", "sample", "target", "spec", "n", "events", "LR_vs_VOL", "t_Z_tilt", "t_SI", "t_Z_rho1", "t_Z_VR5", "t_tau"]
    P("=== Real-data logits (HAC-10 t-values; standardized regressors) ===")
    P(M[show].round(2).to_string(index=False))

    d = real["db4"]
    P("\n=== Sample check ===")
    for S in ("S_hold", "S_SI", "S_all"):
        dd = d[d[S]]; P(f"{S}: n={len(dd)} events={int(dd.Y.sum())} {dd.Date.min().date()}..{dd.Date.max().date()}")
    P(f"corr(SI, Z_tilt) on S_SI = {d[d.S_SI][['SI','Z_tilt']].corr().iloc[0,1]:.3f}")

    # H2 descriptive: quartile gap with cut-points from S_SI
    q25, q75 = d.loc[d.S_SI, "Z_tilt"].quantile([.25, .75])
    P(f"\n=== H2 quartile gap (cut-points from S_SI: q25={q25:.3f}, q75={q75:.3f}) ===")
    for S in ("S_hold", "S_SI", "S_all"):
        dd = d[d[S]]; y = dd.Y.values; hi = (dd.Z_tilt >= q75).values; lo = (dd.Z_tilt <= q25).values
        ci = block_boot_gap(y, hi, lo)
        P(f"{S}: top {y[hi].mean():.4f} (n={hi.sum()}) bottom {y[lo].mean():.4f} (n={lo.sum()}) "
          f"gap {y[hi].mean()-y[lo].mean():+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    P("\n=== Per-year mean of Z_tilt (db4) and SI ===")
    P(d[d.S_all].groupby(d.Date.dt.year)[["Z_tilt", "SI"]].mean().round(3).T.to_string())

    # Surrogates
    t_real = {S: M.query("wavelet=='db4' and target=='Y' and spec=='VOL+Z_tilt' and sample==@S")["t_Z_tilt"].iloc[0]
              for S in ("S_SI", "S_hold")}
    rng = np.random.default_rng(2026); G = su.Garch(r); np.random.seed(2026); srows = []
    for kind, fn in [("signflip", su.signflip), ("garch", G)]:
        for i in range(nsim):
            srows.append(dict(kind=kind, sim=i + 1, **surrogate_t(fn(r, rng), dates)))
        P(f"{kind}: {nsim} done")
    Sdf = pd.DataFrame(srows); Sdf.to_csv(ROOT / "research/tilt_surrogates.csv", index=False)
    P("\n=== Surrogate p-values for t(Z_tilt), db4, target Y ===")
    for S in ("S_SI", "S_hold"):
        for kind, k in Sdf.groupby("kind"):
            v = k[S].dropna()
            P(f"{S:7s} {kind:9s} real t={t_real[S]:.3f}  null mean {v.mean():.3f} p95 {v.quantile(.95):.3f}  "
              f"P(t>1.96)={np.mean(v>1.96):.3f}  p={(1+(v>=t_real[S]).sum())/(1+len(v)):.4f}  (n={len(v)})")
    log.close()
