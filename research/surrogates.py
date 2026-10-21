"""Surrogate-data tests: which property of the return series produces the SI -> future-stress link?

Each surrogate keeps some properties of NIFTY returns and destroys others:
  signflip : r_t * s_t with random signs. Keeps |r_t| exactly (whole volatility path),
             destroys all sign structure (autocorrelation, leverage, return predictability).
  garch    : GJR-GARCH(1,1)-t fitted to NIFTY, simulated. Volatility clustering + leverage,
             zero autocorrelation in returns, no other structure.
  iaaft    : Iterated amplitude-adjusted Fourier transform (Schreiber & Schmitz 1996). Keeps
             the marginal distribution and the global power spectrum (linear autocorrelation),
             destroys volatility clustering and all nonlinear dependence.
For each series the full pipeline (rolling DB4 SWT -> causal percentiles -> SI) and the
stress label (sum of next 5 log returns < -3%) are recomputed from that series alone.
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import spectral_pipeline as sp
ROOT = Path(__file__).resolve().parent.parent
WAVELET, H, THR = "db4", 5, -0.03

def load_returns():
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv", parse_dates=["Date"])
    return df.Log_Return.values, df.Date.dt.normalize().values

def stats(r, wavelet=WAVELET):
    """Statistics of the SI-stress link computed from one return series."""
    s = pd.Series(r)
    fut = sum(s.shift(-i) for i in range(1, H + 1))
    rv = lambda k: np.log(np.sqrt((s ** 2).rolling(k).mean()))
    d = pd.DataFrame({"SI": sp.si_from_returns(r, wavelet)[:, 0], "fut": fut,
                      "l5": rv(5), "l20": rv(20), "l60": rv(60)}).dropna()
    d["Y"] = (d.fut < THR).astype(int)
    q25, q75 = d.SI.quantile([.25, .75])
    gap = d.Y[d.SI >= q75].mean() - d.Y[d.SI <= q25].mean()
    z = lambda c: (d[c] - d[c].mean()) / d[c].std()
    out = {"gap": gap, "rate": d.Y.mean(), "corr_SI_l20": d.SI.corr(d.l20)}
    for name, X in [("t_SI_rv20", ["l20"]), ("t_SI_har", ["l5", "l20", "l60"])]:
        Z = sm.add_constant(pd.concat([z(c) for c in X + ["SI"]], axis=1))
        try:
            m = sm.Logit(d.Y, Z).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": 10})
            out[name] = m.tvalues["SI"]
        except Exception:
            out[name] = np.nan
    return out

def signflip(r, rng): return r * rng.choice([-1.0, 1.0], len(r))

def iaaft(r, rng, iters=200):
    amp = np.abs(np.fft.rfft(r)); srt = np.sort(r); y = rng.permutation(r)
    for _ in range(iters):
        Y = np.fft.rfft(y); y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(r))
        y = srt[np.argsort(np.argsort(y))]
    return y

class Garch:
    def __init__(self, r):
        from arch import arch_model
        self.res = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        self.n = len(r)
    def __call__(self, r, rng):
        from arch import arch_model
        p = self.res.params
        m = arch_model(None, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        m.distribution._random_state = None
        sim = m.simulate(p, self.n, burn=500, initial_value=None)
        return sim["data"].values / 100

if __name__ == "__main__":
    nsim = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    r, _ = load_returns(); rng = np.random.default_rng(2026)
    real = stats(r); print("REAL", {k: round(v, 4) for k, v in real.items()})
    g = Garch(r); print(g.res.params.round(4).to_dict())
    rows = [dict(kind="real", sim=0, **real)]
    for kind, fn in [("signflip", signflip), ("garch", g), ("iaaft", iaaft)]:
        if kind == "garch": np.random.seed(2026)
        for i in range(nsim):
            rows.append(dict(kind=kind, sim=i + 1, **stats(fn(r, rng))))
        k = pd.DataFrame([x for x in rows if x["kind"] == kind])
        print(f"{kind}: done {nsim}")
    R = pd.DataFrame(rows); R.to_csv(ROOT / "research/surrogate_results.csv", index=False)
    summ = []
    for kind, k in R[R.kind != "real"].groupby("kind"):
        row = {"kind": kind, "n": len(k)}
        for c in ["gap", "t_SI_rv20", "t_SI_har"]:
            row[f"{c}_mean"] = k[c].mean(); row[f"{c}_p95"] = k[c].quantile(.95)
            row[f"p_{c}"] = (1 + (k[c] >= real[c]).sum()) / (1 + k[c].notna().sum())
        row["frac_t_har_gt_1.96"] = (k.t_SI_har > 1.96).mean()
        row["corr_SI_l20_mean"] = k.corr_SI_l20.mean(); row["rate_mean"] = k.rate.mean()
        summ.append(row)
    S = pd.DataFrame(summ); S.to_csv(ROOT / "research/surrogate_summary.csv", index=False)
    pd.set_option("display.width", 250); print(S.round(4).T)
