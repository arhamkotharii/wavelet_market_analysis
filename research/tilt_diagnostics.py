"""EXPLORATORY diagnostics, written AFTER seeing the pre-registered results (hold-out sign reversal).
Not confirmatory; each check targets one candidate explanation of the reversal:
  D1 per-year:   is the hold-out reversal one year (2011 bear market) or spread out?
  D2 drift:      autocorrelations are not demeaned, so a strong trend (mu^2 > 0) shows up as persistence.
                 Recompute Z_tilt on demeaned windows.
  D3 momentum:   add past-60 and past-250-day returns to VOL.
  D4 relative:   SI ranks band powers against their own past; does tilt RELATIVE to its trailing
                 250-day mean behave like SI (and does it also reverse in the hold-out)?
Output: research/tilt_diagnostics_log.txt
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view as swv
sys.path.insert(0, str(Path(__file__).parent))
import tilt, tilt_tests as tt
ROOT = Path(__file__).resolve().parent.parent
VOL = tt.VOL

def z_demeaned(r, wavelet="db4"):
    b = tilt.lag_weights(wavelet); L = len(b); X = swv(np.asarray(r, float), tilt.N)
    X = X - X.mean(1, keepdims=True)
    U = sum(b[l - 1] * X[:, L - l:tilt.N - l] for l in range(1, L + 1)); Xt = X[:, L:]
    out = np.full(len(r), np.nan); out[tilt.N - 1:] = (Xt * U).sum(1) / np.sqrt((Xt ** 2 * U ** 2).sum(1))
    return out

if __name__ == "__main__":
    log = open(ROOT / "research/tilt_diagnostics_log.txt", "w")
    def P(*a): print(*a); print(*a, file=log)
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv", parse_dates=["Date"])
    r = df.Log_Return.values; d = tt.frame(r, df.Date.dt.normalize(), with_si=True)
    s = pd.Series(r)
    d["ret60"] = s.rolling(60).sum(); d["ret250"] = s.rolling(250).sum()
    d["Z_dm"] = z_demeaned(r)
    d["Z_rel"] = d.Z_tilt - d.Z_tilt.rolling(250).mean().shift(1)
    d["mu2_over_var"] = (s.rolling(256).mean() ** 2) / (s ** 2).rolling(256).mean()
    def t(dd, cols, c):
        try: return round(tt.logit_t(dd, "Y", cols).tvalues[c], 2)
        except Exception as e: return np.nan

    P("D1: per-year, S_hold and S_SI (t of Z_tilt in Y ~ VOL + Z_tilt; few events per year -> noisy)")
    for y, dd in d[d.S_all].groupby(d.Date.dt.year):
        P(f"  {y}: n={len(dd)} events={int(dd.Y.sum())} mean Z={dd.Z_tilt.mean():+.2f} "
          f"corr(Z,Y)={dd[['Z_tilt','Y']].corr().iloc[0,1]:+.3f}")
    P(f"  S_hold excluding 2011: t={t(d[d.S_hold & (d.Date.dt.year > 2011)], VOL+['Z_tilt'], 'Z_tilt')}")
    P(f"  S_hold 2011 only:      t={t(d[d.S_hold & (d.Date.dt.year == 2011)], VOL+['Z_tilt'], 'Z_tilt')}")

    P("\nD2: drift. mean of mu^2/E[x^2] in window: " +
      ", ".join(f"{S} {d.loc[d[S],'mu2_over_var'].mean():.4f}" for S in ("S_hold", "S_SI")))
    P(f"  corr(Z_tilt, Z_demeaned) = {d[['Z_tilt','Z_dm']].corr().iloc[0,1]:.3f}")
    for S in ("S_hold", "S_SI", "S_all"):
        P(f"  {S}: t(Z_tilt)={t(d[d[S]], VOL+['Z_tilt'], 'Z_tilt')}  t(Z_demeaned)={t(d[d[S]], VOL+['Z_dm'], 'Z_dm')}")

    P("\nD3: momentum controls (VOL + ret60 + ret250 + Z_tilt)")
    for S in ("S_hold", "S_SI", "S_all"):
        P(f"  {S}: t(Z_tilt)={t(d[d[S] & d.ret250.notna()], VOL+['ret60','ret250','Z_tilt'], 'Z_tilt')}")

    P("\nD4: relative tilt Z_rel = Z_tilt - trailing 250-day mean (rows where defined)")
    m = d.Z_rel.notna()
    P(f"  corr(SI, Z_rel) on S_SI = {d.loc[d.S_SI, ['SI','Z_rel']].corr().iloc[0,1]:.3f}  (corr(SI, Z_tilt) = {d.loc[d.S_SI, ['SI','Z_tilt']].corr().iloc[0,1]:.3f})")
    for S in ("S_hold", "S_SI", "S_all"):
        dd = d[d[S] & m]
        P(f"  {S} (n={len(dd)}, events={int(dd.Y.sum())}): t(Z_rel)={t(dd, VOL+['Z_rel'], 'Z_rel')}  "
          f"t(Z_tilt) same rows={t(dd, VOL+['Z_tilt'], 'Z_tilt')}")
    dd = d[d.S_SI & m]
    P(f"  S_SI joint VOL+SI+Z_rel: t(SI)={t(dd, VOL+['SI','Z_rel'], 'SI')} t(Z_rel)={t(dd, VOL+['SI','Z_rel'], 'Z_rel')}")
    log.close()
