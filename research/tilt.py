"""Spectral tilt statistic: the percentile-free, known-null analogue of Scale Imbalance (SI).

Notation (one rolling window ending at day t, N = 256 returns x_0..x_{N-1}, oldest first):
  c(l)   = (1/N) sum_n x_n x_{(n+l) mod N}            circular autocovariance (not demeaned, as in the SWT)
  rho(l) = c(l)/c(0)
  W_j(k) = |H_j(k)|^2                                 squared gain of band j in {D1, D2, D3, A3}
  A_j(l) = (1/N) sum_k W_j(k) e^{-2 pi i k l/N}       autocorrelation sequence of the band-j filter
Parseval gives V_j = sum_l A_j(l) c(l) = c(0) A_j(0) [1 + sum_{l != 0} Atil_j(l) rho(l)],  Atil = A/A(0).

Under white noise E V_j = c(0) A_j(0), so  ell_j = log( V_j / (A_j(0) c(0)) )  is 0 in expectation.
Tilt, built with exactly SI's signs but logs instead of percentile ranks:
    tau = (ell_D1 + ell_D2 - ell_D3 - ell_A3) / 2        (scale-free: c(0) cancels)
First-order (delta method) expansion about rho = 0:
    tau ~= sum_{l>=1} b_l rho(l),   b_l = sum_j s_j Atil_j(l),   s = (+1, +1, -1, -1)
i.e. tau is a fixed weighted sum of autocorrelations: a wavelet-shaped variance-ratio statistic.
Null: x_t a martingale difference sequence (MDS) with arbitrary conditional heteroskedasticity.
With u_t = sum_{l=1}^{L} b_l x_{t-l} (one-sided, inside the window), the linear statistic is
sum_t x_t u_t / sum_t x_t^2, and its MDS-robust standard error is sqrt(sum_t x_t^2 u_t^2) / sum_t x_t^2
(Lo-MacKinlay / Lobato-Nankervis-Savin type). The tilt z-score (primary statistic) is
    Z_tilt = tau_lin / se = sum_t x_t u_t / sqrt(sum_t x_t^2 u_t^2).
Z_tilt ~ N(0,1) approximately under the MDS null (simulation: research/PREREG_tilt.md). The exact log form
tau/se ("Z_exact") carries a Jensen bias of about +0.1 to +0.2 under the null and is descriptive only.
Positive tau/Z = short scales carry more power than white noise implies = anti-persistence (choppiness).
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view as swv
sys.path.insert(0, str(Path(__file__).parent)); import spectral_pipeline as sp
ROOT = Path(__file__).resolve().parent.parent
N = sp.WINDOW
SIGNS = np.array([1.0, 1.0, -1.0, -1.0])


def filter_acf(wavelet):
    """A_j(l), shape 4 x N (real, symmetric in l)."""
    return np.real(np.fft.ifft(sp.band_weights(wavelet), axis=1))


def lag_weights(wavelet, tol=1e-12):
    """b_l for l = 1..L (L = last lag with a non-negligible weight)."""
    A = filter_acf(wavelet); At = A / A[:, :1]
    b = SIGNS @ At                      # length N, b[0] = 0
    nz = np.nonzero(np.abs(b[1:N // 2]) > tol)[0]
    L = nz.max() + 1
    return b[1:L + 1]


def tilt(returns, wavelet):
    """Aligned with `returns` (NaN before N-1). Columns: tau, tau_lin, se, Z_exact (= tau/se)."""
    r = np.asarray(returns, float)
    X = swv(r, N)                                            # windows, oldest first
    P = np.abs(np.fft.fft(X, axis=1)) ** 2
    V = P @ sp.band_weights(wavelet).T / N ** 2              # identical to sp.scale_variances
    c0 = (X ** 2).mean(1)
    A0 = filter_acf(wavelet)[:, 0]
    ell = np.log(V / (A0[None, :] * c0[:, None]))
    tau = 0.5 * ell @ SIGNS
    b = lag_weights(wavelet); L = len(b)
    # u_t = sum_l b_l x_{t-l}, one-sided inside each window (t = L..N-1 only, so u uses N-window data)
    U = np.zeros((X.shape[0], N - L))
    for l in range(1, L + 1):
        U += b[l - 1] * X[:, L - l:N - l]
    Xt = X[:, L:]
    ss = (X ** 2).sum(1)
    tau_lin = (Xt * U).sum(1) / ss
    se = np.sqrt((Xt ** 2 * U ** 2).sum(1)) / ss
    out = np.full((len(r), 4), np.nan)
    out[N - 1:] = np.c_[tau, tau_lin, se, tau / se]
    return pd.DataFrame(out, columns=["tau", "tau_lin", "se", "Z_exact"])


def check_null(nsim=200, n=4133, seed=7):
    """Null calibration WITHOUT touching outcomes: Z on iid-t and GJR-GARCH-t (MDS, zero autocorrelation)."""
    from arch import arch_model
    rng = np.random.default_rng(seed); res = {}
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv"); r = df.Log_Return.values
    fit = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    sims = {"iid_t5": lambda: rng.standard_t(5, n) * 0.01,
            "gjr_garch_t": lambda: arch_model(None, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
                           .simulate(fit.params.drop("mu"), n, burn=500)["data"].values / 100}
    np.random.seed(seed)
    for name, f in sims.items():
        Zs = []
        for _ in range(nsim):
            T_ = tilt(f(), "db4"); z = (T_.tau_lin / T_.se).values[N - 1::N]           # non-overlapping windows -> independent draws
            Zs.append(z[~np.isnan(z)])
        Zs = np.concatenate(Zs)
        res[name] = dict(n=len(Zs), mean=Zs.mean(), sd=Zs.std(), p_abs_gt_1_96=(np.abs(Zs) > 1.96).mean(),
                         p_gt_1_645=(Zs > 1.645).mean())
    return pd.DataFrame(res).T


def z_weights(returns, b):
    """Self-normalised MDS-robust z of sum_l b_l rho(l) over the same rolling N-window (aligned, NaN before N-1)."""
    r = np.asarray(returns, float); X = swv(r, N); b = np.asarray(b, float); L = len(b)
    U = np.zeros((X.shape[0], N - L))
    for l in range(1, L + 1):
        U += b[l - 1] * X[:, L - l:N - l]
    Xt = X[:, L:]
    out = np.full(len(r), np.nan)
    out[N - 1:] = (Xt * U).sum(1) / np.sqrt((Xt ** 2 * U ** 2).sum(1))
    return out


COMPARATORS = {"Z_rho1": [-1.0], "Z_VR5": [-2 * (1 - l / 5) for l in range(1, 5)]}


def features(returns, wavelet="db4"):
    """Z_tilt (primary, = tau_lin/se), tau (exact log form), and the two conventional comparators."""
    T = tilt(returns, wavelet)
    out = pd.DataFrame({"Z_tilt": T.tau_lin / T.se, "tau": T.tau})
    for k, b in COMPARATORS.items():
        out[k] = z_weights(returns, b)
    return out


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data/nifty50_processed.csv", parse_dates=["Date"]); r = df.Log_Return.values
    for w in ["haar", "db4", "coif1"]:
        b = lag_weights(w)
        T = tilt(r, w)
        # exactness checks vs the existing pipeline
        V_ref = sp.scale_variances(r, w)
        X = swv(r, N); A = filter_acf(w)
        c = np.real(np.fft.ifft(np.abs(np.fft.fft(X, axis=1)) ** 2, axis=1)) / N   # circular c(l)
        V_acf = c @ A.T
        print(f"{w}: L={len(b)}, b[:6]={np.round(b[:6], 3)}, max rel err V(acf form)={np.max(np.abs(V_acf / V_ref - 1)):.1e}, "
              f"corr(tau, tau_lin)={T.tau.corr(T.tau_lin):.4f}, valid from {df.Date[T.tau.first_valid_index()].date()}")
    print("\nNull calibration of Z (db4, non-overlapping windows):")
    print(check_null().round(4))
