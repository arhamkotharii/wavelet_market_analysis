"""Fast, exact re-implementation of the SWT scale-variance + causal percentile + SI pipeline.

Key identity (periodic SWT on an N-sample window, no decimation):
    D_j = x (*) h_j   (circular convolution with the level-j equivalent filter)
    V_j = mean(D_j^2) = (1/N^2) * sum_k |H_j(k)|^2 |X(k)|^2      (Parseval)
So every scale variance is a fixed weighted sum of the window periodogram, and SI is a
rank-based function of four spectral band powers. This file computes them via FFT, which
is exact and ~1000x faster than looping pywt.swt, enabling surrogate/simulation tests.
"""
import numpy as np, pandas as pd, pywt
from numpy.lib.stride_tricks import sliding_window_view

WINDOW, LEVELS, PCT_WINDOW = 256, 3, 1000
_cache = {}

def band_weights(wavelet, n=WINDOW, levels=LEVELS):
    """|H_j(k)|^2 for D1..D3 and A3, obtained by pushing a unit impulse through pywt.swt."""
    key = (wavelet, n, levels)
    if key in _cache: return _cache[key]
    d = np.zeros(n); d[0] = 1.0
    c = pywt.swt(d, wavelet, level=levels, trim_approx=False, norm=False)
    resp = [c[2][1], c[1][1], c[0][1], c[0][0]]          # D1, D2, D3, A3
    W = np.array([np.abs(np.fft.fft(r)) ** 2 for r in resp])  # shape 4 x n
    _cache[key] = W
    return W

def scale_variances(returns, wavelet):
    """Rolling V_D1..V_A3 for windows ending at each index >= WINDOW-1 (same as main.py)."""
    r = np.asarray(returns, float)
    X = np.abs(np.fft.fft(sliding_window_view(r, WINDOW), axis=1)) ** 2
    return X @ band_weights(wavelet).T / WINDOW ** 2      # shape (len(r)-WINDOW+1) x 4

def causal_percentiles(V, w=PCT_WINDOW):
    """Q_t = 100 * #(history < V_t)/w over the previous w values (strict <, as percentiles.py)."""
    n = len(V); Q = np.full(V.shape, np.nan)
    for j in range(V.shape[1]):
        H = sliding_window_view(V[:-1, j], w)               # rows: history ending at t-1
        Q[w:, j] = 100.0 * (H < V[w:, j][:, None]).mean(axis=1)
    return Q

def si_from_returns(returns, wavelet):
    """Returns array aligned with `returns` (NaN where unavailable): SI, and the 4 Q's."""
    V = scale_variances(returns, wavelet)
    Q = causal_percentiles(V)
    out = np.full((len(returns), 5), np.nan)
    out[WINDOW - 1:, 1:] = Q
    out[:, 0] = (out[:, 1] + out[:, 2] - out[:, 3] - out[:, 4]) / 2
    return out
