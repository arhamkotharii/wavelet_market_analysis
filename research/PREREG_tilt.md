# Pre-registration: spectral tilt z-score on NIFTY 50 (step 1 of the handover plan)

Frozen 22 Oct 2026, **before** any tilt-outcome relationship was computed. Committed to git before
`research/tilt_tests.py` was run; the commit hash is the timestamp. Anything not listed here is exploratory
and will be labelled as such.

Already known when this was written (outcome-only or feature-only facts, no feature-outcome link):
hold-out sample size and base rate (below); null calibration of the statistic on simulated data
(`research/tilt.py`); feature-only summaries on NIFTY (Z_tilt mean -0.43, sd 0.87; corr with Z_VR5
0.90, with Z_rho1 0.47). Everything in the handover about SI on 2015-2026 was also known.

## 1. Statistic (defined in `research/tilt.py`)

For the 256-day window ending at day t (returns x, oldest first) and band filters D1, D2, D3, A3 of the
existing 3-level SWT, let `A_j(l)` be the autocorrelation sequence of band j's filter and
`b_l = A_D1(l)/A_D1(0) + A_D2(l)/A_D2(0) - A_D3(l)/A_D3(0) - A_A3(l)/A_A3(0)`, l = 1..L.
This is the first-order (delta-method) expansion of SI's own sign pattern applied to log white-noise-
normalised band powers: `tau = (l_D1 + l_D2 - l_D3 - l_A3)/2 ~= sum_l b_l rho(l)`.
With `u_t = sum_{l=1}^{L} b_l x_{t-l}` (inside the window):

    Z_tilt = sum_t x_t u_t / sqrt( sum_t x_t^2 u_t^2 )

Under a martingale-difference null with arbitrary heteroskedasticity, Z_tilt is approximately N(0,1)
(simulation, DB4, non-overlapping windows: iid t5 mean 0.04 sd 1.01 size 5.1%; GJR-GARCH-t mean 0.11 sd 1.00
size 5.3%). The exact log form `tau` has a Jensen bias (mean +0.13 to +0.21 under the null), so it is
secondary/descriptive only. Positive Z_tilt = anti-persistence (choppiness).

* Primary wavelet: **DB4** (L = 21). Haar (L = 3) and Coif1 (L = 15) are robustness only.
* No percentiles, no 1,000-day burn-in: valid from 2011-01-10.

Comparators (same window, same self-normalised construction, sign chosen so positive = anti-persistent):
* `Z_rho1`: weights b = (-1) (lag-1 autocorrelation only).
* `Z_VR5`: weights b_l = -2(1 - l/5), l = 1..4 (Lo-MacKinlay robust variance ratio, q = 5).

## 2. Data, target, controls (unchanged from the handover)

* `data/nifty50_processed.csv`; Y_t = 1(sum of next 5 log returns < -0.03). Secondary: 5-day MDD < -3%.
* VOL = log RV5, log RV20, log RV60, past 5-day return (the horse-race baseline).
* Logit with HAC(10) standard errors on standardized regressors.

## 3. Samples

* **S_hold** (confirmatory): labelled days 2011-01-10 to 2015-01-29. 1,000 days, 99 events (9.9%),
  about 29 event clusters. No feature-outcome analysis in this project has used these dates (every earlier
  feature-outcome script read the percentile files, valid only from 2015-01-30).
* **S_SI**: the 2,873 labelled SI-valid days (2015-01-30 onward). Already explored; used for the gate only.
* **S_all**: the union, 2011-01-10 to 2026-09.

## 4. Hypotheses, tests, decision rules

**H1 gate (S_SI, not confirmatory).** Does the percentile-free statistic keep SI's link?
Report t of Z_tilt in Y ~ VOL + Z_tilt next to t of SI in Y ~ VOL + SI on identical rows, plus surrogate
p-values for Z_tilt (500 sign-flip, 500 GJR-GARCH-t surrogates of the full series, seed 2026; statistic
= HAC t of Z_tilt; p = (1 + #{t_sur >= t_real})/(1 + 500)).
*Rule:* the gate passes if t(Z_tilt) >= 1.96 **and** both surrogate p < 0.05. If it fails, the percentile
step was doing the work: report this and stop the tilt line (go to the surrogate-methods chapter).

**H2 hold-out (S_hold, confirmatory, one-sided).** beta(Z_tilt) > 0 in Y ~ VOL + Z_tilt.
*Primary p-value:* sign-flip surrogate (keeps the exact volatility path; the most conservative null in
the handover), same 500 surrogates, t recomputed on S_hold rows only. Reject at 0.05.
Secondary: GARCH surrogate p; HAC t; top-minus-bottom quartile stress gap of Z_tilt (quartile cut-points
from S_SI, so they are fixed before looking) with a block-20 bootstrap 95% CI.
Power note: if the SI-era effect transfers, expected t is about 3.1 x sqrt(99/177) ~ 2.3, so power is
roughly 0.7. A non-rejection is weak evidence against the effect, not a refutation; report the CI.

**H3 is the DSP weighting needed? (exploratory on S_SI and S_all; reported on S_hold).**
Fit Y ~ VOL + Z_rho1 + Z_tilt and Y ~ VOL + Z_VR5 + Z_tilt. If Z_tilt's t falls below 1.96 once a
one-number conventional statistic is added, the wavelet weighting adds nothing beyond that statistic and
the write-up must say so.

**Robustness (reported, not used for decisions):** Haar and Coif1; secondary MDD target; the exact log
form tau; per-year mean of Z_tilt (does the level drift, which SI's percentile step would have removed?).

## 5. Explicitly not done in this round
No new thresholds, horizons, windows, wavelets, models, or walk-forward AUC tuning on NIFTY.
Multi-market testing (step 2 of the plan) needs data the user must download; it will be pre-registered
separately with this statistic frozen as defined here.
