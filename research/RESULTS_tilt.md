# Step 1 results: spectral tilt z-score and the pre-registered hold-out test (22 Oct 2026)

Pre-registration: `research/PREREG_tilt.md`, commit `c7b5d93`, pushed before any tilt-outcome number was
computed. Code: `research/tilt.py` (statistic), `research/tilt_tests.py` (pre-registered tests),
`research/tilt_diagnostics.py` (exploratory, written after seeing the results).
Numbers: `research/tilt_tests_log.txt`, `tilt_models.csv`, `tilt_surrogates.csv`, `tilt_diagnostics_log.txt`.

## 1. What the statistic is (established, verified numerically)

Parseval turns each rolling SWT band variance into a fixed weighted sum of circular autocovariances,
`V_j = sum_l A_j(l) c(l)` (max relative error 4e-15 against the existing pipeline), where `A_j` is the
autocorrelation of band j's filter. Taking logs of white-noise-normalised band powers with SI's signs and
linearising gives `tau ~= sum_l b_l rho(l)`. For DB4 the weights are
b = (-2.05, -1.79, -0.92, 0, +0.38, +0.36, ...) out to lag 21: the tilt contrasts lags 1-3 against lags 5-21.
The self-normalised z, `Z_tilt = sum x_t u_t / sqrt(sum x_t^2 u_t^2)` with `u = b * x` (one-sided), is
N(0,1) under a heteroskedastic martingale-difference null (simulated rejection rates 4.1-6.3% at nominal 5%). The exact log
form has a Jensen bias (+0.1 to +0.2 null mean) and is descriptive only.

Literature position: this is a member of the multiscale variance-ratio family of Gençay & Signori (2015,
J. Econometrics 184(1), 62-80), which tests white noise by the share of wavelet variance per scale under a
heteroskedastic white-noise null. Our version differs only in using SI's four-band contrast and an
MDS-robust self-normalisation. No novelty is claimed for the statistic.

## 2. Pre-registered results (DB4, target Y = next-5-day return < -3%, controls VOL)

| Test | Sample | Result | Rule | Outcome |
|---|---|---|---|---|
| H1 gate | 2015-01-30..2026-09 (2,873 d, 177 ev) | t(Z_tilt) = 2.03; surrogate p: sign-flip 0.040, GARCH 0.062. SI on the same rows: t = 3.14 | t >= 1.96 and both p < 0.05 | **Fails** (GARCH p > 0.05; Haar t = 1.44, Coif1 t = 1.76) |
| H2 hold-out (confirmatory) | 2011-01-10..2015-01-29 (1,000 d, 99 ev, ~29 clusters) | t(Z_tilt) = **-2.97**; one-sided surrogate p: sign-flip 0.998, GARCH 0.994 | beta > 0, p < 0.05 | **Not supported**; the estimate has the opposite sign |
| H2 secondary | same | Top Z quartile (cut-points fixed from 2015-26): 0/48 stress days; bottom quartile 16.4% (36/220). Gap -16.4 pp, block-bootstrap CI [-24.6, -8.9] | — | Opposite sign |
| Pooled | 2011-01..2026-09 (3,873 d, 276 ev) | t(Z_tilt) = 0.11 | — | No association |

Exploratory, two-sided view of H2: the hold-out t is in the lower 0.4% (sign-flip) and 0.8% (GARCH) of its
surrogate null distribution, so the reversal itself is unlikely to be noise. Robustness: Haar -1.87, Coif1
-2.71; 5-day max-drawdown target -3.31 (DB4). Surrogate null t-values have SD 1.14-1.28, confirming once
more that HAC t-values from overlapping labels are too optimistic.

H3 (does the wavelet weighting matter beyond conventional statistics?): on 2015-26, Z_tilt is significant
only alongside Z_rho1 or Z_VR5 (opposite-signed, collinear pairs: suppression, not interpretable). In the
hold-out, the robust 5-day variance ratio alone gives t = -2.46 and Z_tilt's t falls to -1.51 once it is
added. There is no evidence that the wavelet-shaped weights carry information a variance ratio lacks.

## 3. Exploratory diagnostics (after the fact; labelled)

* Not one year: excluding 2011 the hold-out t is -2.78; 2011 alone is -0.31.
* Not a drift artefact: demeaning each window changes Z_tilt little (corr 0.986; hold-out t -3.03).
* Not momentum: adding past-60 and past-250-day returns leaves t = -2.85 (hold-out), 1.98 (2015-26).
* Not the percentile step: Z relative to its trailing 250-day mean (an SI-like normalisation) also reverses
  in the hold-out (t = -2.51) and is +2.19 on 2015-26.
* Year-by-year sign of corr(Z_tilt, Y): 8 positive, 8 negative across 2011-2026 (univariate, noisy).
* No data artefacts in 2011-14 (no zero-return runs, no gaps over 6 calendar days).
* On 2015-26, SI keeps t = 2.24 with Z_tilt in the model while Z_tilt drops to 0.30: whatever made SI
  significant there is not the absolute spectral tilt. Given the hold-out, the most economical reading is
  that SI's 2015-26 association is partly a sample-specific fit, not a missing ingredient.

## 4. Verdict

The hypothesis "local anti-persistence precedes 5-day downside stress in the NIFTY 50" failed its
pre-registered out-of-sample test, and the hold-out shows a significant association of the opposite sign
(persistence preceded stress in 2011-14). Over 2011-2026 as a whole there is no association. The SI
association reported in the handover is real in its sample but is not a stable property of the index.

## 5. Decision log

* 22 Oct 2026: chose the linearised self-normalised z (not the exact log form) as primary because the log
  form is biased under the null. Decided before outcome linkage.
* 22 Oct 2026: gate failed and hold-out reversed. **Stop** the SI/tilt-as-stress-predictor line on NIFTY.
  No further NIFTY specifications (windows, wavelets, regimes, thresholds): each would be a fork on a
  sample whose sign is already known to flip.
* Open: whether to run the multi-market test of the sign question (needs user-downloaded data), or close
  the project as a NIFTY methods-and-refutation study. See the project handover addendum.
