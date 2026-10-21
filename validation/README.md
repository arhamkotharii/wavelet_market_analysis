# SI robustness & validation (branch `si-robustness-validation`)

All scripts run from the repo root and are deterministic (fixed seeds). Primary target: `Future_Return_5D < -0.03`; `Future_MDD_5D < -0.03` is secondary.

| Script | Purpose | Output |
|---|---|---|
| `si_robustness.py` | Period-by-period top-vs-bottom SI quartile stress frequency (causal expanding cutoffs and within-period), block-bootstrap CIs, ex-2020 check | `si_robustness_results.csv` |
| `walkforward.py` | Rebuild of the handover walk-forward (RV20 / SI / RV20+SI). **Reproduces all handover ROC-AUC and PR-AUC values exactly** (1,152 OOS rows). `--drop-unlabeled` removes the last 5 rows with undefined labels | `walkforward_*.csv` |
| `bootstrap_sensitivity.py` | Paired moving-block bootstrap of metric differences, block lengths 5/10/20/40 | `bootstrap_sensitivity*.csv` |
| `incremental_test.py` | Logit with HAC SEs: does SI add to log RV20? Per-period coefficients; SI effect within RV20 terciles | `incremental_logit.csv`, `si_within_rv20_terciles.csv` |

## Findings (preliminary; see caveats)
1. **Data note.** The last 5 valid rows have no 5D label; the original `si_predictive_analysis.py` and the walk-forward treat them as non-stress. Dropping them (2,873 rows) changes the OOS event rate 4.77% -> 5.04% and AUCs by <=0.015; conclusions unchanged.
2. **Pooled association is robust.** Top-minus-bottom SI-quartile stress frequency is about +6 to +7 pts (95% block-bootstrap CI lower bounds +2.5 to +3.1) for Haar, DB4, Coif1, and survives excluding 2020.
3. **Persistence is partial.** DB4/Coif1 are positive in all 5 periods under causal cutoffs; individual-period CIs mostly include 0. 2017-05 to 2019-09 shows no effect (slight reversal with within-period quartiles). Effect is smaller in 2024-26 (low-stress regime).
4. **Incremental to RV20 (in-sample, full sample).** SI coefficient controlling for log RV20: p = 0.017 (Haar), 0.004 (DB4), 0.005 (Coif1), HAC-10 SEs; LR stats 11.2/16.5/15.7. Per-period: positive in P1, P3, P4 (DB4/Coif1 mostly significant), ~0 in P2, insignificant in P5.
5. **Walk-forward gain is not established.** Return5D: observed ROC-AUC gain of RV20+SI over RV20 is +0.02 to +0.06; all bootstrap CIs include 0 and are essentially unchanged across block lengths 5-40 (the OOS set has only ~55 events, so power is low). PR-AUC gains are +0.03 to +0.04 with CIs only just touching 0 for DB4/Coif1. MDD5D: no gain (ROC-AUC difference negative).
6. **Where the signal lives.** Within RV20 terciles, the top-SI group has higher stress than the bottom-SI group in the low-RV and high-RV terciles; there is essentially no difference in the mid tercile. Low-RV & low-SI had 0 events in 240 days.

## Caveats
Overlapping labels -> few independent stress clusters; quartile/tercile choices were fixed a priori but wavelets, periods (5 equal blocks) and block lengths are researcher choices; in-sample incremental tests are not out-of-sample evidence; no multiple-testing correction across 3 wavelets x 2 targets.
