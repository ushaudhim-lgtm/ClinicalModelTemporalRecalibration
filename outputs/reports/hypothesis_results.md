# Hypothesis Results

Student: Usha Oudeacoumar (20057009, 20057009@mydbs.ie)

Quick-reference summary of each project hypothesis and what we found so far.
Source of truth is `notebooks/12_hypothesis_results.ipynb` and the tables in
`outputs/tables/`. This file should be updated whenever that notebook changes.

Last updated: after notebook 07 (temporal drift assessment).

---

## H1 - Temporal calibration drift

**Claim:** a model trained on historical data (2011-2014) will show measurable
calibration deterioration on later years (2015-2019), if frozen and left unchanged.

**Verdict: SUPPORTED**

| | internal val | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|
| auroc (log reg) | 0.821 | 0.814 | 0.809 | 0.798 | 0.793 | 0.798 |
| auroc (gb) | 0.830 | 0.822 | 0.817 | 0.807 | 0.799 | 0.805 |
| brier (log reg) | 0.092 | 0.080 | 0.079 | 0.090 | 0.100 | 0.094 |
| brier (gb) | 0.090 | 0.078 | 0.078 | 0.088 | 0.098 | 0.092 |
| cal_intercept (log reg, ideal 0) | 0.018 | -0.284 | -0.323 | -0.214 | -0.091 | -0.183 |
| cal_intercept (gb, ideal 0) | 0.106 | -0.181 | -0.230 | -0.087 | -0.004 | -0.081 |
| cal_slope (log reg, ideal 1) | 1.007 | 0.984 | 0.993 | 0.945 | 0.955 | 0.979 |
| cal_slope (gb, ideal 1) | 1.066 | 1.045 | 1.047 | 1.022 | 1.004 | 1.041 |
| ece (log reg) | 0.009 | 0.022 | 0.026 | 0.013 | 0.009 | 0.015 |
| ece (gb) | 0.010 | 0.021 | 0.025 | 0.011 | 0.003 | 0.013 |

What this means:
- The frozen model consistently **over-predicts** admission risk once applied
  to later years (calibration intercept is negative in almost every eval year,
  vs near-zero on internal validation).
- AUROC and Brier score both get steadily **worse from 2015 to 2018** for both
  models, then partly recover in 2019.
- ECE does not track Brier/AUROC exactly - 2018 is the best ECE year despite
  being the worst Brier/AUROC year. Calibration and discrimination drift are
  not the same thing and don't have to move together.
- Gradient boosting stays a bit closer to ideal calibration than logistic
  regression across most years, but both clearly drift.

Conclusion: yes, leaving a clinical prediction model frozen for several years
measurably hurts both its calibration and its discrimination. This is the
motivation for testing recalibration (H2) next.

---

## H2 - Recalibration effectiveness

**Claim:** retrospective recalibration (intercept-only, intercept+slope, Platt
scaling, isotonic regression) will improve calibration compared to the
unchanged frozen model.

**Status: PENDING** - needs `notebooks/08_apply_recalibration.ipynb`.

---

## H3 - Recalibration versus retraining

**Claim:** some recalibration approaches may get close to the calibration
performance of full model retraining, at lower computational cost.

**Status: PENDING** - needs `notebooks/08_apply_recalibration.ipynb` and
`notebooks/09_retrain_models.ipynb` (rolling 4-year retraining comparator).

---

## H4 - Temporal adaptation

**Claim:** more recent calibration observations may adapt better to changing
distributions than older calibration observations - this must be tested, not
assumed.

**Status: PENDING** - needs the two calibration protocols from
`notebooks/08_apply_recalibration.ipynb`: Protocol A (fixed historical
calibration set) vs Protocol B (rolling prior-year calibration window),
compared against each other.
