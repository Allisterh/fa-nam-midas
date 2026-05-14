# FA-NAM-MIDAS

Code accompanying the MSc thesis *Neural Additive and Factor-Augmented
Extensions of MIDAS Regression for High-Dimensional Corporate Survival
Forecasting* (Copenhagen Business School).

The thesis builds on the sparse-group LASSO MIDAS survival framework of
Miao et al. (2025) and develops a family of neural extensions: an
additive Neural Additive Model (NAM-MIDAS) family with Ablation, Full,
and Residual variants, a factor-augmented variant (FA-NAM-MIDAS), a
non-additive baseline (Deep-MIDAS), and a factor-only ablation
(FA-MIDAS). All variants are estimated under a shared IPCW-weighted
logistic protocol and evaluated by time-dependent AUC at three forecast
horizons on two panels: a Chinese manufacturing panel and a parallel US
panel.

## Data

The code is **not** accompanied by data. Both panels are built from
licensed sources and cannot be redistributed:

* **Chinese panel** — Special-Treatment (ST) distress events and
  quarterly financial ratios, following Miao et al. (2025).
* **US panel** — Compustat North America Fundamentals Quarterly and CRSP
  delisting codes, retrieved through Wharton Research Data Services
  (WRDS).

With WRDS access, `build_us_dataset.R` reproduces the US panel in a
format that is drop-in compatible with the training scripts.

## Pipeline

1. **Data construction** — `build_us_dataset.R` builds the US panel
   from WRDS; the Chinese panel follows the Miao et al. (2025)
   construction.
2. **Training** — `train_baseline.R` trains the sg-LASSO-MIDAS
   baseline and `train_nam_family.R` trains the NAM-MIDAS family on the
   full panel; `train_nam_holdout.R` retrains FA-NAM-MIDAS with the six
   case-study firms held out.
3. **Shape-stability analysis** — `procrustes_align.R` performs
   orthogonal Procrustes alignment of the per-covariate aggregations
   across restarts and produces the population-level shape figures and
   signal-to-noise diagnostics. It is panel-agnostic: set the input
   paths and worker-file pattern before sourcing.
4. **Case study** — `predict_case_study.R` produces out-of-sample
   distress probabilities for the six held-out firms.

## Requirements

* **R** with `torch`, `Survivalml`, `survival`, `midasml`, `ggplot2`,
  `dplyr`, `tidyr`, `patchwork`.
* **Python 3** with `matplotlib`, `numpy`, `pandas`.

## Citation

If you use this code, please cite the thesis. The underlying framework
is Miao, W., Beyhum, J., Striaukas, J., and Van Keilegom, I. (2025),
*High-Dimensional Censored MIDAS Logistic Regression for Corporate
Survival Forecasting*.
