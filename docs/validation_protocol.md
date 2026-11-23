# Validation Protocol & Quality Gates

**Project:** EPICS23-554 — GreenPlan

---

## 1. Summary of Required Validation Outputs

| Component | Minimum Required Metrics | Quality Standards / Rules |
|---|---|---|
| **Random Forest LULC** | Confusion matrix, Overall Accuracy (OA), Cohen's Kappa, Producer's & User's Accuracies, F1-scores, Area-adjusted class proportions with 95% Confidence Intervals (Olofsson et al. 2014) | Spatial-block cross-validation (no pixel leakage). Conventional benchmark: Overall Accuracy $\ge 85\%$. Any class below benchmark must be documented with diagnostic explanation rather than suppressed. |
| **Change Analysis** | Transition matrices ($k \times k$), Gross/Net change per class, flagged implausible transitions (e.g. Built-up $\to$ Water/Forest) | Counts and spatial masks of implausible transitions explicitly reported in change audit tables. |
| **LST Regression** | Ordinary Least Squares (OLS) coefficients, Standard Errors, 95% CI, $t$-statistics, $p$-values, $R^2$, Adjusted $R^2$, Variance Inflation Factors (VIF), Moran's $I$ of residuals, Spatial Error/Lag Model comparison | Non-circular predictors (no thermal indices). Spatial autocorrelation diagnosed. Results framed strictly as *associations*, never causal claims. |
| **CA-Markov Simulation** | Overall Agreement, Kappa, Kappa-location ($K_{\text{loc}}$), Kappa-quantity ($K_{\text{quant}}$), Pontius Figure of Merit (FoM), Quantity vs Allocation disagreement | Must compare against a **no-change null model**. A simulation that fails to outperform the null model is rejected as having no predictive skill. |
| **Scenario Modeling** | Projected loss area (ha and $\text{km}^2$), seed-to-seed variance across multiple stochastic runs | Framed explicitly as policy scenarios ("what-if"), never as definitive prophecies. |
| **Bivariate Planning Map** | Sensitivity analysis of priority acreage across tercile vs quartile thresholds | Assesses robustness of high-priority conservation zoning to classification cutoffs. |

---

## 2. Null Model Comparison Specification
In urban change modeling, if 92% of the landscape remains unchanged between 2010 and 2020, a naive model that predicts zero change achieves 92% raw accuracy while being useless. GreenPlan implements the Pontius & Millones (2011) and Pontius et al. (2008) protocol:
1. **Null Model Baseline:** Map at $t_1$ projected forward as $t_2$ without change.
2. **Figure of Merit (FoM):**
$$\text{FoM} = \frac{B}{A + B + C + D}$$
Where:
* $A$: Observed change predicted as persistence (misses)
* $B$: Observed change correctly predicted as change (hits)
* $C$: Observed persistence predicted as change (false alarms)
* $D$: Observed change predicted as change to wrong category
3. A valid CA-Markov model must demonstrate $\text{FoM}_{\text{model}} > \text{FoM}_{\text{null}}$ and superior allocation agreement.

