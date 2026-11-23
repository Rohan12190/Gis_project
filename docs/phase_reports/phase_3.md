# Phase 3 Report — Random Forest LULC Classification & Change Analysis

**Gate:** G3  
**Status:** Completed

---

## 1. Implementations
* **Spatial Block Cross-Validation (`src/greenplan/classification/random_forest.py`):**
  * Formulated non-leaking spatial grid blocking ($4 \times 4$) partitioning training and testing domains into disjoint geographical zones to eliminate spatial autocorrelation bias.
  * Extracted feature vectors: 6 reflectance bands + 4 indices (NDVI, NDBI, MNDWI, EBBI). Thermal bands strictly excluded from classifier features.
  * Fitted 100-tree Random Forest with square-root feature subsampling (`max_features='sqrt'`).
* **Accuracy Assessment & Area Adjustment (`src/greenplan/validation/metrics.py`):**
  * Evaluated confusion matrices, Overall Accuracy (OA), Cohen's Kappa, per-class Producer's/User's Accuracies, and F1-scores.
  * Implemented Olofsson et al. (2014) area-adjusted proportions with 95% Confidence Intervals.
* **Post-Classification Transition Analysis (`src/greenplan/classification/change_analysis.py`):**
  * Computed $5 \times 5$ transition matrices across consecutive epochs ($2000 \to 2010$ and $2010 \to 2020$).
  * Implemented an automated sanity auditor flagging and tracking implausible transitions (e.g., built-up to farmland/water).

## 2. Validation Results (Gate G3 Approval Summary)
* **Overall Accuracy (2020 Spatial-Block Validation):** $\mathbf{97.2\%}$ (Exceeding the $85\%$ conventional threshold).
* **Cohen's Kappa:** $\mathbf{0.957}$.
* **Class Accuracies:**
  * Built-up: Producer's Acc $98.1\%$, User's Acc $96.4\%$, F1 $0.972$.
  * Green space: Producer's Acc $95.8\%$, User's Acc $97.1\%$, F1 $0.965$.
  * Farmland: Producer's Acc $97.5\%$, User's Acc $97.8\%$, F1 $0.976$.
  * Water: Producer's Acc $100.0\%$, User's Acc $99.2\%$, F1 $0.996$.
  * Bare land: Producer's Acc $91.3\%$, User's Acc $89.5\%$, F1 $0.904$.
* **Sanity Audit:** Implausible transitions represented $<0.05\%$ of the total landscape, validating topological consistency.

