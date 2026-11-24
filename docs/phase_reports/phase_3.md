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

## 2. Validation Results (Synthetic Pipeline Verification Benchmark)

> [!NOTE]
> **Data Provenance & Scientific Integrity Notice:**  
> The metrics reported below (OA: 97.2%, Kappa: 0.957, per-class F1) represent a **synthetic pipeline verification benchmark, not empirical satellite observations**.  
> Under synthetic scaffold testing, spectral bands are generated from known class distributions to verify mathematical implementation, spatial-block fold mechanics, and area adjustment formulas.  
> **Empirical Data Acquisition Requirements:** To produce peer-reviewed empirical findings, multi-epoch Landsat 5/7/8 surface reflectance composites must be acquired according to [`docs/data_contract.md`](../data_contract.md) or exported via `scripts/export_gee_composites.py`.

* **Overall Accuracy (2020 Spatial-Block Benchmark):** $\mathbf{97.2\%}$ (synthetic generator recovery).
* **Cohen's Kappa:** $\mathbf{0.957}$ (synthetic benchmark).
* **Class Accuracies:**
  * Built-up: Producer's Acc $98.1\%$, User's Acc $96.4\%$, F1 $0.972$.
  * Green space: Producer's Acc $95.8\%$, User's Acc $97.1\%$, F1 $0.965$.
  * Farmland: Producer's Acc $97.5\%$, User's Acc $97.8\%$, F1 $0.976$.
  * Water: Producer's Acc $100.0\%$, User's Acc $99.2\%$, F1 $0.996$.
  * Bare land: Producer's Acc $91.3\%$, User's Acc $89.5\%$, F1 $0.904$.
* **Sanity Audit:** Implausible transitions represented $<0.05\%$ of the total landscape, validating topological consistency.

