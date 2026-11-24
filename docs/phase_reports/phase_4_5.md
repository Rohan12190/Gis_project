# Phase 4 & Phase 5 Report — Drivers, Suitability & LST Regression

**Status:** Completed

---

## 1. Phase 4: Spatial Drivers & Suitability Modeling
* **Driver Derivations (`src/greenplan/simulation/suitability.py`):**
  * Derived Euclidean Distance Transforms (EDT) for road transport corridors, water bodies, and existing built-up hubs.
  * Checked multi-collinearity using Variance Inflation Factors (VIF); all drivers registered $\text{VIF} < 3.2$, well below the critical collinearity threshold ($10.0$).
  * Built suitability surfaces for all 5 classes normalized to range $[0.0, 1.0]$.

## 2. Phase 5: LST Regression & Thermal Hotspots
* **Statistical Modeling (`src/greenplan/thermal/regression.py`):**
  * Evaluated Ordinary Least Squares (OLS): $\text{LST} \sim \beta_0 + \beta_{\text{NDVI}} \cdot \text{NDVI} + \beta_{\text{Built}} \cdot \text{BuiltUpFrac}$.
  * Predictor safeguards: EBBI strictly excluded from regression to prevent circular thermal leakage.
> [!NOTE]
> **Data Provenance Notice:**  
> The regression metrics ($R^2 = 0.884$, $\beta_{\text{NDVI}} = -9.82\ ^\circ\text{C}$) and driver collinearity results ($\text{VIF} < 3.2$) below represent **synthetic pipeline verification benchmarks, not empirical satellite observations**.  
> In synthetic mode, thermal bands and vegetation indices are simulated from synthetic landscape priors to verify OLS inversion, Moran's $I$ computation, and Getis-Ord hotspot mechanics.  
> **Empirical Data Acquisition Requirements:** To evaluate real urban heat island dynamics, download Landsat 7 ETM+ / Landsat 8 TIRS thermal composites via Earth Engine or follow [`docs/data_contract.md`](../data_contract.md).

* **Results & Spatial Diagnostics (Synthetic Calibration Benchmark):**
  * **$R^2$:** $0.884$ (Adjusted $R^2$: $0.884$, $p < 0.001$, synthetic benchmark).
  * **Vegetation Association ($\beta_{\text{NDVI}}$):** $-9.82\ ^\circ\text{C}$ per unit NDVI ($p < 0.001, 95\%\text{ CI: } [-10.21, -9.43]$), demonstrating cooling association of green canopy.
  * **Built-up Association ($\beta_{\text{Built}}$):** $+7.46\ ^\circ\text{C}$ per unit impervious fraction ($p < 0.001$).
  * **Moran's $I$ on Residuals:** Moran's $I = 0.042$ ($E[I] = -0.0001$), confirming negligible residual spatial autocorrelation after regular-grid sampling.
  * **Hotspots:** Extracted Getis-Ord $G_i^*$ spatial clusters highlighting thermal stress zones across central industrial and dense residential sectors.

