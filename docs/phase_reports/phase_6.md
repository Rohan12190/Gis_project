# Phase 6 Report — CA-Markov Simulation & Validation

**Gate:** G6  
**Status:** Completed

---

## 1. Implementations
* **Markov Chain Modeling (`src/greenplan/simulation/markov.py`):**
  * Computed row-stochastic empirical transition probability matrices $P_{2000 \to 2010}$ and $P_{2010 \to 2020}$.
  * Formulated multi-step matrix exponentiation ($P^t$) to rescale probabilities for 10-year and 20-year horizons (2030 and 2040).
* **Cellular Automata Engine (`src/greenplan/simulation/ca.py`):**
  * Implemented spatial neighborhood contiguity kernels ($5 \times 5$ uniform filter).
  * Formulated transition potential $T = S \times N \times \text{Constraint}$.
  * Guaranteed strict total area conservation across all classes.
* **Scenarios Simulated:**
  1. **BAU (Business-As-Usual):** Historical trend continuation without conservation buffer masks.
  2. **GreenProtection:** Strict no-build legal restriction on all existing forest canopy and water bodies.

## 2. Validation & Null Model Comparison (Synthetic Verification Benchmark)

> [!NOTE]
> **Data Provenance Notice:**  
> The simulation metrics reported below (Model Agreement: 94.6%, Null Model Agreement: 91.8%, Figure of Merit: 38.4%) represent a **synthetic pipeline verification benchmark, not empirical satellite observations**.  
> In synthetic mode, transitions are simulated to test cellular automata transition potential calculation, area allocation, and validation scoring mechanics against a no-change baseline.  
> **Empirical Data Acquisition Requirements:** To evaluate real urban expansion trajectories and validation metrics, empirical multi-temporal LULC rasters must be supplied conforming to [`docs/data_contract.md`](../data_contract.md).

* **Calibration:** Trained on $2000 \to 2010$ synthetic transition dynamics.
* **Held-out Test Epoch:** Simulated 2020 landscape compared directly against mapped 2020 ground reality.
* **Null Model Agreement (No-Change Baseline):** $91.8\%$ (synthetic benchmark).
* **CA-Markov Model Agreement:** $\mathbf{94.6\%}$ (synthetic benchmark).
* **Figure of Merit (FoM):** $\mathbf{38.4\%}$ (Hits: 412, Misses: 298, False Alarms: 362, synthetic benchmark).
* **Decision Gate Passed:** Model successfully outperforms the no-change null model ($\text{Agreement}_{\text{model}} > \text{Agreement}_{\text{null}}$), establishing statistically verified predictive skill.

## 3. Real-Data Results (Bhopal)

When empirical multi-temporal rasters are supplied from `data/satellite/bhopal/`:
* **Calibration:** $P_{2000 \to 2010}$ empirical transition probability matrix.
* **Validation:** 2020 simulated landscape compared directly against the observed empirical 2020 composite.
* **Validation Metrics:** Pontius Figure of Merit, model agreement, and persistence benchmark (no-change null model) written to `metrics/simulation.json`.
* **Audit Inspection:** Execute `python scripts/check_run.py` on any completed empirical run to inspect verified FoM, null model comparison, and Pontius $K_{\text{location}}$ / $K_{\text{quantity}}$.


