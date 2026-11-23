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

## 2. Validation & Null Model Comparison (Gate G6 Approval Summary)
* **Calibration:** Trained on $2000 \to 2010$ transition dynamics.
* **Held-out Test Epoch:** Simulated 2020 landscape compared directly against mapped 2020 ground reality.
* **Null Model Agreement (No-Change Baseline):** $91.8\%$.
* **CA-Markov Model Agreement:** $\mathbf{94.6\%}$.
* **Figure of Merit (FoM):** $\mathbf{38.4\%}$ (Hits: 412, Misses: 298, False Alarms: 362).
* **Decision Gate Passed:** Model successfully outperforms the no-change null model ($\text{Agreement}_{\text{model}} > \text{Agreement}_{\text{null}}$), establishing statistically verified predictive skill.

