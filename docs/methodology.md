# Methodology & Method Traceability Matrix

## 1. Traceability to Peer-Reviewed Literature

| Step | Method | Research Question Answered | Primary Sources | Validation Approach |
|---|---|---|---|---|
| **1** | Multi-temporal Landsat (5/7/8/9) Collection 2 Level-2 surface reflectance; cloud/shadow masking via `QA_PIXEL` & `QA_RADSAT`; spectral indices (NDVI, NDBI, MNDWI, EBBI) | *How green, built-up, and wet is the study area across epochs?* | Patel et al. 2024; Arshad et al. 2025; Hu et al. 2023; As-Syakur et al. 2012 | Valid observation count maps, sensor cross-comparison on stable targets |
| **2** | Random Forest supervised classification per epoch (2000, 2010, 2020); post-classification transition analysis | *What is on the ground and how did green space convert over 20 years? (RQ1)* | Cheng et al. 2022; Patel et al. 2024; Hussain et al. 2023; Chaudhary et al. 2023 | Confusion matrix, Overall Accuracy (OA), Kappa, Per-class Producer/User Accuracies, Olofsson et al. (2014) area-adjusted estimates with 95% CI |
| **3** | Land Surface Temperature (LST) derivation from thermal bands (ST_B6/ST_B10); OLS and spatial error/lag regression on NDVI and built-up density | *How strongly is surface temperature associated with vegetation and built-up density? (RQ2)* | Mondal et al. 2021; Adnan et al. 2020; Feng et al. 2024; Kumar et al. 2019 | $R^2$, adjusted $R^2$, VIF, p-values, Moran's I residual spatial autocorrelation test, Getis-Ord Gi* hotspot maps |
| **4** | Cellular Automata Markov (CA-Markov) spatial simulation under BAU and Green-Protection scenarios | *Where will green space be lost next under baseline vs conservation trajectories? (RQ3)* | Talukdar et al. 2020; Hussain et al. 2025; Guan et al. 2011; Ozturk 2015; Bhattacharjee et al. 2025 | Calibration on 2000→2010 to simulate 2020; validation against actual 2020 map using Kappa-location, Kappa-quantity, Figure of Merit (FoM), and comparison against a no-change null model |
| **5** | Bivariate 3×3 priority matrix combining simulated loss risk and thermal stress into 500m planning units | *Which locations are both at high risk of loss and thermally stressed? (RQ4)* | Synthesis of Steps 2–4 | Threshold sensitivity analysis (terciles vs quartiles), sourced recommendation rulebook |

---

## 2. Mathematical Formulations

### 2.1 Spectral Indices
$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}$$

$$\text{NDBI} = \frac{\rho_{\text{SWIR1}} - \rho_{\text{NIR}}}{\rho_{\text{SWIR1}} + \rho_{\text{NIR}}}$$

$$\text{MNDWI} = \frac{\rho_{\text{Green}} - \rho_{\text{SWIR1}}}{\rho_{\text{Green}} + \rho_{\text{SWIR1}}}$$

$$\text{EBBI} = \frac{\rho_{\text{SWIR1}} - \rho_{\text{NIR}}}{10 \cdot \sqrt{\rho_{\text{SWIR1}} + \text{TIR}}}$$
*(Note: EBBI is used strictly as a classifier feature and excluded from LST regressions to eliminate circularity).*

### 2.2 Thermal Regression Model
$$\text{LST}_i = \beta_0 + \beta_1 \text{NDVI}_i + \beta_2 \text{BuiltUpFrac}_i + \varepsilon_i$$
Residual spatial autocorrelation evaluated with Moran's $I$:
$$I = \frac{N}{S_0} \frac{\sum_i \sum_j w_{ij}(e_i - \bar{e})(e_j - \bar{e})}{\sum_i (e_i - \bar{e})^2}$$

### 2.3 Markov Transition Matrix
$$P = \begin{bmatrix}
p_{11} & p_{12} & \dots & p_{1k} \\
p_{21} & p_{22} & \dots & p_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
p_{k1} & p_{k2} & \dots & p_{kk}
\end{bmatrix}, \quad \sum_{j=1}^k p_{ij} = 1$$
Scaled across time steps via matrix exponentiation:
$$P(t) = P^t$$

