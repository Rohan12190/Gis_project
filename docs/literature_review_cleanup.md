# Literature Review Audit & Pruning Log

**Context:** The Phase II project report contained ~60 bibliography entries, several of which incorporated artificial publisher index labels (e.g., "(IEEE/Nature index cross-reference)", "(Taylor & Francis cited classic)") or strayed into unrelated domains (pedestrian movement, real-estate valuation, construction logistics, clinical epidemiology). This document classifies all candidate literature into **Core**, **Background**, or **Exclude**.

---

## 1. Classification Framework
* **Core (18 papers):** Directly anchors a computational or methodological step in the GreenPlan pipeline (GEE compositing, spectral indices, Random Forest, LST regression, CA-Markov, spatial accuracy).
* **Background (14 papers):** Provides foundational domain context for urban sprawl, Urban Heat Island phenomena, and satellite remote sensing.
* **Exclude (28 papers):** Off-topic studies (e.g., traffic routing, structural building logistics, health surveys, participatory mapping) to be removed from the scientific paper.

---

## 2. Core Methodological References
1. **Patel et al. (2024)** — GEE workflow & cloud-based LULC modeling (*Core: Step 1 & 2*)
2. **Arshad et al. (2025)** — Multi-temporal vegetation & canopy tracking in GEE (*Core: Step 1*)
3. **Hu et al. (2023)** — NDVI phenology & urban green dynamics (*Core: Step 1*)
4. **As-Syakur et al. (2012)** — Formulation of Enhanced Built-Up and Bareness Index (EBBI) (*Core: Step 1 & 2*)
5. **Cheng et al. (2022)** — Machine learning classification of urban drivers (*Core: Step 2 & 4*)
6. **Hussain et al. (2023)** — LULC transformation & environmental impact (*Core: Step 2 & 3*)
7. **Chaudhary et al. (2023)** — Spatio-temporal urban growth & thermal environment (*Core: Step 2 & 3*)
8. **Mondal et al. (2021)** — Spatial regression of LST against urban landscape indices (*Core: Step 3*)
9. **Adnan et al. (2020)** — Quantitative association between urban density and surface temperature (*Core: Step 3*)
10. **Feng et al. (2024)** — Hot-season LST compositing and green cooling effects (*Core: Step 3*)
11. **Kumar et al. (2019)** — Green space degradation and thermal comfort in fast-growing cities (*Core: Step 3*)
12. **Talukdar et al. (2020)** — Machine learning & Markov transition probabilities (*Core: Step 4*)
13. **Hussain et al. (2025)** — CA-Markov predictive simulation under growth scenarios (*Core: Step 4*)
14. **Guan et al. (2011)** — Cellular automata transition neighborhood rules (*Core: Step 4*)
15. **Ozturk (2015)** — CA-Markov simulation and validation protocols (*Core: Step 4*)
16. **Bhattacharjee et al. (2025)** — Urban sprawl simulation in Indian metropolitan areas (*Core: Step 4 & 5*)
17. **Olofsson et al. (2014)** — Area-adjusted classification error matrices & 95% CI (*Core: Validation*)
18. **Pontius & Millones (2011)** — Quantity and allocation disagreement metrics (*Core: Validation*)

---

## 3. Background Literature
* **Anselin (1995)** — Local Indicators of Spatial Association (LISA)
* **Getis & Ord (1992)** — Spatial hotspot statistics ($G_i^*$)
* **Breiman (2001)** — Random Forest algorithm
* **Roy et al. (2016)** — Landsat 7 ETM+ to Landsat 8 OLI reflectance harmonization
* **Pontius et al. (2008)** — Figure of merit for land change models
* **Dietzel & Clarke (2005)** — Cellular automata calibration
* **Torrens (2006)** — Cellular automata spatial mechanics of urban sprawl
* **Voogt & Oke (2003)** — Thermal remote sensing of urban surface temperatures
* **Weng (2009)** — Thermal infrared remote sensing for urban climate studies
* **Seto et al. (2011)** — Meta-analysis of global urban land expansion
* **Pickett et al. (2011)** — Urban ecological frameworks
* **Grimm et al. (2008)** — Global consequences of land transformation
* **Gorelick et al. (2017)** — Google Earth Engine planetary-scale geospatial platform
* **Li et al. (2019)** — Deep learning remote sensing review (cited in future work)

---

## 4. Excluded Literature (Removed from Citations)
* *Reason 1: Pedestrian and Traffic Logistics:* Papers examining pedestrian sidewalk flow, road pavement roughness, and traffic micro-simulations (Entries 41–44).
* *Reason 2: Structural Construction & Materials:* Papers analyzing cement logistics, structural load testing, and BIM engineering (Entries 47–49).
* *Reason 3: Clinical & Public Health:* Papers focused on hospital patient intake spatial tracking (Entries 58–60).
* *Reason 4: Real Estate Valuation:* Papers focusing on hedonic pricing of urban housing without remote sensing linkages (Entry 50).

