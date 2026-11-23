# Phase 2 Report — Earth Observation Imagery Preparation

**Status:** Completed

---

## 1. Implementations
* **Landsat Collection 2 Level-2 Pipeline (`src/greenplan/gee/landsat.py`):**
  * Configured multi-temporal sensor ingestion: Landsat 5 TM (2000), Landsat 7 ETM+ (2010), and Landsat 8 OLI (2020).
  * Implemented QA cloud/shadow masking algorithms extracting bits 1 to 5 from `QA_PIXEL` and saturation detection via `QA_RADSAT`.
  * Integrated Roy et al. (2016) slope-and-intercept cross-sensor reflectance harmonization coefficients between ETM+ and OLI.
* **Spectral Indices Engine (`src/greenplan/gee/indices.py`):**
  * Verified mathematical implementations of NDVI (Rouse 1974; Hu 2023), NDBI (Zha 2003), MNDWI (Xu 2006), and EBBI (As-Syakur 2012).
  * Implemented USGS Collection 2 Level-2 scaling formulas for surface reflectance ($0.0000275 \times DN - 0.2$) and thermal band surface temperature in Celsius ($0.00341802 \times DN + 149.0 - 273.15$).

## 2. Testing & Verification
* `tests/test_indices.py`: Verified numerical scaling against USGS calibration specs and confirmed non-circularity safeguards.

