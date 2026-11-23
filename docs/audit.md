# Repository Audit (Phase 0)

**Date:** 2026-11-19  
**Git Tag:** `legacy-unet-v1` (Commit: `9a34f0d`)  
**Status:** Audit Complete — Ready for Gate G0 / Scaffolding

---

## 1. Inventory of Legacy Repository

### 1.1 Directory & File Structure
```
EPICS23-554/
├── PRESENTATION/
│   └── Analyzing Urban Growth for Green Space Planning Using GIS.pdf (Slide presentation)
├── REPORT/
│   ├── EPICS23-554 PLAGIARISM DETECTION REPORT.pdf
│   └── EPICS23_554_23BCE11695_Phase2_Final Report.pdf (100k+ chars academic report)
└── PROJECT/
    └── gis/
        ├── app.py                         (Flask REST server + inline U-Net inference & heuristics)
        ├── explore.py                     (Dataset exploration & RGB mask check)
        ├── prepare_data.py                (Tile patch extraction to 256x256 image/mask pairs)
        ├── train.py                       (U-Net model training with weighted sparse CCE)
        ├── predict.py                     (Sliding-window patch evaluation & mIoU metric)
        ├── growth_prediction.py           (Spatial growth score calculation via EDT/density filter)
        ├── environmental_recommendations.py (Rule-based recommendation generator)
        ├── unet_model.keras               (Trained U-Net weights, 94.5 MB)
        ├── training_curves.png            (Training loss/accuracy plot)
        ├── static/
        │   └── index.html                 (Dark-mode single page web UI)
        ├── data/
        │   └── Semantic segmentation dataset/ (72 image/mask pairs across 8 tiles)
        │       └── classes.json           (Class color definitions)
        ├── patches/
        │   ├── images/                    (1,305 extracted 256x256 image patches)
        │   └── masks/                     (1,305 extracted 256x256 categorical mask arrays)
        └── predictions/                   (3 sample prediction comparison PNGs)
```

### 1.2 Dependencies & Runtime Environment
* Current Python environment: Python 3.12 (standard libraries, NumPy, SciPy, Matplotlib, Scikit-learn, Pillow, etc.).
* Missing from local environment: `tensorflow` (not installed in current Python 3.12, causing `app.py` and `train.py` to fail if executed directly), `earthengine-api`, `geopandas`, `rasterio`.
* Hard-coded paths detected across legacy scripts:
  * `C:\Users\Balaji\OneDrive\Desktop\gis\...` appears in `app.py`, `train.py`, `prepare_data.py`, `growth_prediction.py`, `predict.py`, `environmental_recommendations.py`.
  * These fail on any machine other than the original author's setup.

### 1.3 Model Weights & Dataset Status
* **Weights:** `unet_model.keras` (94,510,768 bytes, Keras v3 format).
* **Dataset:** 72 aerial images (Tiles 1–8) from the "Humans in the Loop / Dubai Aerial Imagery" dataset.
* **Patches:** 1,305 training pairs (256×256 pixels).

---

## 2. Legacy Component Mapping

| Component | Target Action | One-Line Reason |
| :--- | :--- | :--- |
| `app.py` & `static/index.html` | **Move to legacy** | No frontend in initial phases; legacy Flask code violates separation of concerns (Phase 9 defines clean schemas). |
| `train.py` & `unet_model.keras` | **Move to legacy** | Deep learning on single-snapshot drone imagery is not supported by the literature review; replaced by RF in Earth Engine. |
| `prepare_data.py` & `patches/` | **Move to legacy** | Patching 72 Dubai drone images is irrelevant to multi-temporal Landsat regional modeling. |
| `data/Semantic segmentation dataset/` | **Move to legacy** | Foreign drone dataset; project scope targets regional Landsat Collection 2 Level-2 imagery for Bhopal. |
| `predict.py` | **Move to legacy** | Sliding window U-Net evaluation replaced by standard multi-epoch confusion matrices and area-adjusted metrics. |
| `growth_prediction.py` | **Move to legacy** | Heuristic weighted combination replaced by peer-reviewed Cellular Automata Markov (CA-Markov) model. |
| `environmental_recommendations.py` | **Adapt** | Core concept kept, but unverified hardcoded thresholds (e.g., 30% green cover, 50m water buffer) replaced with sourced rule table in `configs/recommendations.yaml`. |
| `explore.py` | **Move to legacy** | Ad-hoc EDA script replaced by unit-tested preprocessing and compositing scripts. |
| `PRESENTATION/` & `REPORT/` | **Keep** | Retained for reference, prior literature review, and academic record. |

---

## 3. Discrepancy & Gap Analysis

1. **Study Area & Data Source Gap:**
   * *Legacy:* Single-date high-resolution drone imagery from Dubai.
   * *Rework:* Multi-decadal satellite imagery (Landsat 5 TM, Landsat 7 ETM+, Landsat 8 OLI/TIRS, Landsat 9) focused on Bhopal Metropolitan Area.
2. **Temporal Dynamic Gap:**
   * *Legacy:* Static spatial classification with zero time dimension.
   * *Rework:* Multi-epoch analysis across 2000, 2010, 2020 (plus optional 2025 out-of-sample check) enabling real LULC change detection and transition matrices.
3. **Classification Methodology Gap:**
   * *Legacy:* U-Net trained on 1,305 patches.
   * *Rework:* Random Forest classifier inside Google Earth Engine (using multi-spectral bands, spectral indices: NDVI, NDBI, MNDWI, EBBI), validated using spatial-block cross-validation and area-adjusted error matrices (Olofsson et al. 2014).
4. **Thermal / Heat Consequence Gap:**
   * *Legacy:* Non-existent. No temperature data or heat risk analysis.
   * *Rework:* Landsat Level-2 Surface Temperature (ST) extraction, hot-season compositing, OLS/spatial error regression against NDVI and built-up density, and spatial hotspot mapping (Getis-Ord Gi*).
5. **Urban Growth Simulation Gap:**
   * *Legacy:* Heuristic formula mixing distance-to-roads and building density with arbitrary weights `[0.25, 0.30, 0.20, 0.10, 0.10, 0.05]`.
   * *Rework:* CA-Markov simulation with transition probability matrices, suitability modeling, constraint masking, validation against held-out epochs, and comparison against a no-change null model.
6. **Planning & Decision Matrix Gap:**
   * *Legacy:* Hardcoded static advice.
   * *Rework:* Multi-criteria 3×3 matrix integrating loss risk and thermal stress into planning units (500m grid/wards), backed by a sourced policy rulebook.

---

## 4. Open Decisions (D1–D8) Resolutions

As authorized by the owner, default resolutions from Section 3 of the Rework Plan are adopted:
* **D1 (Study Area):** Bhopal urban area plus surrounding suburban buffer (GeoJSON polygon in `configs/aoi/bhopal.geojson`, center approx 23.25°N, 77.41°E).
* **D2 (Epochs):** 2000, 2010, 2020 (with 2025 as secondary out-of-sample check where available).
* **D3 (Class Scheme):** 5 regional classes:
  1. Green Space (forests, parks, persistent tree canopy)
  2. Built-up (impervious structures, residential, commercial, roads)
  3. Water (Upper Lake, Lower Lake, reservoirs, rivers)
  4. Farmland (cultivated agricultural fields)
  5. Bare Land (open soil, unpaved tracts, cleared land)
* **D4 (Compositing Windows):** Dry pre-monsoon / early summer window (March–May) to separate perennial green canopy from seasonal agricultural crops.
* **D5 (LST Window):** Peak hot-season window matching Landsat thermal passes.
* **D6 (Projection Horizons):** 2030 and 2040 simulation horizons under Business-As-Usual (BAU) and Green-Protection scenarios.
* **D7 (Projected Coordinate System):** WGS 84 / UTM Zone 43N (`EPSG:32643`).
* **D8 (Planning Aggregation Units):** 500m regular planning grid cells covering the AOI.

