# GreenPlan Real-Data Ingestion Contract

This document defines the schema and file specifications required for empirical Landsat satellite ingestion into GreenPlan.

## 1. Directory Structure

Each regional study area must be located under `data/satellite/{area_slug}/` (e.g., `data/satellite/bhopal/`):

```text
data/satellite/{area_slug}/
├── lulc_2000.npy               # Classified LULC raster for epoch 2000 (uint8)
├── lulc_2010.npy               # Classified LULC raster for epoch 2010 (uint8)
├── lulc_2020.npy               # Classified LULC raster for epoch 2020 (uint8)
├── reference_2020.npy          # (Optional) Remapped independent reference layer (e.g. ESA WorldCover) (int8)
├── accuracy_reference.json     # (Optional) Non-circular CV accuracy metrics against proxy reference
├── roads.npy                   # (Optional) Transport network / road centerlines mask (2D bool/uint8)
├── bands_2020.npz              # Surface reflectance & thermal band arrays for epoch 2020
├── meta.json                   # Spatial reference, spatial resolution, and bounding coordinates
└── quicklook.png               # 3-panel visual overview of 2000-2020 LULC transitions
```

## 2. Design Decision: Independent Reference Layer (Non-Circularity)

To prevent circular evaluation (training and validating a classifier on features derived from the exact same labels):
1. **Proxy Reference**: We ingest the independent **ESA WorldCover 2020** 10-meter global product (`ESA/WorldCover/v100`), remapped to GreenPlan's 5-class taxonomy (`configs/worldcover_mapping.yaml`).
2. **Honest Accuracy Reporting**: Random Forest cross-validation is evaluated strictly against the independent WorldCover proxy reference with 4-fold spatial-block CV.
3. **Labels**: Accuracy metrics and the dashboard state `"reference": "ESA WorldCover 2020 (proxy reference)"`, never claiming ground truth.

## 3. Array Specifications

### LULC Rasters (`lulc_{epoch}.npy`)
* **Format:** 2D NumPy array of shape `(H, W)`.
* **Data Type:** `uint8`.
* **Taxonomy (Class IDs):**
  * `0`: Built-up (Impervious surface, paved infrastructure)
  * `1`: Green space (Forest canopy, urban parks, perennial plantations)
  * `2`: Farmland (Agricultural cropland, fallow land)
  * `3`: Water (Lakes, rivers, wetlands)
  * `4`: Bare land (Exposed soil, cleared land)

### Multi-Spectral Bands (`bands_2020.npz`)
A compressed `.npz` archive containing 5 aligned 2D floating-point arrays of shape `(H, W)`:
* `nir`: Near-Infrared surface reflectance (0.0 to 1.0).
* `red`: Red surface reflectance (0.0 to 1.0).
* `swir1`: Shortwave-Infrared 1 surface reflectance (0.0 to 1.0).
* `green`: Green surface reflectance (0.0 to 1.0).
* `tir`: Thermal Infrared band. May be provided in Kelvin ($>200.0$) or Celsius.

### Metadata (`meta.json`)
```json
{
  "crs": "EPSG:32643",
  "pixel_size_m": 30.0,
  "bounds": {
    "min_lon": 77.25,
    "min_lat": 23.12,
    "max_lon": 77.58,
    "max_lat": 23.38
  }
}
```

## 4. Ingestion Behavior
* When all contract files exist in `data/satellite/{area_slug}/`, `greenplan` automatically ingests them and sets `data_source: "empirical_satellite_data"`.
* If files are missing or incomplete, the pipeline raises an explicit `FileNotFoundError` or `ValueError` stating the required files.
* Under the real-data mandate, all runs must use verified satellite observations, live Earth Engine ingestion, or verified offline snapshots.
