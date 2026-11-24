# GreenPlan Real-Data Ingestion Contract

This document defines the schema and file specifications required for empirical Landsat satellite ingestion into GreenPlan.

## 1. Directory Structure

Each regional study area must be located under `data/satellite/{city_slug}/` (e.g., `data/satellite/bhopal/`):

```text
data/satellite/{city_slug}/
├── lulc_2000.npy       # Ground truth / classified reference LULC raster for epoch 2000
├── lulc_2010.npy       # Ground truth / classified reference LULC raster for epoch 2010
├── lulc_2020.npy       # Ground truth / classified reference LULC raster for epoch 2020
├── bands_2020.npz      # Surface reflectance & thermal band arrays for epoch 2020
└── meta.json           # Spatial reference, spatial resolution, and bounding coordinates
```

## 2. Array Specifications

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

## 3. Ingestion Behavior
* When all contract files exist in `data/satellite/{city}/`, `greenplan` automatically ingests them and sets `data_source: "empirical_satellite_data"`.
* If files are partially missing, the pipeline raises an explicit `ValueError` identifying the missing files.
* If no local files exist, running with `--synthetic` executes synthetic landscape simulation tagged with `data_source: "synthetic_scaffold_testing"`.
