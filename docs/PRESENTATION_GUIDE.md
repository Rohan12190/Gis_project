# GreenPlan GIS — Teacher Presentation & Viva Guide

This guide explains the entire **GreenPlan** project in simple, clear terms so you can confidently present how it works, where the data comes from, what algorithms it uses, and what every chart/map on the website means.

---

## 1. What Does GreenPlan Do? (The 30-Second Summary)
**GreenPlan** is a real-data **Urban Growth & Green Space Planning** system built on **Google Earth Engine (GEE)**.
* You click any location inside **Madhya Pradesh** (or load a saved offline city like **Bhopal** or **Kothri**).
* It pulls **real Landsat satellite images** from **2000 to 2025**, classifies what is on the ground (buildings, trees/green space, farmland, water, bare land), measures **urban heat islands**, predicts **future urban growth for 2035 and 2045**, and highlights **which green spaces need urgent protection**.
* **Strict Rule:** Zero fake or synthetic data. Every number comes from real satellite observations or verified offline snapshots.

---

## 2. Where Does the Data Come From?

| Data Source | What We Use It For | Time Period / Resolution |
| :--- | :--- | :--- |
| **Landsat 5 TM, 7 ETM+, 8 OLI, 9 OLI-2** (USGS/NASA Collection 2 Level-2 on GEE) | Historical & recent satellite imagery + Thermal Infrared (temperature) | Checkpoints: `2000, 2004, 2008, 2012, 2016, 2020, 2025` at **30 m** resolution |
| **ESA WorldCover 2020** (`ESA/WorldCover/v100`) | Independent global land-cover map used as a **proxy reference** to train & evaluate our classifier | 2020 (10 m global product, remapped to our 5 classes) |
| **FAO GAUL 2015 Level-1** | Official state boundary polygon of **Madhya Pradesh** (`configs/region.yaml`) | Used to validate clicked coordinates |
| **Offline Snapshots** (`data/snapshots/bhopal`, `kothri`) | Pre-computed real GEE runs saved on disk with **SHA-256 checksums** | Works instantly even without internet/GEE |

### Important Satellite Preprocessing Details (`src/greenplan/gee/composites.py`)
1. **Dry-Season Window:** We use **March 1 – May 31** (and widen by 30 days if too few cloud-free images exist) because monsoon clouds block satellites, and summer shows peak heat stress.
2. **Cloud & Saturation Masking:** Bad pixels (clouds, cloud shadows, cirrus, snow, saturated pixels) are automatically removed using `QA_PIXEL` and `QA_RADSAT`.
3. **Sensor Harmonization (Roy et al., 2016):** Older Landsat 5/7 sensors have slightly different reflectance calibrations than newer Landsat 8/9 sensors. We apply Roy et al. (2016) linear harmonization coefficients so all years from 2000 to 2025 are directly comparable.

---

## 3. How the Analysis Pipeline Works (Step-by-Step)

When you click **"Run Satellite Analysis"** (or run `uv run greenplan analyze`), here is what happens inside `src/greenplan/io/live_inputs.py` and `src/greenplan/pipeline.py`:

### Step 1: Build the Study Area Grid (`live_inputs.py`)
* Checks that your clicked `(lat, lon)` lies inside Madhya Pradesh.
* Converts the area (radius 1–25 km) into a local **UTM metric grid** at **30 m × 30 m** pixel resolution.
* For large areas (> 512 × 512 pixels), it automatically downloads data from Google Earth Engine in **512 × 512 tiles** and stitches them together so GEE memory limits are never exceeded.

### Step 2: Land-Cover Classification (`train_regional_model.py` + `live_inputs.py`)
* Inside Google Earth Engine, a **Random Forest** (`ee.Classifier.smileRandomForest`) classifies every 30 m pixel into **5 classes**:
  * `0`: **Built-up** (buildings, roads, concrete)
  * `1`: **Green space** (trees, parks, dense vegetation)
  * `2`: **Farmland** (cropland, agriculture)
  * `3`: **Water** (lakes, rivers, reservoirs)
  * `4`: **Bare land** (open soil, rock, cleared ground)
* **9 Input Features:** `blue`, `green`, `red`, `nir`, `swir1`, `swir2`, `ndvi` (vegetation index), `ndbi` (built-up index), `mndwi` (water index).
  * *Why are Thermal (TIR) and Lat/Lon excluded?* Thermal varies by weather/day, and Lat/Lon would cause the model to memorize map locations instead of learning real land properties.

### Step 3: Temporal Persistence Filter (`pipeline.py`)
* Satellite classification can occasionally misclassify a single year due to haze or shadows.
* Because real cities rarely un-build themselves, once a pixel becomes **Built-up (`0`)**, it stays Built-up unless at least **two consecutive** later checkpoints confirm it is non-built (e.g., permanent flooding). This removes noisy flickering.

### Step 4: Accuracy Validation (`validation/metrics.py`)
* We compare our 2020 classification against **ESA WorldCover 2020** (independent proxy reference) using **4-fold Spatial-Block Cross-Validation** (splitting the map into geographic blocks rather than random pixels so nearby pixels don't leak into the test set).
* Outputs **Overall Accuracy**, **Cohen's Kappa ($\kappa$)**, **Quantity/Allocation Disagreement**, and **Olofsson area-adjusted accuracy**.

### Step 5: Urban Heat Island & Thermal Analysis (`thermal/regression.py`)
* Computes **Land Surface Temperature (LST)** in °C from Landsat's thermal infrared band.
* Runs three statistical checks:
  1. **OLS Linear Regression (`LST ~ NDVI + BuiltUpDensity`):** Quantifies how much vegetation cools the land (negative NDVI coefficient) and how much concrete/buildings heat it up (positive Built-up coefficient), with $R^2$, p-values, and VIF.
  2. **Moran's I:** Checks whether regression residuals are spatially clustered.
  3. **Getis-Ord Gi\* Hotspot Analysis:** Finds statistically significant **hotspots** (90%, 95%, 99% confidence) where high temperatures cluster together in space, rather than just looking at isolated warm pixels.

### Step 6: Future Urban Growth Prediction (`simulation/`)
1. **Markov Chain (`markov.py`):** Measures historical land-cover change rates across adjacent 4-year steps (`2000→2004`, `2004→2008`, ..., `2020→2025`) to calculate **how much** land wants to convert in the future.
2. **Suitability Model (`suitability.py`):** Uses a **Random Forest** trained on spatial drivers — *proximity to existing built-up areas* and *distance from water* (plus *roads* if available) — to calculate **where** growth is most likely to happen. Checks **VIF** (Variance Inflation Factor) so drivers aren't redundant.
3. **Cellular Automata (`ca.py`):** Allocates pixels for **2035** (`2025 + 10`) and **2045** (`2025 + 20`) under 3 scenarios:
   * **BAU (Business-as-Usual):** Historical growth trend continues unchanged.
   * **Conservation-First:** Policy scenario assuming a **50% reduction** in green-space conversion probability.
   * **Infill:** Encourages compact growth near existing urban centers rather than sprawl.
4. **Out-of-Sample Hindcast Validation (`pipeline.py`):** To prove the simulation works before trusting 2035, we train a separate Markov + Suitability model using **only `2000→2016` data** (hiding 2020 completely), predict **2020**, and score it against real 2020 data using **Pontius Figure of Merit (FoM)**.

### Step 7: 3×3 Bivariate Planning Priority Matrix (`planning/decision_matrix.py`)
* Does **not** predict the future itself — instead, it **uses the 2035 BAU prediction** to guide action today.
* Splits every valid pixel into **Low / Medium / High** on two axes (3 × 3 = **9 zones**):
  1. **Green Loss Risk:** Combines current 2025 green cover, predicted **2035 BAU** loss, and urban suitability.
  2. **Thermal Stress:** Derived from real Landsat Land Surface Temperature (`lst_recent`).
* Assigns policy actions from `configs/recommendations.yaml` (e.g., *High Risk + High Heat* = Urgent Green Corridor & Canopy Protection).

---

## 4. Why These Algorithms? (Teacher / Viva Q&A)

### Why Random Forest (and how does it work)?
* **How it works:** Random Forest builds an ensemble of **100 decision trees** (`numberOfTrees=100`, `minLeafPopulation=5`). Each tree sees a random subset of training pixels and spectral bands and votes on the land-cover class (e.g., Built-up vs. Green space). The majority vote wins.
* **Why we chose it:**
  1. Satellite bands (`NDVI`, `SWIR`, `NIR`, etc.) have **non-linear** relationships (e.g., wet soil vs. concrete vs. dry farmland). Random Forest captures these non-linear boundaries easily without needing strict statistical assumptions or feature scaling.
  2. It is **resistant to outliers/noise** in satellite imagery.
  3. It runs **natively inside Google Earth Engine's cloud servers** (`ee.Classifier.smileRandomForest`), so we don't have to download gigabytes of raw imagery to classify a city.
  4. It gives **feature importances**, letting us verify which bands drive classification.

### Why Spatial-Block Cross-Validation instead of Random Split?
* Nearby pixels on a map look very similar (**spatial autocorrelation**). If you split pixels randomly, training and test pixels sit right next to each other, giving an artificially high accuracy (~98%). Splitting into geographic blocks gives an **honest** accuracy estimate.

### Why CA-Markov instead of a Deep Learning (U-Net / CNN) model?
* Deep learning models need huge labeled historical time-series datasets for every region and act like a "black box".
* **CA-Markov** works on *any* clicked location in Madhya Pradesh immediately, requires no pre-labeled future maps, is mathematically transparent (Markov = *how much* change, Suitability RF = *where*, CA = *spatial allocation*), and lets urban planners test **what-if scenarios** (BAU vs. Conservation vs. Infill).

### Why OLS + Moran's I + Getis-Ord Gi\* for Heat?
* Just plotting temperature doesn't prove *why* a spot is hot or whether a hotspot is statistically real.
* **OLS** proves the statistical relationship between green loss and temperature; **Moran's I** checks spatial clustering; **Getis-Ord Gi\*** isolates statistically significant heat clusters ($p < 0.05$ / $0.01$).

---

## 5. What Everything on the Frontend Dashboard Means

When you run `uv run greenplan serve` and open `http://localhost:8000`:

### Top Bar
* **Status Badge:**
  * `● Online (GEE Live)` — Google Earth Engine is authenticated and ready for live clicks anywhere in Madhya Pradesh.
  * `● Offline (Snapshots Only)` — GEE not connected; saved snapshots (`Bhopal`, `Kothri`) still work completely offline.

### Left Sidebar
* **Study Area:** Quick buttons for **Bhopal (Saved)** and **Kothri (Saved)**, or enter/click custom `Latitude`, `Longitude`, and `Radius (km)` and press **Run Satellite Analysis**.
* **Spatial Overlays (5 Map Layers):**
  1. **3×3 Planning Matrix:** Shows priority zones (green = low risk/heat, red/dark orange = urgent high risk + high heat stress).
  2. **Thermal Hotspots (Gi\*):** Red = statistically significant urban heat islands; blue = cool spots (water/dense canopy).
  3. **Urban Expansion:** Highlights pixels that converted from non-built (`2000`) to Built-up (`2025`).
  4. **Recent LULC (2025):** Latest 5-class land-cover map.
  5. **Baseline LULC (2000):** Starting 5-class land-cover map for visual comparison.
* **Layer Opacity Slider & Legend:** Adjust transparency over the satellite map.

### Interactive Map
* Defaults to **Satellite View** (Esri World Imagery, free, no API key) with a top-right toggle to switch to **Street View** (OpenStreetMap).
* Shows the Madhya Pradesh boundary and lets you click anywhere inside it to analyze that area.

### 6 KPI Cards
1. **Baseline Green Cover (2000):** % of the study area covered by green space in 2000.
2. **Recent Green Cover (2025):** % of the study area covered by green space in 2025.
3. **Net Green Change:** Total $\text{km}^2$ of green space gained or lost (`2000 → 2025`).
4. **Built-up Expansion:** New $\text{km}^2$ of concrete/urbanized area added (`2000 → 2025`).
5. **Mean LST (Recent):** Average dry-season Land Surface Temperature (°C) in 2025.
6. **High-Priority Conservation:** $\text{km}^2$ falling in the highest-priority zone (High Loss Risk + High Thermal Stress).

### Bottom Analytics & Footer
* **Historical Greenness Trend (NDVI):** Line chart of average vegetation health (`NDVI`) across the checkpoints (`2000–2025`).
* **2035 CA-Markov Scenario Projections:** Bar chart comparing predicted 2035 **Built-up** (red) vs. **Green** (green) area ($\text{km}^2$) across **BAU**, **Conservation**, and **Infill**.
* **3×3 Bivariate Planning Priority Matrix:** 9-cell grid showing exact hectares (`ha`) in each Risk × Heat combination.
* **Methodological Disclosure Footer:** Shows **Proxy Overall Accuracy** (vs. ESA WorldCover 2020), **Cohen's Kappa ($\kappa$)**, **Pontius FoM (Hindcast)** (or `Not available` if fewer than 3 checkpoints / no 2020), and **Compute Time**.

---

## 6. Quick 2-Minute Demo Flow for Your Teacher
1. Start the server: `uv run greenplan serve` and open `http://localhost:8000`.
2. Point out the **Satellite View** map and the **Madhya Pradesh** boundary.
3. Click **Bhopal (Saved)** (or **Kothri (Saved)**) to show instant offline loading (~220 ms) with **SHA-256 integrity verification**.
4. Toggle between **Baseline LULC (2000)**, **Recent LULC (2025)**, **Urban Expansion**, **Thermal Hotspots (Gi\*)**, and **3×3 Planning Matrix** so your teacher sees the story: *urban growth → vegetation loss → heat islands → where to conserve*.
5. Walk through the **KPI cards**, the **NDVI trend**, the **2035 BAU vs. Conservation vs. Infill** chart, and the **Methodological Disclosure** footer (mentioning Roy et al. 2016 harmonization, ESA WorldCover 2020 proxy validation, and held-out 2020 hindcast).
6. *(If internet/GEE is connected)* Click a new point inside Madhya Pradesh and click **Run Satellite Analysis** to show live Google Earth Engine execution!
