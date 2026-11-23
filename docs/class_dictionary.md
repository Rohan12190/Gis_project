# Operational LULC Class Dictionary

**Project:** EPICS23-554 — GreenPlan  
**Sensor Resolution:** 30 m × 30 m ($900\text{ m}^2$ per pixel)  
**Minimum Mapping Unit (MMU):** 4 contiguous pixels ($3,600\text{ m}^2 \approx 0.36\text{ ha}$)

---

## Important Spatial Resolution Disclaimer
Landsat Level-2 surface reflectance provides $30\text{ m}$ spatial resolution. At this resolution, **individual roadside trees, small residential front/back gardens, green roofs, and pocket parks smaller than $0.36\text{ ha}$ cannot be resolved as pure pixels**; they appear as mixed pixels (often blended with asphalt roads and residential roof tiles). Therefore, "Green space" in this study operationally designates contiguous vegetative canopy, regional parks, urban forests, and persistent natural/semi-natural plant cover.

---

## Class Definitions

| Class ID | Class Name | Operational Definition & Spectral Characteristics | Typical Bhopal Examples | Distinguishing Rule vs Farmland |
| :---: | :---: | :--- | :--- | :--- |
| **0** | **Built-up** | Impervious man-made structures including residential complexes, commercial zones, industrial sheds, paved asphalt/concrete roadways, and paved public squares. High NDBI, low NDVI. | MP Nagar, New Market, Kolar Road developments, BHEL industrial township, airport tarmac. | High density of impervious reflectance across all seasonal windows. |
| **1** | **Green space** | Perennial tree canopy, conserved dense forest reserves, large public recreational parks, and permanent vegetated buffers. Characterized by high NDVI ($>0.45$) during both monsoon and dry pre-monsoon summer windows. | Van Vihar National Park, Regional Park, Shyamala Hills forested tracts, MANIT campus greens, military cantonment canopy. | **Key Rule:** Maintains persistent high NDVI in pre-monsoon dry season (March–May), whereas farmland is harvested/fallow. |
| **2** | **Farmland** | Cultivated agricultural croplands, seasonal agro-ecosystems, and actively farmed rural/peri-urban plots. Displays high phenological variability: high NDVI during post-monsoon crop cycles (Kharif/Rabi) dropping sharply to bare soil reflectance post-harvest. | Agricultural plains surrounding Berasia road, Phanda block croplands, Misrod peri-urban farming tracts. | **Key Rule:** Exhibits strong seasonal drop in NDVI during the dry pre-monsoon compositing window (April–May). |
| **3** | **Water** | Permanent and semi-permanent surface water bodies, lakes, rivers, large reservoirs, and perennial wetlands. High MNDWI ($>0.2$), low/negative NDVI, near-zero SWIR reflectance. | Upper Lake (Bhojtal), Lower Lake, Kaliyasot Dam reservoir, Kerwa Dam reservoir, Halali backwaters. | Consistently negative NDBI and high MNDWI. |
| **4** | **Bare land** | Exposed mineral soil, rocky outcrops, dry unpaved open ground, excavated construction plots, and fallow non-vegetated terrain. Moderate-to-high SWIR and Red reflectance, low NDVI ($<0.15$), moderate NDBI. | Cleared development tracts along outer bypass, open quarry areas near Kerwa, fallow uncultivated wasteland. | Distinguishable from built-up by lack of road grid geometry and lower spatial texture/heterogeneity. |

