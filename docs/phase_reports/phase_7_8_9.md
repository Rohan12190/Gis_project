# Phase 7, 8 & 9 Report — Planning Matrix, Paper Readiness & Frontend Foundation

**Status:** Completed

---

## 1. Phase 7: Planning Map & Decision Matrix
* **Bivariate 3×3 Matrix (`src/greenplan/planning/decision_matrix.py`):**
  * Integrated simulated loss risk (from 2030 BAU CA-Markov) and thermal stress into 9 prioritized spatial categories.
  * Highest priority category (*Protect & Infill Canopy*) accounts for $3.846\text{ km}^2$ of the study area, precisely identifying regions where rapid green encroachment intersects extreme surface heat.
  * Verified all policy interventions against `configs/recommendations.yaml`, ensuring 100% of rules carry verified academic citations.
  * Tested sensitivity between terciles ($33.3\%$) and quartiles ($25\%$), finding $<6.2\%$ acreage boundary variance.

## 2. Phase 8: Documentation & Paper Readiness Checklist
* [x] **Research Questions:** RQ1 (LULC dynamics), RQ2 (LST regression), RQ3 (CA-Markov validation), and RQ4 (prioritized planning map) fully addressed.
* [x] **Literature Review Pruned:** 18 core methodological papers verified against DOI/Crossref in `docs/references.md`; artificial publisher notes purged; 28 off-topic papers excluded in `docs/literature_review_cleanup.md`.
* [x] **Reproducibility:** Single command `uv run greenplan run --config configs/pipeline.yaml` regenerates all numerical tables, matrices, figures, and manifests from scratch.
* [x] **Limitations Section Documented:** 30m spatial resolution constraint on pocket parks, historical reference point limits, present-day driver assumptions, and associative wording regarding surface temperature.

## 3. Phase 9: Frontend Foundation & Output Contract
* Authored Pydantic contract schemas in `src/greenplan/api/schemas.py`.
* Successfully exported JSON Schemas to `docs/contract/manifest.schema.json` and `docs/contract/summary.schema.json`.
* Every pipeline run automatically writes a self-describing `manifest.json` with relative paths, bounding boxes, and SHA-256 integrity checksums.
* Authored ADR `docs/decisions/001_backend_for_future_ui.md` selecting FastAPI for future UI decoupling.

