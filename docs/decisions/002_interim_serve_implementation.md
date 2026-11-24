# ADR 002: Interim HTTP Server Implementation for Viewer Dashboard

**Date:** 2026-11-24  
**Status:** Approved  
**Context:** Phase 9 / Developer Preview & Report Review

---

## Decision
For developer inspection and dashboard viewing, the CLI provides an interim `greenplan serve` command based on Python's built-in `http.server.SimpleHTTPRequestHandler` with socket address reuse, serving static HTML/GIS overlays from the latest or specified pipeline run directory (`outputs/run_*`), with a lightweight `/api/predict_gee` endpoint proxying requests to Earth Engine.

## Rationale
1. **Zero External Overhead:** Developers and reviewers can immediately view interactive reports, maps, and figures without installing or configuring external ASGI servers, database drivers, or web frameworks.
2. **Decoupled Architecture:** As decided in ADR 001, the core scientific pipeline remains entirely independent of web presentation logic. `serve` merely hosts pre-generated static artifacts (`dashboard.html`, transparent PNG overlays, and CSV/JSON tables) and exposes live multi-spectral queries.
3. **Production Path:** Full interactive deployment (Phase 9+) will migrate this interim handler to the FastAPI ASGI micro-framework as approved in ADR 001.
