# ADR 001: Backend Architecture for Future User Interface

**Date:** 2026-11-19  
**Status:** Approved  
**Context:** Phase 9 / Architectural Foundation

---

## Decision
For any future interactive web visualization dashboard (Phase 9+), **FastAPI** is selected as the recommended backend micro-framework, and the legacy Flask application (`legacy/gis/app.py`) will **not** be carried forward.

## Rationale
1. **Schema Reuse:** The pipeline models in `src/greenplan/api/schemas.py` are natively authored using Pydantic v2. FastAPI natively uses Pydantic for request and response serialization, automatic OpenAPI documentation, and runtime validation. Reusing these schemas avoids duplicate data transfer objects.
2. **Asynchronous I/O:** Reading large GeoJSON features, vector tiles, and Cloud-Optimized GeoTIFFs (COG) benefits from asynchronous non-blocking event loops, which FastAPI provides via ASGI (Starlette / Uvicorn).
3. **Decoupled Architecture:** The scientific GIS pipeline produces persistent data artifacts (GeoTIFFs, GeoJSONs, manifests, and summary JSONs). The web backend will act purely as an artifact serving and querying API, rather than mixing deep learning inference with web request threads.

