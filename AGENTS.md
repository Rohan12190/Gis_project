# Rules of Engagement for Coding Agents

This repository is being reworked under a peer-review defensibility mandate. All agentic operations must adhere to these rules:

1. **Audit before changing:** No blind edits. Changes must align with the Rework Plan.
2. **Never delete legacy work:** All pre-existing work is tagged under `legacy-unet-v1` and moved to `legacy/`.
3. **Never invent results:** Every number, metric, figure, or citation detail must come from verified code execution or DOI sources. When unknown, use `TODO(owner): ...`.
4. **Never invent citations:** All reference citations must be verified against DOI/Crossref in `docs/references.md`.
5. **Config over constants:** Zero hardcoded paths, years, or bounding coordinates. All settings reside in `configs/*.yaml` and are validated by Pydantic.
6. **Small reviewable commits:** Ensure tests pass before committing.
7. **Phase reports:** Document phase outputs in `docs/phase_reports/phase_N.md`.
8. **Stop and ask:** If breaking conflicts or authentication blockers arise.
9. **No premature UI:** Keep business logic independent of UI; provide schemas in `src/greenplan/api/schemas.py`.
10. **Data persistence rule:** Every result is stored as data (GeoTIFF, GeoJSON, CSV, JSON). Visual plots are downstream artifacts.

