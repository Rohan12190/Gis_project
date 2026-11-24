"""FastAPI backend server for GreenPlan real-data GIS analysis, runs, and snapshots."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from greenplan.config import load_region_config
from greenplan.gee.client import init_ee, is_ee_initialized
from greenplan.io.live_inputs import build_live_inputs, validate_point_in_region
from greenplan.pipeline import run_analysis
from greenplan.snapshots import list_snapshots, load_snapshot

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GreenPlan: Real-Data GIS Planning Engine",
    description="Real satellite-based urban growth, green space loss, and urban heat island analysis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_STORE: Dict[str, Dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    """Payload for initiating a live satellite analysis run."""

    lat: float = Field(..., description="Target latitude coordinate")
    lon: float = Field(..., description="Target longitude coordinate")
    radius_km: float = Field(10.0, ge=1.0, le=25.0, description="Analysis buffer radius in km")
    epochs: Optional[List[int]] = Field(None, description="Optional custom checkpoint epochs")
    name: Optional[str] = Field(None, description="Optional human-readable study area label")


def _execute_analysis_job(job_id: str, payload: AnalyzeRequest, run_dir: Path) -> None:
    """Background task executing the complete live analysis workflow."""
    try:
        JOBS_STORE[job_id]["status"] = "running"

        def _progress(msg: str, frac: float):
            JOBS_STORE[job_id]["progress_pct"] = int(frac * 100)
            JOBS_STORE[job_id]["message"] = msg

        inputs = build_live_inputs(
            lat=payload.lat,
            lon=payload.lon,
            radius_km=payload.radius_km,
            epochs=payload.epochs,
            name=payload.name,
            progress=_progress,
        )

        summary = run_analysis(inputs, run_dir, progress=_progress)

        with open(run_dir / "results.json", "r", encoding="utf-8") as f:
            full_results = json.load(f)

        # bounds in inputs: [min_lat, min_lon, max_lat, max_lon] -> [min_lon, min_lat, max_lon, max_lat]
        bounds = [inputs.bounds[1], inputs.bounds[0], inputs.bounds[3], inputs.bounds[2]]

        JOBS_STORE[job_id]["status"] = "completed"
        JOBS_STORE[job_id]["progress_pct"] = 100
        JOBS_STORE[job_id]["message"] = "Analysis finished successfully!"
        JOBS_STORE[job_id]["run_id"] = run_dir.name
        JOBS_STORE[job_id]["result_url"] = f"/api/runs/{run_dir.name}"
        JOBS_STORE[job_id]["summary"] = summary.model_dump()
        JOBS_STORE[job_id]["result"] = {
            "run_id": run_dir.name,
            "bounds": bounds,
            "overlays": {
                "planning_priority": f"/api/runs/{run_dir.name}/overlays/planning_priority.png",
                "thermal_hotspots": f"/api/runs/{run_dir.name}/overlays/thermal_hotspots.png",
                "urban_growth": f"/api/runs/{run_dir.name}/overlays/urban_growth.png",
                "lulc_recent": f"/api/runs/{run_dir.name}/overlays/lulc_recent.png",
                "lulc_baseline": f"/api/runs/{run_dir.name}/overlays/lulc_baseline.png",
            },
            "results": full_results,
        }

    except Exception as exc:
        logger.exception("Analysis job '%s' failed: %s", job_id, exc)
        JOBS_STORE[job_id]["status"] = "failed"
        JOBS_STORE[job_id]["error"] = str(exc)
        JOBS_STORE[job_id]["message"] = f"Failed: {exc}"


@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Return backend operational status, regional boundary config, and Earth Engine state."""
    try:
        init_ee()
        online = is_ee_initialized()
    except Exception:
        online = False

    reg_cfg = load_region_config("configs/region.yaml")
    return {
        "status": "online" if online else "offline_ready",
        "earth_engine_connected": online,
        "gee_available": online,
        "region_name": reg_cfg.name,
        "boundary_asset": reg_cfg.boundary_asset,
        "default_checkpoints": reg_cfg.default_checkpoints,
        "default_radius_km": reg_cfg.default_radius_km,
        "max_radius_km": reg_cfg.max_radius_km,
    }


@app.get("/api/region")
def get_region_boundary() -> Any:
    """Return regional boundary polygon as GeoJSON for frontend interactive mapping."""
    cached_geojson = Path("configs/region_boundary.geojson")
    if cached_geojson.exists():
        with open(cached_geojson, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import ee

        init_ee()
        reg_cfg = load_region_config("configs/region.yaml")
        region_fc = ee.FeatureCollection(reg_cfg.boundary_asset).filter(
            ee.Filter.eq(reg_cfg.filter_field, reg_cfg.filter_value)
        )
        geo = region_fc.geometry().getInfo()
        with open(cached_geojson, "w", encoding="utf-8") as f:
            json.dump(geo, f)
        return geo
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Regional boundary unavailable offline: {e}. Run with live network once to cache.",
        ) from e


@app.post("/api/analyze")
def submit_analysis(payload: AnalyzeRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Validate requested point and launch background satellite analysis job."""
    # 1. Point validation against regional boundary
    try:
        validate_point_in_region(payload.lat, payload.lon)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Earth Engine validation unavailable: {exc}. Ensure credentials are valid.",
        ) from exc

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = Path("outputs") / f"run_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    JOBS_STORE[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0,
        "message": "Queued for processing...",
        "result_url": None,
        "error": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(_execute_analysis_job, job_id, payload, run_dir)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Query progress and result status for an active or completed background job."""
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOBS_STORE[job_id]


@app.get("/api/runs")
def list_runs(base_dir: Path = Path("outputs")) -> List[Dict[str, Any]]:
    """List historical analysis runs with headline KPIs."""
    if not base_dir.exists():
        return []

    runs = []
    for r_dir in sorted(base_dir.iterdir(), reverse=True):
        if r_dir.is_dir() and (r_dir / "summary.json").exists():
            try:
                with open(r_dir / "summary.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)
                runs.append({
                    "run_id": r_dir.name,
                    "study_area": summary.get("study_area", r_dir.name),
                    "baseline_year": summary.get("baseline_year"),
                    "recent_year": summary.get("recent_year"),
                    "green_loss_sqkm": summary.get("net_green_loss_sqkm"),
                    "urban_growth_sqkm": summary.get("builtup_growth_sqkm"),
                    "mean_lst_recent": summary.get("mean_lst_celsius_recent"),
                })
            except Exception:
                continue
    return runs


@app.get("/api/runs/{run_id}")
def get_run_results(run_id: str, base_dir: Path = Path("outputs")) -> Any:
    """Retrieve complete results.json and metadata for a completed run."""
    run_path = base_dir / run_id / "results.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Run results not found for '{run_id}'.")
    with open(run_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_overlay_file(folder: Path, name: str) -> Optional[Path]:
    """Resolve an overlay image file robustly across standard names and aliases."""
    if not folder.exists():
        return None

    aliases = {
        "planning_priority": ["overlay_priority.png", "overlay_planning_priority.png", "priority.png"],
        "thermal_hotspots": ["overlay_thermal.png", "overlay_thermal_hotspots.png", "thermal.png"],
        "urban_growth": ["overlay_expansion.png", "overlay_urban_growth.png", "expansion.png"],
        "lulc_baseline": ["overlay_lulc_2000.png", "overlay_lulc_baseline.png"],
        "lulc_recent": ["overlay_lulc_2025.png", "overlay_lulc_recent.png"],
    }
    candidate_names = aliases.get(name, [f"overlay_{name}.png", f"{name}.png"])

    # Check dynamic years for LULC if not exact
    if name == "lulc_baseline":
        lulc_files = sorted(folder.glob("overlay_lulc_*.png"))
        if lulc_files:
            return lulc_files[0]
    elif name == "lulc_recent":
        lulc_files = sorted(folder.glob("overlay_lulc_*.png"))
        if lulc_files:
            return lulc_files[-1]

    for cand in candidate_names:
        p = folder / cand
        if p.exists():
            return p
        if not cand.endswith(".png"):
            p_png = folder / f"{cand}.png"
            if p_png.exists():
                return p_png

    return None


@app.get("/api/runs/{run_id}/overlays/{overlay_name}.png")
def get_run_overlay(run_id: str, overlay_name: str, base_dir: Path = Path("outputs")) -> FileResponse:
    """Serve georeferenced transparent PNG overlay for Leaflet dashboard map."""
    figures_dir = base_dir / run_id / "figures"
    target = _find_overlay_file(figures_dir, overlay_name)
    if target:
        return FileResponse(target, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Overlay '{overlay_name}' not found for run '{run_id}'.")


@app.get("/api/snapshots")
def get_available_snapshots() -> List[Dict[str, Any]]:
    """List pre-computed offline snapshots (e.g. Bhopal, Kothri)."""
    return list_snapshots()


@app.get("/api/snapshots/{snapshot_id}")
def get_snapshot_details(snapshot_id: str) -> Any:
    """Load verified results and metadata for a specific offline snapshot."""
    try:
        data = load_snapshot(snapshot_id)
        meta = data["metadata"]
        res = data["results"]
        snap_dir = data["snapshot_dir"]

        # Read bounds from inputs.npz if present
        bounds = None
        inputs_file = snap_dir / "inputs.npz"
        if inputs_file.exists():
            with np.load(inputs_file) as npz:
                if "bounds" in npz:
                    # bounds in inputs.npz: [min_lat, min_lon, max_lat, max_lon]
                    b = npz["bounds"].tolist()
                    # Return [min_lon, min_lat, max_lon, max_lat] for Leaflet
                    bounds = [b[1], b[0], b[3], b[2]]

        if bounds is None:
            # Fallback approximate bounding box from lat, lon, radius_km
            lat, lon, r = meta["lat"], meta["lon"], meta.get("radius_km", 15.0)
            dlat = r / 111.0
            dlon = r / (111.0 * np.cos(np.radians(lat)))
            bounds = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

        overlays = {
            "lulc_baseline": f"/api/snapshots/{snapshot_id}/overlays/lulc_baseline.png",
            "lulc_recent": f"/api/snapshots/{snapshot_id}/overlays/lulc_recent.png",
            "urban_growth": f"/api/snapshots/{snapshot_id}/overlays/urban_growth.png",
            "thermal_hotspots": f"/api/snapshots/{snapshot_id}/overlays/thermal_hotspots.png",
            "planning_priority": f"/api/snapshots/{snapshot_id}/overlays/planning_priority.png",
        }

        return {
            **res,
            "id": snapshot_id,
            "name": meta.get("name", snapshot_id),
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "radius_km": meta.get("radius_km", 15.0),
            "bounds": bounds,
            "overlays": overlays,
            "results": res,
            "metadata": meta,
        }
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe)) from fe
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=f"Integrity check failed: {ve}") from ve


@app.get("/api/snapshots/{snapshot_id}/overlays/{overlay_name}.png")
def get_snapshot_overlay(
    snapshot_id: str, overlay_name: str, base_dir: Path = Path("data/snapshots")
) -> FileResponse:
    """Serve transparent overlay PNG from a verified snapshot directory."""
    overlays_dir = base_dir / snapshot_id / "overlays"
    target = _find_overlay_file(overlays_dir, overlay_name)
    if target:
        return FileResponse(target, media_type="image/png")

    raise HTTPException(
        status_code=404, detail=f"Overlay '{overlay_name}' not found for snapshot '{snapshot_id}'."
    )


@app.get("/api/snapshots/{snapshot_id}/thumbnail.png")
def get_snapshot_thumbnail(
    snapshot_id: str, base_dir: Path = Path("data/snapshots")
) -> FileResponse:
    """Serve RGB true-color thumbnail image for snapshot card."""
    thumb_path = base_dir / snapshot_id / "thumbnail.png"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail not found for snapshot '{snapshot_id}'.")
    return FileResponse(thumb_path, media_type="image/png")


# Mount static frontend files if static directory exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
