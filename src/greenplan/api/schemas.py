"""Pydantic schemas defining the frontend contract and run manifests."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ArtifactRecord(BaseModel):
    """Metadata for a single output artifact."""

    name: str
    artifact_type: str = Field(..., description="raster, vector, table, metric, or figure")
    path: str = Field(..., description="Relative path from run root")
    format: str = Field(..., description="tif, geojson, csv, json, png")
    sha256: str = Field(..., description="SHA-256 checksum of the file")
    description: str = ""


class BoundingBox(BaseModel):
    """Geographic bounds in WGS 84 [min_lon, min_lat, max_lon, max_lat]."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class RunManifest(BaseModel):
    """Self-describing manifest for a completed GreenPlan pipeline run."""

    schema_version: str = "1.0.0"
    run_id: str
    timestamp_utc: str
    project_name: str
    study_area: str
    projected_crs: str
    bounds_wgs84: BoundingBox
    config_hash: str
    artifacts: List[ArtifactRecord]


class ClassAccuracyMetric(BaseModel):
    """Accuracy metrics for an individual LULC class."""

    class_id: int
    class_name: str
    producers_accuracy: float
    users_accuracy: float
    f1_score: float
    sample_count: int
    area_adjusted_ha: float
    area_ci_95_ha: float


class LULCAccuracyReport(BaseModel):
    """Accuracy assessment report for a single epoch."""

    epoch: int
    overall_accuracy: float
    cohen_kappa: float
    class_metrics: List[ClassAccuracyMetric]


class ThermalRegressionSummary(BaseModel):
    """Summary of LST regression against NDVI and built-up fraction."""

    epoch: int
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    p_value: float
    ndvi_coef: float
    ndvi_pvalue: float
    builtup_coef: float
    builtup_pvalue: float
    moran_i_residuals: float
    moran_pvalue: float


class SimulationValidationMetric(BaseModel):
    """Validation metrics comparing simulated map to actual held-out map."""

    validation_year: int
    overall_agreement: float
    null_model_agreement: float
    figure_of_merit: float
    kappa_standard: float
    kappa_location: float
    kappa_quantity: float
    beats_null_model: bool


class PlanningUnitSummary(BaseModel):
    """Aggregated planning data for a single 500m grid cell."""

    unit_id: str
    loss_risk_level: str = Field(..., pattern=r"^(low|medium|high)$")
    thermal_stress_level: str = Field(..., pattern=r"^(low|medium|high)$")
    priority_category: str
    recommended_action_ids: List[str]


class RunSummary(BaseModel):
    """Headline metrics for dashboard display."""

    run_id: str
    study_area: str
    baseline_year: int
    recent_year: int
    green_cover_pct_baseline: float
    green_cover_pct_recent: float
    net_green_loss_sqkm: float
    builtup_growth_sqkm: float
    mean_lst_celsius_recent: float
    high_priority_conservation_sqkm: float
    farmland_pct_baseline: Optional[float] = None
    farmland_pct_recent: Optional[float] = None
    builtup_pct_baseline: Optional[float] = None
    builtup_pct_recent: Optional[float] = None
    water_pct: Optional[float] = None
    data_source: Optional[str] = "empirical_satellite"
    accuracy_reference: Optional[str] = None
