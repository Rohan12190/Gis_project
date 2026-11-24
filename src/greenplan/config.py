"""Configuration models and loader for GreenPlan using Pydantic."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class WindowConfig(BaseModel):
    """Seasonal date window for compositing."""

    start: str = Field(..., description="Start date in MM-DD format")
    end: str = Field(..., description="End date in MM-DD format")

    @field_validator("start", "end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Date window must be in MM-DD format, got: {v}")
        m, d = int(parts[0]), int(parts[1])
        if not (1 <= m <= 12 and 1 <= d <= 31):
            raise ValueError(f"Invalid month or day in window: {v}")
        return v


class LSTWindowConfig(BaseModel):
    """Hot-season window for Land Surface Temperature calculation."""

    start: str = Field("04-01", description="Hot season start MM-DD")
    end: str = Field("05-31", description="Hot season end MM-DD")


class ClassSchemeItem(BaseModel):
    """Single LULC class definition."""

    id: int = Field(..., ge=0, le=10)
    name: str = Field(..., min_length=1)
    color: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")
    description: str = ""


class AOIConfig(BaseModel):
    """Area of Interest definition."""

    name: str = Field(..., min_length=1)
    geojson_path: str = Field(..., description="Relative path to AOI GeoJSON file")
    buffer_meters: float = Field(0.0, ge=0.0)


class ScenarioConfig(BaseModel):
    """Simulation scenario definition."""

    name: str
    description: str
    protection_mask: bool = False
    green_protection_buffer_m: float = 0.0


class TriggerConfig(BaseModel):
    """Structured trigger condition matching planning unit characteristics."""

    risk: List[str] = Field(default_factory=lambda: ["low", "medium", "high"])
    heat: List[str] = Field(default_factory=lambda: ["low", "medium", "high"])
    min_green_pct: Optional[float] = Field(None, ge=0.0, le=100.0)

    @field_validator("risk", "heat")
    @classmethod
    def validate_levels(cls, v: List[str]) -> List[str]:
        valid_levels = {"low", "medium", "high"}
        if not v:
            raise ValueError("Trigger level list cannot be empty")
        for level in v:
            if level not in valid_levels:
                raise ValueError(
                    f"Invalid risk/heat level '{level}'. Must be one of: {sorted(valid_levels)}"
                )
        return v


class RecommendationsRule(BaseModel):
    """Single environmental policy recommendation with structured trigger condition."""

    id: str
    title: str
    severity: str = Field(..., pattern=r"^(critical|moderate|protect|good)$")
    trigger: TriggerConfig
    actions: List[str] = Field(..., min_length=1)
    source: Optional[str] = None
    assumption: bool = False

    @field_validator("id", "title")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_source_or_assumption(self) -> "RecommendationsRule":
        if not self.source and not self.assumption:
            raise ValueError(
                f"Rule '{self.id}' must provide either a 'source' citation or 'assumption: true'"
            )
        return self


class PipelineConfig(BaseModel):
    """Master pipeline configuration schema."""

    project_name: str = Field("EPICS23-554", min_length=1)
    study_area: AOIConfig
    projected_crs: str = Field("EPSG:32643", pattern=r"^EPSG:\d+$")
    random_seed: int = Field(42, ge=0)
    epochs: List[int] = Field(default_factory=lambda: [2000, 2010, 2020])
    projection_years: List[int] = Field(default_factory=lambda: [2030, 2040])
    class_scheme: List[ClassSchemeItem]
    composite_window: WindowConfig
    lst_window: LSTWindowConfig
    planning_grid_size_m: float = Field(500.0, gt=0.0)
    rf_num_trees: int = Field(100, ge=10)
    rf_min_split: int = Field(2, ge=1)

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v: List[int]) -> List[int]:
        if len(v) < 2:
            raise ValueError("At least 2 epochs are required for change detection.")
        if v != sorted(v):
            raise ValueError("Epochs must be sorted in ascending order.")
        return v


def load_pipeline_config(config_path: Path | str) -> PipelineConfig:
    """Load and validate pipeline configuration YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)
