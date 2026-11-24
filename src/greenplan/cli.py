"""Command-line interface for GreenPlan."""

from pathlib import Path

import typer

from greenplan.config import load_pipeline_config, load_region_config
from greenplan.gee.landsat import load_dotenv

load_dotenv()

app = typer.Typer(help="GreenPlan: Real-Data GIS Urban Growth & Green Space Planning CLI")


@app.callback()
def main():
    """GreenPlan CLI."""
    pass


@app.command()
def validate_config(
    config: Path = typer.Option("configs/pipeline.yaml", help="Path to pipeline config YAML"),
    region: Path = typer.Option("configs/region.yaml", help="Path to region config YAML"),
):
    """Validate pipeline and region configuration against Pydantic schema."""
    reg_cfg = load_region_config(region)
    pipe_cfg = load_pipeline_config(config)
    typer.echo(
        f"Region Configuration Valid! Region: '{reg_cfg.name}' | Asset: '{reg_cfg.boundary_asset}'"
    )
    typer.echo(f"Default Checkpoints: {reg_cfg.default_checkpoints}")
    typer.echo(
        f"Pipeline Configuration Valid! Project: '{pipe_cfg.project_name}' | Classes: {[c.name for c in pipe_cfg.class_scheme]}"
    )


@app.command()
def train_model(
    region: Path = typer.Option("configs/region.yaml", help="Path to region config YAML"),
    worldcover_mapping: Path = typer.Option(
        "configs/worldcover_mapping.yaml", help="Path to WorldCover mapping YAML"
    ),
    output_dir: Path = typer.Option("models", help="Directory to save model metrics"),
    samples_path: Path = typer.Option(
        "data/training/samples.csv", help="Path to training samples CSV cache"
    ),
    resample: bool = typer.Option(
        False, "--resample", help="Force re-sampling from Earth Engine even if cache exists"
    ),
):
    """Train regional land-cover Random Forest model using spatial-block cross-validation."""
    from greenplan.models.train_regional_model import run_training_pipeline

    typer.echo("Initiating regional land-cover model training...")
    metrics = run_training_pipeline(
        region_config=region,
        worldcover_mapping=worldcover_mapping,
        samples_path=samples_path,
        output_dir=output_dir,
        force_resample=resample,
    )
    typer.echo(f"Training Complete! Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    typer.echo(f"Kappa: {metrics['kappa']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    typer.echo(
        f"Quantity Disagreement: {metrics['quantity_disagreement']:.4f} | "
        f"Allocation Disagreement: {metrics['allocation_disagreement']:.4f}"
    )
    typer.echo(f"Metrics written to '{output_dir}/regional_model_metrics.json'")


@app.command()
def analyze(
    lat: float = typer.Option(..., "--lat", help="Latitude coordinate (inside region boundary)"),
    lon: float = typer.Option(..., "--lon", help="Longitude coordinate (inside region boundary)"),
    radius: float = typer.Option(10.0, "--radius", help="Analysis radius in kilometers"),
    output: Path = typer.Option(Path("outputs"), "--output", help="Directory to save run results"),
    name: str = typer.Option(None, "--name", help="Optional human-readable place name"),
):
    """Execute live Earth Engine multi-epoch GIS analysis for clicked coordinates."""
    from datetime import datetime, timezone

    from greenplan.io.live_inputs import build_live_inputs
    from greenplan.pipeline import run_analysis

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Fetching real satellite inputs from Earth Engine for ({lat:.4f}, {lon:.4f})...")

    def _progress(msg: str, frac: float):
        typer.echo(f"[{int(frac*100):3d}%] {msg}")

    inputs = build_live_inputs(lat=lat, lon=lon, radius_km=radius, name=name, progress=_progress)

    typer.echo("Executing place-agnostic spatial analysis and simulation pipeline...")
    summary = run_analysis(inputs=inputs, run_dir=run_dir, progress=_progress)

    typer.echo("\n--- Analysis Complete ---")
    typer.echo(f"Location: {summary.study_area}")
    typer.echo(
        f"Green Cover Change: {summary.green_cover_pct_baseline}% ({summary.baseline_year}) "
        f"-> {summary.green_cover_pct_recent}% ({summary.recent_year})"
    )
    typer.echo(f"Net Green Loss: {summary.net_green_loss_sqkm} sq km")
    typer.echo(f"Built-up Urban Growth: {summary.builtup_growth_sqkm} sq km")
    typer.echo(f"Recent Mean LST: {summary.mean_lst_celsius_recent} °C")
    typer.echo(f"High-Priority Conservation Area: {summary.high_priority_conservation_sqkm} sq km")
    typer.echo(f"Full results written to: {run_dir / 'results.json'}")


if __name__ == "__main__":
    app()
