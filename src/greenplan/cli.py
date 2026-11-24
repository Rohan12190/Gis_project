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


if __name__ == "__main__":
    app()
