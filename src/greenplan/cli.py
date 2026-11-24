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


if __name__ == "__main__":
    app()
