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


snapshot_app = typer.Typer(help="Manage frozen offline GIS snapshots")
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("create")
def snapshot_create(
    snapshot_id: str = typer.Option(..., "--id", help="Unique identifier for the snapshot"),
    lat: float = typer.Option(..., "--lat", help="Center latitude coordinate"),
    lon: float = typer.Option(..., "--lon", help="Center longitude coordinate"),
    radius: float = typer.Option(10.0, "--radius", help="Snapshot radius in kilometers"),
    output: Path = typer.Option(
        Path("data/snapshots"), "--output", help="Base directory for snapshots"
    ),
):
    """Create a self-contained frozen snapshot from real Earth Engine data."""
    from greenplan.snapshots import create_snapshot

    typer.echo(
        f"Creating snapshot '{snapshot_id}' for ({lat:.4f}, {lon:.4f}) with radius {radius} km..."
    )

    def _prog(msg: str, frac: float):
        typer.echo(f"[{int(frac*100):3d}%] {msg}")

    meta = create_snapshot(
        snapshot_id=snapshot_id,
        lat=lat,
        lon=lon,
        radius_km=radius,
        output_base=output,
        progress=_prog,
    )
    typer.echo(f"Snapshot '{snapshot_id}' created successfully!")
    typer.echo(f"Location: {meta['name']} | Checkpoints: {meta['checkpoints']}")


@snapshot_app.command("verify")
def snapshot_verify(
    base_dir: Path = typer.Option(
        Path("data/snapshots"), "--dir", help="Base directory of snapshots"
    ),
):
    """Verify cryptographic SHA-256 integrity of all snapshot files."""
    from greenplan.snapshots import verify_snapshots

    results = verify_snapshots(base_dir=base_dir)
    if not results:
        typer.echo("No snapshots found to verify.")
        return

    all_ok = True
    for snap_id, ok in results.items():
        status = "PASSED" if ok else "FAILED"
        if not ok:
            all_ok = False
        typer.echo(f"Snapshot '{snap_id}': {status}")

    if not all_ok:
        raise typer.Exit(code=1)


@snapshot_app.command("list")
def snapshot_list(
    base_dir: Path = typer.Option(
        Path("data/snapshots"), "--dir", help="Base directory of snapshots"
    ),
):
    """List all available snapshots and headline KPIs."""
    from greenplan.snapshots import list_snapshots

    snaps = list_snapshots(base_dir=base_dir)
    if not snaps:
        typer.echo("No snapshots available.")
        return

    for s in snaps:
        hd = s.get("headline", {})
        typer.echo(f"- [{s['snapshot_id']}] {s['name']} ({s['lat']:.4f}, {s['lon']:.4f})")
        typer.echo(
            f"  Green Cover: {hd.get('green_cover_pct_baseline')}% -> {hd.get('green_cover_pct_recent')}% | "
            f"Net Loss: {hd.get('net_green_loss_sqkm')} sq km"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Server host IP address"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Launch GreenPlan FastAPI backend server and dashboard UI."""
    import uvicorn

    typer.echo(f"Starting GreenPlan API & Dashboard server on http://{host}:{port} ...")
    uvicorn.run("greenplan.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
