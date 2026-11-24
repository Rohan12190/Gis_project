"""Command-line interface for GreenPlan."""

from pathlib import Path
from typing import Optional

import typer

from greenplan.config import load_pipeline_config

app = typer.Typer(help="GreenPlan: Urban Growth & Green Space Planning CLI")


@app.command()
def validate_config(
    config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML"),
):
    """Validate pipeline configuration against Pydantic schema."""
    cfg = load_pipeline_config(config)
    typer.echo(
        f"Configuration valid! Project: {cfg.project_name}, Study Area: {cfg.study_area.name}"
    )
    typer.echo(f"Epochs: {cfg.epochs}, Projection Years: {cfg.projection_years}")
    typer.echo(f"Classes: {[c.name for c in cfg.class_scheme]}")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to run HTTP server on"),
    run_dir: Optional[str] = typer.Option(
        None, help="Specific run directory to serve, defaults to latest run"
    ),
):
    """Serve interactive GreenPlan report dashboard on local web server."""
    import http.server
    import json
    import socketserver
    from urllib.parse import parse_qs, urlparse

    runs = sorted(list(Path("outputs").glob("run_*")))
    if not runs:
        typer.echo("No runs found in outputs/! Run 'greenplan run' first.")
        raise typer.Exit(1)

    target_dir = Path(run_dir) if run_dir else runs[-1]
    report_file = target_dir / "report" / "dashboard.html"

    if not report_file.exists():
        from greenplan.viewer import generate_html_report

        generate_html_report(target_dir)

    typer.echo(
        f"Serving GreenPlan Dashboard for {target_dir.name} at http://localhost:{port}/report/dashboard.html"
    )
    typer.echo("Press Ctrl+C to stop the server.")

    class GreenPlanRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(target_dir), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/predict_gee":
                query = parse_qs(parsed.query)
                city = query.get("city", ["Indore"])[0]
                lat = float(query.get("lat", [22.7196])[0])
                lon = float(query.get("lon", [75.8577])[0])
                radius = float(query.get("radius", [7.0])[0])

                from greenplan.gee.landsat import predict_location_gee

                result = predict_location_gee(lat, lon, city, buffer_km=radius)
                response_bytes = json.dumps(result).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
                return

            super().do_GET()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), GreenPlanRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nServer stopped.")


@app.command()
def verify_gee(
    project_id: Optional[str] = typer.Option(
        None, help="Google Cloud project ID with Earth Engine API enabled"
    ),
    aoi: str = typer.Option("bhopal", help="AOI name: bhopal, sehore, kothri, indore"),
):
    """Verify Google Earth Engine connection, credentials, and catalog access."""
    from greenplan.gee.landsat import GEEClient

    client = GEEClient(project_id=project_id)
    status = client.get_status()

    typer.echo("================ GEE Status ================")
    typer.echo(f"Status:       {status['status']}")
    typer.echo(f"Auth Mode:    {status['auth_mode']}")
    typer.echo(f"Project ID:   {status['project_id']}")
    typer.echo(f"Endpoint:     {status['endpoint_url']}")
    typer.echo(f"Collections:  {', '.join(status['collections'])}")
    typer.echo(f"Cloud Mask:   {status['cloud_masking']}")
    typer.echo(f"Harmonize:    {status['harmonization']}")
    typer.echo("--------------------------------------------")
    typer.echo(f"Querying USGS Landsat Catalog for AOI '{aoi}'...")
    scenes = client.fetch_scene_metadata(aoi)
    for yr, sc in scenes.items():
        typer.echo(
            f"  Epoch {yr}: Scene {sc['scene_id']} | Date: {sc['acquisition_date']} | "
            f"Cloud: {sc['cloud_cover']}% | Sensor: {sc['sensor']}"
        )
    typer.echo("==================================================")


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
    city: Optional[str] = typer.Option(
        None, help="Shortcut city name: bhopal, sehore, kothri, indore"
    ),
    synthetic: bool = typer.Option(
        False, help="Run in explicit synthetic demo mode for software verification"
    ),
):
    """Execute end-to-end GreenPlan pipeline from Phase 2 through Phase 7."""
    from greenplan.pipeline import run_pipeline

    if config is None:
        if city:
            city_lower = city.lower()
            city_configs = {
                "bhopal": Path("configs/pipeline.yaml"),
                "sehore": Path("configs/pipeline_sehore.yaml"),
                "kothri": Path("configs/pipeline_kothri.yaml"),
                "ashta": Path("configs/pipeline_kothri.yaml"),
                "indore": Path("configs/pipeline_indore.yaml"),
            }
            config = city_configs.get(city_lower, Path("configs/pipeline.yaml"))
        else:
            config = Path("configs/pipeline.yaml")

    cfg = load_pipeline_config(config)
    typer.echo(
        f"Starting end-to-end GreenPlan execution for {cfg.study_area.name} (Config: {config})..."
    )
    run_dir = run_pipeline(config, synthetic_mode=synthetic)
    typer.echo(f"Run completed successfully! Artifacts written to: {run_dir}")
    typer.echo(f"View report at: {run_dir / 'report' / 'dashboard.html'}")
    typer.echo("Or launch server: uv run greenplan serve")


if __name__ == "__main__":
    app()
