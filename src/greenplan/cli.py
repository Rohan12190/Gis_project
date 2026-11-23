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
def imagery(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 2: Prepare multi-epoch imagery composites & indices."""
    typer.echo("Executing Phase 2: Landsat imagery compositing & index calculation...")


@app.command()
def classify(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 3: Random Forest multi-epoch LULC classification & validation."""
    typer.echo("Executing Phase 3: Random Forest classification...")


@app.command()
def drivers(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 4: Spatial driver extraction and suitability modeling."""
    typer.echo("Executing Phase 4: Driver layers and suitability modeling...")


@app.command()
def thermal(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 5: Land Surface Temperature regression and spatial diagnostics."""
    typer.echo("Executing Phase 5: LST regression and hotspot modeling...")


@app.command()
def simulate(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 6: CA-Markov urban growth simulation and scenario comparison."""
    typer.echo("Executing Phase 6: CA-Markov simulation...")


@app.command()
def plan(config: Path = typer.Option("configs/pipeline.yaml", help="Path to config YAML")):
    """Phase 7: Bivariate risk/heat planning map and recommendation matrix."""
    typer.echo("Executing Phase 7: Generating planning map & decision matrix...")


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
    import os
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

    os.chdir(target_dir)

    class GreenPlanRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/predict_gee":
                query = parse_qs(parsed.query)
                lat = float(query.get("lat", [23.18])[0])
                lon = float(query.get("lon", [75.77])[0])
                city = query.get("city", ["Target Area"])[0]
                radius = float(query.get("radius", [8.0])[0])
                year = int(query.get("year", [2020])[0])

                from greenplan.gee.landsat import predict_location_gee

                result = predict_location_gee(
                    lat=lat, lon=lon, city_name=city, buffer_km=radius, year=year
                )
                payload = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            super().do_GET()

    with socketserver.TCPServer(("", port), GreenPlanRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nServer stopped.")


@app.command()
def gee_predict(
    city: str = typer.Option("Ujjain", help="Name of target city / location"),
    lat: float = typer.Option(23.18, help="Latitude"),
    lon: float = typer.Option(75.77, help="Longitude"),
    radius: float = typer.Option(8.0, help="Buffer radius in km"),
    year: int = typer.Option(2020, help="Landsat epoch year"),
):
    """Stream live Landsat satellite imagery from Google Earth Engine and run ML prediction for ANY place."""
    from greenplan.gee.landsat import predict_location_gee

    typer.echo(f"Connecting to Google Earth Engine for {city} [{lat}, {lon}]...")
    res = predict_location_gee(lat=lat, lon=lon, city_name=city, buffer_km=radius, year=year)
    typer.echo("==================================================")
    typer.echo(f"Earth Engine Live ML Prediction: {city.upper()}")
    typer.echo("==================================================")
    typer.echo(f"Satellite Scene : {res['scene_id']} ({res['sensor']})")
    typer.echo(f"Acquisition Date: {res['acquisition_date']} (Cloud: {res['cloud_cover']}%)")
    typer.echo(f"Green Canopy    : {res['green_cover_pct']}%")
    typer.echo(f"Built-up Concrete: {res['builtup_pct']}%")
    typer.echo(f"Farmland Area   : {res['farmland_pct']}%")
    typer.echo(f"Surface Temp LST: {res['mean_lst_celsius']} °C")
    typer.echo(f"Priority Deficit: {res['priority_planting_sqkm']} km²")
    typer.echo(f"Recommendation  : {res['recommendation']}")
    typer.echo("==================================================")


@app.command()
def gee_auth(
    login: bool = typer.Option(
        False, "--login", help="Launch OAuth2 browser authentication with your Google account"
    ),
):
    """Instructions, status check, and browser OAuth2 login for Google Earth Engine."""
    from greenplan.gee.landsat import GEEClient

    client = GEEClient()
    if login:
        import ee

        typer.echo(
            f"Launching Google Earth Engine browser authentication for project: {client.project_id}..."
        )
        ee.Authenticate(auth_mode="localhost")
        try:
            ee.Initialize(project=client.project_id)
            typer.echo(
                f"Successfully authenticated and initialized Earth Engine with project: {client.project_id}!"
            )
        except Exception as e:
            typer.echo(f"Authentication token saved. Note on initialization: {e}")

    st = client.get_status()
    typer.echo("==================================================")
    typer.echo("Google Earth Engine Environment Configuration (.env)")
    typer.echo("==================================================")
    typer.echo(f"Active Project ID : {st['project_id']}")
    typer.echo(f"Auth Credential   : {st['auth_mode']}")
    typer.echo(f"Cloud Endpoint    : {st['endpoint_url']}")
    typer.echo("--------------------------------------------------")
    typer.echo("How to configure your custom credentials:")
    typer.echo("1. Edit '.env' in this repository.")
    typer.echo(f"   EE_PROJECT_ID={st['project_id']}")
    typer.echo("2. To authenticate with your Google/Gmail account, run:")
    typer.echo("   uv run greenplan gee-auth --login")
    typer.echo("   or: uv run earthengine authenticate")
    typer.echo("==================================================")


@app.command()
def gee_fetch(
    city: str = typer.Option(
        "bhopal", help="Target city / region (bhopal, sehore, kothri, indore)"
    ),
    project_id: Optional[str] = typer.Option(None, help="Google Earth Engine Cloud Project ID"),
    api_key: Optional[str] = typer.Option(None, help="GEE API Key or OAuth2 Bearer Token"),
):
    """Query and verify Google Earth Engine (GEE) Landsat data ingestion route."""
    from greenplan.gee.landsat import GEEClient

    client = GEEClient(project_id=project_id, api_key=api_key)
    status = client.get_status()
    typer.echo("==================================================")
    typer.echo("Google Earth Engine (GEE) Ingestion Route & Status")
    typer.echo("==================================================")
    typer.echo(f"Endpoint URL  : {status['endpoint_url']}")
    typer.echo(f"Project ID    : {status['project_id']}")
    typer.echo(f"API Key       : {status['api_key']}")
    typer.echo(f"Cloud Status  : {status['status']}")
    typer.echo(f"Target City   : {city.capitalize()}")
    typer.echo(
        f"WRS-2 Orbit   : {status['regional_orbits'].get(city.lower(), 'Path 146 / Row 044')}"
    )
    typer.echo("Active Collections:")
    for c in status["collections"]:
        typer.echo(f"  - {c}")
    typer.echo("--------------------------------------------------")
    typer.echo(f"Querying scene catalog for {city.capitalize()}...")
    meta = client.fetch_scene_metadata(city.lower())
    for yr, sc in meta.items():
        typer.echo(
            f"  Epoch {yr}: Scene {sc['scene_id']} | Date: {sc['acquisition_date']} | "
            f"Cloud: {sc['cloud_cover']}% | Sensor: {sc['sensor']}"
        )
    typer.echo("==================================================")
    typer.echo("GEE route verification successful.")


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
    city: Optional[str] = typer.Option(
        None, help="Shortcut city name: bhopal, sehore, kothri, indore"
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
    run_dir = run_pipeline(config)
    typer.echo(f"Run completed successfully! Artifacts written to: {run_dir}")
    typer.echo(f"View report at: {run_dir / 'report' / 'dashboard.html'}")
    typer.echo("Or launch server: uv run greenplan serve")


if __name__ == "__main__":
    app()
