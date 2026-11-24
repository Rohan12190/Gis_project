"""Inspection script for GreenPlan finished pipeline runs.

Reads metrics, summaries, and transition tables from an output directory
and prints a clear, structured peer-review audit table.

Usage:
    python scripts/check_run.py [--run-dir outputs/run_YYYYMMDD_HHMMSS]
"""

import argparse
import json
from pathlib import Path


def check_run(run_dir: Path) -> None:
    """Print structured summary and sanity verification for a finished pipeline run."""
    print("=" * 70)
    print(f"GREENPLAN RUN SANITY CHECK: {run_dir.name}")
    print("=" * 70)

    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        print(f"ERROR: summary.json not found in '{run_dir}'. Run may be incomplete.")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Provenance & Reference
    data_source = summary.get("data_source", "unknown")
    acc_ref = summary.get("accuracy_reference", "Not specified")
    print("\n[1] PROVENANCE & CALIBRATION")
    print(f"  Study Area:         {summary.get('study_area', 'N/A')}")
    print(f"  Data Source:        {data_source}")
    print(f"  Accuracy Reference: {acc_ref}")

    # 2. Accuracy Assessment
    acc_file = run_dir / "metrics" / "accuracy.json"
    if acc_file.exists():
        with open(acc_file, "r", encoding="utf-8") as f:
            acc_data = json.load(f)
        pixel_cv = acc_data.get("pixel_cv", {})
        oa = pixel_cv.get("overall_accuracy", pixel_cv.get("cv_overall_accuracy_mean", 0.0))
        kappa = pixel_cv.get("kappa", pixel_cv.get("cv_kappa_mean", 0.0))
        oa_std = pixel_cv.get("cv_overall_accuracy_std", 0.0)
        kappa_std = pixel_cv.get("cv_kappa_std", 0.0)

        print("\n[2] CLASSIFICATION ACCURACY")
        print(f"  Overall Accuracy:   {oa * 100:.2f}% (± {oa_std * 100:.2f}%)")
        print(f"  Cohen's Kappa:      {kappa:.3f} (± {kappa_std:.3f})")
        if "area_adjusted_sample" in acc_data:
            n_samp = acc_data["area_adjusted_sample"].get("sample_size", 0)
            print(f"  Olofsson Sample:    {n_samp} stratified reference points")

    # 3. Landscape Dynamics
    print("\n[3] 20-YEAR LANDSCAPE TRANSITIONS (2000 -> 2020)")
    print(f"  Built-up Growth:    {summary.get('builtup_growth_sqkm', 0.0):+.2f} km²")
    print(f"  Net Green Loss:     {summary.get('net_green_loss_sqkm', 0.0):+.2f} km²")
    print(f"  Green Baseline:     {summary.get('green_cover_pct_baseline', 0.0):.1f}% -> Recent: {summary.get('green_cover_pct_recent', 0.0):.1f}%")
    print(f"  Built-up Baseline:  {summary.get('builtup_pct_baseline', 0.0):.1f}% -> Recent: {summary.get('builtup_pct_recent', 0.0):.1f}%")
    print(f"  Farmland Baseline:  {summary.get('farmland_pct_baseline', 0.0):.1f}% -> Recent: {summary.get('farmland_pct_recent', 0.0):.1f}%")
    print(f"  Water Fraction:     {summary.get('water_pct', 0.0):.1f}%")

    # 4. Simulation & Model Validation (CA-Markov vs Null Model)
    sim_file = run_dir / "metrics" / "simulation.json"
    if sim_file.exists():
        with open(sim_file, "r", encoding="utf-8") as f:
            sim = json.load(f)
        fom_val = sim.get("figure_of_merit", 0.0)
        fom = fom_val.get("mean", 0.0) if isinstance(fom_val, dict) else float(fom_val)
        fom_std = fom_val.get("std", 0.0) if isinstance(fom_val, dict) else 0.0

        mod_agr_val = sim.get("model_agreement", 0.0)
        mod_agr = mod_agr_val.get("mean", 0.0) if isinstance(mod_agr_val, dict) else float(mod_agr_val)

        null_agr = float(sim.get("null_model_agreement", 0.0))
        beats_null = sim.get("beats_null_model", False)

        k_loc_val = sim.get("kappa_location", 0.0)
        k_loc = k_loc_val.get("mean", 0.0) if isinstance(k_loc_val, dict) else float(k_loc_val)

        k_quant_val = sim.get("kappa_quantity", 0.0)
        k_quant = k_quant_val.get("mean", 0.0) if isinstance(k_quant_val, dict) else float(k_quant_val)

        print("\n[4] CA-MARKOV SIMULATION VALIDATION")
        if fom_std > 0:
            print(f"  Figure of Merit:    {fom * 100:.2f}% (± {fom_std * 100:.2f}%)")
        else:
            print(f"  Figure of Merit:    {fom * 100:.2f}%")
        print(f"  Model Agreement:    {mod_agr * 100:.2f}%")
        print(f"  Null Agreement:     {null_agr * 100:.2f}% (Persistence benchmark)")
        print(f"  Beats Null Model:   {'YES' if beats_null else 'NO'}")
        print(f"  Pontius K_loc:      {k_loc:.3f} | K_quant: {k_quant:.3f}")

    # 5. Thermal Environment & UHI
    therm_file = run_dir / "metrics" / "thermal.json"
    if therm_file.exists():
        with open(therm_file, "r", encoding="utf-8") as f:
            therm = json.load(f)
        r2 = therm.get("r_squared", 0.0)
        moran = therm.get("morans_i", 0.0)
        print("\n[5] THERMAL (LST) & CONSERVATION")
        print(f"  Mean Surface Temp:  {summary.get('mean_lst_celsius_recent', 0.0):.1f} °C")
        print(f"  LST Regression R²:  {r2:.3f}")
        print(f"  LST Moran's I:      {moran:.3f} (Spatial autocorrelation)")
        print(f"  High Priority Zone: {summary.get('high_priority_conservation_sqkm', 0.0):.2f} km²")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and audit GreenPlan run metrics.")
    parser.add_argument("--run-dir", default=None, help="Path to outputs/run_* directory. Defaults to latest.")
    args = parser.parse_args()

    if args.run_dir:
        run_path = Path(args.run_dir)
    else:
        runs = sorted(list(Path("outputs").glob("run_*")))
        if not runs:
            print("ERROR: No run directories found in outputs/.")
            return
        run_path = runs[-1]

    check_run(run_path)


if __name__ == "__main__":
    main()
