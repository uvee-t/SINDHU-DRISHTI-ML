"""
MASTER PIPELINE ORCHESTRATOR

Drop new raw data into data/raw/ subfolders, then run this single script.
It executes every processing step in order and produces:
  - data/outputs/red_zones_final.geojson
  - data/outputs/landslide_rf_model.pkl

Run with: python run_pipeline.py
"""

import os
import subprocess
import sys
from datetime import datetime

# ============================================================
# CONFIG: list every step in the order it must run
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")

PIPELINE_STEPS = [
    (
        "Prepare DEM (merge/clip/slope/aspect)",
        os.path.join(PIPELINE_DIR, "01_prepare_dem.py"),
    ),
    (
        "Prepare distance-to-stream",
        os.path.join(PIPELINE_DIR, "01b_prepare_stream_distance.py"),
    ),
    ("Prepare soil/geology", os.path.join(PIPELINE_DIR, "01c_prepare_soil.py")),
    (
        "Prepare rainfall (clip/reproject/sum)",
        os.path.join(PIPELINE_DIR, "02_prepare_rainfall.py"),
    ),
    (
        "Prepare landslide inventory points",
        os.path.join(PIPELINE_DIR, "03_prepare_landslide_points.py"),
    ),
    ("Build training dataset", os.path.join(PIPELINE_DIR, "04_build_training_data.py")),
    ("Train Random Forest model", os.path.join(PIPELINE_DIR, "05_train_model.py")),
    (
        "Generate district-wide red zone map",
        os.path.join(PIPELINE_DIR, "06_generate_redzone_map.py"),
    ),
    (
        "Prepare population + villages",
        os.path.join(PIPELINE_DIR, "07_prepare_population.py"),
    ),
    (
        "Village <-> red zone risk join",
        os.path.join(PIPELINE_DIR, "08_village_risk_join.py"),
    ),
    ("Prepare land use", os.path.join(PIPELINE_DIR, "13_prepare_landuse.py")),
    (
        "Carrying capacity for relocation sites",
        os.path.join(PIPELINE_DIR, "10_carrying_capacity.py"),
    ),
    ("Flood hazard (HAND)", os.path.join(PIPELINE_DIR, "12_flood_hazard.py")),
    ("Load outputs into PostGIS", os.path.join(PIPELINE_DIR, "11_load_postgis.py")),
]

LOG_FILE = os.path.join(PROJECT_ROOT, "pipeline_run_log.txt")


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_step(step_name, script_path):
    log(f"START: {step_name}")

    if not os.path.exists(script_path):
        log(f"ERROR: Script not found at {script_path}. Skipping.")
        return False

    result = subprocess.run(
        [sys.executable, script_path], capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    # Always print the script's own output so you can see progress/errors live
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        log(f"FAILED: {step_name} (exit code {result.returncode})")
        return False

    log(f"DONE: {step_name}")
    return True


def main():
    log("=" * 60)
    log("PIPELINE RUN STARTED")
    log("=" * 60)

    for step_name, script_path in PIPELINE_STEPS:
        success = run_step(step_name, script_path)
        if not success:
            log("Pipeline halted due to failure. Fix the error above and re-run.")
            sys.exit(1)

    log("=" * 60)
    log("PIPELINE COMPLETE — all steps succeeded")
    log("Final outputs:")
    log("  - data/outputs/red_zones_final.geojson")
    log("  - data/outputs/villages_with_risk.geojson")
    log("  - data/outputs/relocation_sites.geojson")
    log("  - data/outputs/landslide_rf_model.pkl")
    log("  - PostGIS tables: red_zones, villages, relocation_sites")
    log("=" * 60)


if __name__ == "__main__":
    main()
