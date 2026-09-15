"""
STEP 1c: Prepare Soil / Geology (RASTER VERSION — for HWSD2 .bil data)

Input:  HWSD2.bil (+ .hdr, .prj, .stx companion files) in data/raw/soil/
Output: data/processed/soil_code.tif

HWSD2 pixel values are Soil Mapping Unit IDs (integer codes) — a categorical
variable identifying distinct soil units. We don't need to decode these to
real soil names for the ML model; the codes themselves are enough for
Random Forest to learn soil-based patterns.

CRITICAL: uses nearest-neighbor resampling (-r near), never bilinear —
soil codes are categories, not continuous values. Averaging two category
codes together (e.g. bilinear interpolating between code 4 and code 9)
would produce a meaningless code 6.5-ish value.
"""

import glob
import os
import subprocess

import config
import numpy as np
import rasterio


def find_soil_raster():
    candidates = glob.glob(os.path.join(config.SOIL_RAW_DIR, "*.bil")) + glob.glob(
        os.path.join(config.SOIL_RAW_DIR, "*.tif")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .bil or .tif soil raster found in {config.SOIL_RAW_DIR}. "
            "Make sure HWSD2.bil (with its .hdr/.prj/.stx companions) is there."
        )
    return candidates[0]


def main():
    print("=== STEP 1c: Prepare Soil / Geology (raster) ===")

    if not os.path.exists(config.DEM_CLIPPED_UTM):
        raise FileNotFoundError(
            f"{config.DEM_CLIPPED_UTM} not found. Run 01_prepare_dem.py first."
        )

    soil_raster_path = find_soil_raster()
    print(f"Using soil raster: {soil_raster_path}")

    # ---- Get exact target grid from DEM ----
    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        bounds = src.bounds
        width = src.width
        height = src.height

    print(f"Target grid: {width} x {height}, bounds={bounds}")

    # ---- Warp directly: reproject + crop + resample to DEM grid in one step ----
    # -r near is mandatory here since soil codes are categorical, not continuous
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            config.UTM_EPSG,
            "-te",
            str(bounds.left),
            str(bounds.bottom),
            str(bounds.right),
            str(bounds.top),
            "-ts",
            str(width),
            str(height),
            "-r",
            "near",
            "-overwrite",
            soil_raster_path,
            config.SOIL_CODE_PATH,
        ],
        check=True,
    )

    # ---- Sanity check the result ----
    with rasterio.open(config.SOIL_CODE_PATH) as src:
        soil_data = src.read(1)
        print(f"Soil raster shape: {soil_data.shape}")
        unique_vals, counts = np.unique(soil_data, return_counts=True)
        print(f"Unique soil codes found: {len(unique_vals)}")
        print("Sample value counts (top 10 by frequency):")
        sorted_idx = np.argsort(-counts)[:10]
        for i in sorted_idx:
            print(f"  Code {unique_vals[i]}: {counts[i]} pixels")

    print(f"Saved: {config.SOIL_CODE_PATH}")
    print("=== STEP 1c COMPLETE ===\n")


if __name__ == "__main__":
    main()
