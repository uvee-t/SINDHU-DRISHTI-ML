"""
STEP 2: Prepare Rainfall

Input:  All monthly CHIRPS .tif files in data/raw/rainfall/, boundary file, and the
        already-processed dem_utm.tif (must run 01_prepare_dem.py first)
Output: data/processed/total_monsoon_rainfall.tif

Drop any number of monthly rainfall .tif files into data/raw/rainfall/ — they will
all be clipped, aligned to the DEM grid, and summed automatically.
"""

import glob
import os
import subprocess

import config
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def find_boundary_file():
    candidates = glob.glob(
        os.path.join(config.BOUNDARY_RAW_DIR, "*.geojson")
    ) + glob.glob(os.path.join(config.BOUNDARY_RAW_DIR, "*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No boundary file found in {config.BOUNDARY_RAW_DIR}")
    return candidates[0]


def main():
    print("=== STEP 2: Prepare Rainfall ===")

    if not os.path.exists(config.DEM_CLIPPED_UTM):
        raise FileNotFoundError(
            f"{config.DEM_CLIPPED_UTM} not found. Run 01_prepare_dem.py first — "
            "rainfall must align to the DEM grid."
        )

    # ---- Get exact target grid from the DEM (so rainfall aligns pixel-for-pixel) ----
    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        target_bounds = src.bounds
        target_width = src.width
        target_height = src.height
    print(
        f"Target grid from DEM: {target_width} x {target_height}, bounds={target_bounds}"
    )

    # ---- Find all rainfall files ----
    rainfall_files = sorted(glob.glob(os.path.join(config.RAINFALL_RAW_DIR, "*.tif")))
    if not rainfall_files:
        raise FileNotFoundError(
            f"No rainfall .tif files found in {config.RAINFALL_RAW_DIR}"
        )
    print(
        f"Found {len(rainfall_files)} rainfall files: {[os.path.basename(f) for f in rainfall_files]}"
    )

    boundary_path = find_boundary_file()
    boundary = gpd.read_file(boundary_path)

    aligned_paths = []

    for i, fpath in enumerate(rainfall_files):
        fname = os.path.splitext(os.path.basename(fpath))[0]

        # Clip with explicit nodata (avoids the "outside boundary = fake 0" bug)
        with rasterio.open(fpath) as src:
            boundary_reproj = boundary.to_crs(src.crs)
            out_image, out_transform = mask(
                src, boundary_reproj.geometry, crop=True, nodata=-9999, filled=True
            )
            out_meta = src.meta.copy()

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": -9999,
            }
        )
        clipped_path = os.path.join(
            config.PROCESSED_DIR, f"rainfall_{fname}_clipped.tif"
        )
        with rasterio.open(clipped_path, "w", **out_meta) as dst:
            dst.write(out_image)

        # Reproject + align exactly to DEM grid using -te / -ts (not just -tr)
        aligned_path = os.path.join(config.PROCESSED_DIR, f"rainfall_{fname}_utm.tif")
        subprocess.run(
            [
                "gdalwarp",
                "-t_srs",
                config.UTM_EPSG,
                "-te",
                str(target_bounds.left),
                str(target_bounds.bottom),
                str(target_bounds.right),
                str(target_bounds.top),
                "-ts",
                str(target_width),
                str(target_height),
                "-r",
                "bilinear",
                "-overwrite",
                "-dstnodata",
                "-9999",
                clipped_path,
                aligned_path,
            ],
            check=True,
        )

        aligned_paths.append(aligned_path)
        print(f"Processed rainfall file {i + 1}/{len(rainfall_files)}: {fname}")

    # ---- Sum all aligned rainfall rasters ----
    arrays = []
    profile = None
    for f in aligned_paths:
        with rasterio.open(f) as src:
            data = src.read(1).astype("float64")
            data[data == -9999] = np.nan
            arrays.append(data)
            if profile is None:
                profile = src.profile

    stacked = np.stack(arrays)
    valid_mask = ~np.all(np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    total[~valid_mask] = np.nan

    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(config.RAINFALL_TOTAL_PATH, "w", **profile) as dst:
        dst.write(total.astype("float32"), 1)

    print(f"Rainfall total min/max: {np.nanmin(total):.1f} / {np.nanmax(total):.1f}")
    print(f"Saved: {config.RAINFALL_TOTAL_PATH}")
    print("=== STEP 2 COMPLETE ===\n")


if __name__ == "__main__":
    main()
