"""
STEP 1b: Prepare Distance-to-Stream (FIXED — computes hydrology before district clipping)

Input:  data/processed/dem_merged.tif (the full, unclipped merged DEM tiles from step 1)
Output: data/processed/stream_distance.tif (cropped to match the district grid)

IMPORTANT DESIGN NOTE:
Flow accumulation must be computed on the FULL, unclipped DEM — not the
district-clipped one. If we clip first, the district boundary creates
artificial NaN edges that break flow routing (water "disappears" at the
boundary instead of continuing to flow), which starves accumulation values
near edges and produces a badly broken, overly sparse stream network.
(Boundry par NaN ke problem aa rhe the iss wagah se pahle calc then trim kiya hai)

So the order here is: reproject full merged DEM -> fill depressions ->
flow accumulation -> extract streams -> compute distance -> THEN crop/align
the final distance raster to match the district grid (same approach as
rainfall alignment in step 2).
"""

import os
import subprocess

import config
import numpy as np
import rasterio
import whitebox
from scipy.ndimage import distance_transform_edt


def main():
    print("=== STEP 1b: Prepare Distance-to-Stream ===")

    if not os.path.exists(config.DEM_MERGED_PATH):
        raise FileNotFoundError(
            f"{config.DEM_MERGED_PATH} not found. Run 01_prepare_dem.py first."
        )

    if not os.path.exists(config.DEM_CLIPPED_UTM):
        raise FileNotFoundError(
            f"{config.DEM_CLIPPED_UTM} not found. Run 01_prepare_dem.py first."
        )

    # Reproject the FULL merged DEM to UTM (no district clip yet)
    print("Reprojecting full merged DEM to UTM (unclipped)...")
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            config.UTM_EPSG,
            "-r",
            "bilinear",
            "-overwrite",
            config.DEM_MERGED_PATH,
            config.DEM_MERGED_UTM_PATH,
        ],
        check=True,
    )

    # Step B: Hydrology on the full extent
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.work_dir = config.PROCESSED_DIR

    filled_path = os.path.join(config.PROCESSED_DIR, "dem_merged_filled.tif")
    flow_accum_path = os.path.join(config.PROCESSED_DIR, "flow_accum_full.tif")

    print("Filling depressions on full extent...")
    wbt.fill_depressions(config.DEM_MERGED_UTM_PATH, filled_path)

    print("Computing D8 flow accumulation on full extent...")
    wbt.d8_flow_accumulation(filled_path, flow_accum_path, out_type="cells")

    # Threshold to extract streams (still full extent)
    with rasterio.open(flow_accum_path) as src:
        flow_accum = src.read(1).astype("float64")
        full_profile = src.profile
        transform = src.transform
        pixel_size = transform[0]
        nodata = src.nodata

    if nodata is not None:
        flow_accum[flow_accum == nodata] = 0
    flow_accum[np.isnan(flow_accum)] = 0

    total_cells = flow_accum.size
    stream_mask = flow_accum > config.STREAM_FLOW_ACCUM_THRESHOLD
    stream_pct = 100 * stream_mask.sum() / total_cells
    print(
        f"Stream pixels (full extent): {stream_mask.sum()} out of {total_cells} ({stream_pct:.2f}%)"
    )

    if stream_mask.sum() == 0:
        raise ValueError(
            "No stream pixels found. Lower STREAM_FLOW_ACCUM_THRESHOLD in config.py and re-run."
        )

    # Sanity check: a reasonable stream network in mountainous terrain is usually 1-5% of pixels.
    # If this is far outside that range, the threshold likely needs adjusting.
    if stream_pct < 0.3 or stream_pct > 8:
        print(
            f"WARNING: {stream_pct:.2f}% of pixels classified as stream is unusual. "
            f"Consider adjusting STREAM_FLOW_ACCUM_THRESHOLD (currently {config.STREAM_FLOW_ACCUM_THRESHOLD}) "
            "and re-running — lower threshold = denser network, higher = sparser."
        )

    # Step D: Distance transform on full extent
    inverted = ~stream_mask
    distance_pixels = distance_transform_edt(inverted)
    distance_meters = (distance_pixels * pixel_size).astype("float32")

    full_profile.update(dtype="float32", nodata=np.nan)
    full_distance_path = os.path.join(config.PROCESSED_DIR, "stream_distance_full.tif")
    with rasterio.open(full_distance_path, "w", **full_profile) as dst:
        dst.write(distance_meters, 1)

    print(
        f"Full-extent distance min/max: {distance_meters.min():.1f} / {distance_meters.max():.1f} m"
    )

    # Crop/align the distance raster to match the district grid exactly (Abb yaha pe hydro data ko district ke grid ke sath match kar rhe hai)
    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        target_bounds = src.bounds
        target_width = src.width
        target_height = src.height

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
            full_distance_path,
            config.STREAM_DISTANCE_PATH,
        ],
        check=True,
    )

    with rasterio.open(config.STREAM_DISTANCE_PATH) as src:
        final_data = src.read(1)
        print(
            f"Final cropped distance min/max: {np.nanmin(final_data):.1f} / {np.nanmax(final_data):.1f} m"
        )
        print(
            f"Final shape: {final_data.shape} (should match {target_height} x {target_width})"
        )

    print(f"Saved: {config.STREAM_DISTANCE_PATH}")
    print("=== STEP 1b COMPLETE ===\n")


if __name__ == "__main__":
    main()
