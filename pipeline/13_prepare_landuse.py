"""
STEP 13: Prepare Land Use

Input:  One or more ESA WorldCover GeoTIFF tiles in data/raw/landuse/
        (Darjeeling straddles a tile boundary, so 2 tiles is normal/expected —
        this script merges however many it finds automatically)
Output: data/processed/landuse.tif

Aligns land cover classification to the DEM grid using nearest-neighbor
resampling (categorical data — same reasoning as the soil step).
"""

import glob
import os
import subprocess

import config
import numpy as np
import rasterio
from rasterio.merge import merge


def find_landuse_rasters():
    candidates = glob.glob(os.path.join(config.LANDUSE_RAW_DIR, "*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No land use .tif found in {config.LANDUSE_RAW_DIR}. "
            "Download from https://esa-worldcover.org/en/data-access and place it there."
        )
    return candidates


def main():
    print("=== STEP 13: Prepare Land Use ===")

    if not os.path.exists(config.DEM_CLIPPED_UTM):
        raise FileNotFoundError(
            f"{config.DEM_CLIPPED_UTM} not found. Run 01_prepare_dem.py first."
        )

    tile_paths = find_landuse_rasters()
    print(
        f"Found {len(tile_paths)} land use tile(s): {[os.path.basename(t) for t in tile_paths]}"
    )

    if len(tile_paths) > 1:
        src_files = [rasterio.open(fp) for fp in tile_paths]
        mosaic, out_transform = merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
            }
        )
        merged_path = os.path.join(config.PROCESSED_DIR, "landuse_merged.tif")
        with rasterio.open(merged_path, "w", **out_meta) as dst:
            dst.write(mosaic)
        for f in src_files:
            f.close()
        print(f"Merged {len(tile_paths)} tiles: {merged_path}")
        landuse_path = merged_path
    else:
        landuse_path = tile_paths[0]

    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        bounds = src.bounds
        width = src.width
        height = src.height

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
            "near",  # categorical data — never bilinear
            "-overwrite",
            landuse_path,
            config.LANDUSE_PATH,
        ],
        check=True,
    )

    with rasterio.open(config.LANDUSE_PATH) as src:
        data = src.read(1)
        unique_vals, counts = np.unique(data, return_counts=True)
        print("\nLand cover class distribution:")
        class_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare/sparse veg",
            70: "Snow/ice",
            80: "Water",
            90: "Wetland",
            95: "Mangroves",
            100: "Moss/lichen",
        }
        for val, count in sorted(zip(unique_vals, counts), key=lambda x: -x[1]):
            pct = 100 * count / data.size
            name = class_names.get(int(val), f"Unknown ({val})")
            print(f"  {name}: {count} pixels ({pct:.1f}%)")

    print(f"\nSaved: {config.LANDUSE_PATH}")
    print("=== STEP 13 COMPLETE ===\n")


if __name__ == "__main__":
    main()
