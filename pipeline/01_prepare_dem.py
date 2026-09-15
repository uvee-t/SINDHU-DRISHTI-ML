"""
STEP 1: Prepare DEM

Input:  All .hgt tiles in data/raw/dem/, boundary file in data/raw/boundary/
Output: data/processed/slope.tif, data/processed/aspect.tif, data/processed/dem_utm.tif

Drop any number of .hgt tiles into data/raw/dem/ and this will merge all of them
automatically — no need to hardcode tile names.
"""

import glob
import os
import subprocess

import config
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge


def find_boundary_file():
    candidates = glob.glob(
        os.path.join(config.BOUNDARY_RAW_DIR, "*.geojson")
    ) + glob.glob(os.path.join(config.BOUNDARY_RAW_DIR, "*.shp"))
    if not candidates:
        raise FileNotFoundError(
            f"No boundary file found in {config.BOUNDARY_RAW_DIR}. "
            "Put a district boundary .geojson or .shp there first."
        )
    return candidates[0]


def main():
    print("=== STEP 1: Prepare DEM ===")

    # Find all DEM tiles (/pipeline/data/raw/dem/*.hgt)
    tile_paths = glob.glob(os.path.join(config.DEM_RAW_DIR, "*.hgt"))
    if not tile_paths:
        raise FileNotFoundError(f"No .hgt tiles found in {config.DEM_RAW_DIR}")
    print(
        f"Found {len(tile_paths)} DEM tiles: {[os.path.basename(t) for t in tile_paths]}"
    )

    # Merge tiles (Saare dem files ko ek single raster me merge karna)
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
    merged_path = config.DEM_MERGED_PATH
    with rasterio.open(merged_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    for f in src_files:
        f.close()
    print(f"Merged DEM saved: {merged_path}, shape={mosaic.shape}")

    # ---- Clip to district boundary ----
    boundary_path = find_boundary_file()
    boundary = gpd.read_file(boundary_path)
    print(f"Using boundary file: {boundary_path}")

    with rasterio.open(merged_path) as src:
        boundary_reproj = boundary.to_crs(src.crs)
        out_image, out_transform = mask(src, boundary_reproj.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    clipped_path = os.path.join(config.PROCESSED_DIR, "dem_clipped.tif")
    with rasterio.open(clipped_path, "w", **out_meta) as dst:
        dst.write(out_image)
    print(f"Clipped DEM saved: {clipped_path}")

    # ---- Reproject to UTM ----
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            config.UTM_EPSG,
            "-r",
            "bilinear",
            "-overwrite",
            clipped_path,
            config.DEM_CLIPPED_UTM,
        ],
        check=True,
    )
    print(f"Reprojected DEM saved: {config.DEM_CLIPPED_UTM}")

    # ---- Calculate slope and aspect (Merged dem files se slope calc kiya hai) ----
    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        pixel_size = transform[0]
        profile = src.profile
        nodata_val = src.nodata

    if nodata_val is not None:
        dem[dem == nodata_val] = np.nan
    dem[dem <= -1000] = np.nan

    dx, dy = np.gradient(dem, pixel_size)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect_deg = np.degrees(np.arctan2(-dx, dy))
    aspect_deg = np.where(aspect_deg < 0, 90 - aspect_deg, 360 - aspect_deg + 90) % 360

    print(f"Elevation min/max: {np.nanmin(dem):.1f} / {np.nanmax(dem):.1f}")
    print(f"Slope min/max: {np.nanmin(slope_deg):.1f} / {np.nanmax(slope_deg):.1f}")
    print(f"Aspect min/max: {np.nanmin(aspect_deg):.1f} / {np.nanmax(aspect_deg):.1f}")

    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(config.SLOPE_PATH, "w", **profile) as dst:
        dst.write(slope_deg.astype("float32"), 1)
    with rasterio.open(config.ASPECT_PATH, "w", **profile) as dst:
        dst.write(aspect_deg.astype("float32"), 1)

    print(f"Saved: {config.SLOPE_PATH}")
    print(f"Saved: {config.ASPECT_PATH}")
    print("=== STEP 1 COMPLETE ===\n")


if __name__ == "__main__":
    main()
