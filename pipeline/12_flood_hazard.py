"""
STEP 12: Flood Hazard via HAND (Height Above Nearest Drainage)

Input:  data/processed/dem_merged_filled.tif, flow_accum_full.tif
        (both already produced by 01b_prepare_stream_distance.py — reused, not recomputed)
Output: data/outputs/flood_zones_final.geojson

HAND measures, for every pixel, its elevation above the nearest stream cell along
the flow path. Low HAND = low-lying, close to drainage = flood-prone.
High HAND = elevated, far above drainage = safe from flooding.

This is a recognized rule-based method (no ML training needed), so this script
is much shorter than the landslide pipeline — classification thresholds come
from HAND literature rather than a trained model.
"""

import os
import subprocess
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import whitebox

import config


def main():
    print("=== STEP 12: Flood Hazard (HAND) ===")

    for path in [config.DEM_MERGED_FILLED_PATH, config.FLOW_ACCUM_FULL_PATH, config.DEM_CLIPPED_UTM]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}. Run 01_prepare_dem.py and "
                "01b_prepare_stream_distance.py first (this step reuses their output)."
            )

    # ---- Rebuild the stream raster from the already-computed flow accumulation ----
    # (cheap — just a threshold, not a recomputation of flow accumulation itself)
    with rasterio.open(config.FLOW_ACCUM_FULL_PATH) as src:
        flow_accum = src.read(1).astype("float64")
        full_profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        flow_accum[flow_accum == nodata] = 0
    flow_accum[np.isnan(flow_accum)] = 0

    stream_mask = (flow_accum > config.STREAM_FLOW_ACCUM_THRESHOLD).astype("uint8")
    print(f"Stream pixels: {stream_mask.sum()} out of {stream_mask.size}")

    stream_raster_path = os.path.join(config.PROCESSED_DIR, "stream_raster_full.tif")
    stream_profile = full_profile.copy()
    stream_profile.update(dtype="uint8", nodata=0)
    with rasterio.open(stream_raster_path, "w", **stream_profile) as dst:
        dst.write(stream_mask, 1)

    # ---- Compute HAND via WhiteboxTools ----
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.work_dir = config.PROCESSED_DIR

    _original_cwd = os.getcwd()

    print("Computing HAND (elevation above nearest stream)...")
    wbt.elevation_above_stream(
        config.DEM_MERGED_FILLED_PATH,
        stream_raster_path,
        config.HAND_FULL_PATH
    )
    os.chdir(_original_cwd)

    with rasterio.open(config.HAND_FULL_PATH) as src:
        hand_data = src.read(1)
        print(f"HAND (full extent) min/max: {np.nanmin(hand_data):.1f} / {np.nanmax(hand_data):.1f} m")

    # ---- Crop to district grid (same -te/-ts pattern used throughout this pipeline) ----
    with rasterio.open(config.DEM_CLIPPED_UTM) as src:
        target_bounds = src.bounds
        target_width = src.width
        target_height = src.height

    subprocess.run([
        "gdalwarp", "-t_srs", config.UTM_EPSG,
        "-te", str(target_bounds.left), str(target_bounds.bottom),
               str(target_bounds.right), str(target_bounds.top),
        "-ts", str(target_width), str(target_height),
        "-r", "bilinear", "-overwrite",
        config.HAND_FULL_PATH, config.HAND_CLIPPED_PATH
    ], check=True)

    with rasterio.open(config.HAND_CLIPPED_PATH) as src:
        hand = src.read(1).astype("float64")
        profile = src.profile
        nodata_val = src.nodata

    if nodata_val is not None:
        hand[hand == nodata_val] = np.nan

    print(f"HAND (district) min/max: {np.nanmin(hand):.1f} / {np.nanmax(hand):.1f} m")

    # ---- Classify flood risk ----
    valid_mask = ~np.isnan(hand)
    zone_raster = np.full(hand.shape, 255, dtype="uint8")
    zone_raster[valid_mask & (hand <= config.HAND_HIGH_RISK_MAX_M)] = 2       # high risk
    zone_raster[valid_mask & (hand > config.HAND_HIGH_RISK_MAX_M) &
                (hand <= config.HAND_MODERATE_RISK_MAX_M)] = 1               # moderate
    zone_raster[valid_mask & (hand > config.HAND_MODERATE_RISK_MAX_M)] = 0   # low risk

    high = (zone_raster == 2).sum()
    moderate = (zone_raster == 1).sum()
    low = (zone_raster == 0).sum()
    total = high + moderate + low
    print(f"High flood risk: {high} ({100*high/total:.1f}%)  "
          f"Moderate: {moderate} ({100*moderate/total:.1f}%)  "
          f"Low: {low} ({100*low/total:.1f}%)")

    # ---- Vectorize, clean, dissolve (same pattern as 06_generate_redzone_map.py) ----
    results = (
        {"properties": {"flood_zone_class": int(v)}, "geometry": s}
        for s, v in shapes(zone_raster, mask=(zone_raster != 255), transform=profile["transform"])
    )
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=profile["crs"])
    print(f"Raw vectorized polygons: {len(gdf)}")

    gdf["area_sqm"] = gdf.geometry.area
    gdf_filtered = gdf[gdf["area_sqm"] >= config.MIN_POLYGON_AREA_SQM].copy()
    gdf_filtered["geometry"] = gdf_filtered.geometry.simplify(config.SIMPLIFY_TOLERANCE, preserve_topology=True)

    dissolved = gdf_filtered.dissolve(by="flood_zone_class", as_index=False)
    exploded = dissolved.explode(index_parts=False).reset_index(drop=True)
    exploded["area_sqkm"] = exploded.geometry.area / 1_000_000
    exploded = exploded[exploded["area_sqkm"] >= config.MIN_FINAL_ZONE_AREA_SQKM]

    print(f"Final polygon count: {len(exploded)}")
    print(exploded.groupby("flood_zone_class")["area_sqkm"].agg(["count", "sum"]))

    exploded_wgs84 = exploded.to_crs(config.WGS84_EPSG)
    exploded_wgs84.to_file(config.FLOOD_ZONES_FINAL_GEOJSON, driver="GeoJSON")

    print(f"Saved: {config.FLOOD_ZONES_FINAL_GEOJSON}")
    print("flood_zone_class: 0=Low risk, 1=Moderate, 2=High risk")
    print("=== STEP 12 COMPLETE ===\n")


if __name__ == "__main__":
    main()