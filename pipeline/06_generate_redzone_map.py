"""
STEP 6: Generate Red Zone Map

Input:  Trained model + feature rasters (from steps 1, 2, 5)
Output: data/outputs/red_zones_final.geojson, landslide_probability.tif, red_zone_classification.tif
"""

import os

import config
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes

FEATURE_COLS = config.FEATURE_COLS


def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        profile = src.profile
    return data, profile


def main():
    print("=== STEP 6: Generate Red Zone Map ===")

    for path in [
        config.MODEL_PATH,
        config.SLOPE_PATH,
        config.ASPECT_PATH,
        config.RAINFALL_TOTAL_PATH,
        config.DEM_CLIPPED_UTM,
        config.STREAM_DISTANCE_PATH,
        config.SOIL_CODE_PATH,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}. Run earlier steps first."
            )

    model = joblib.load(config.MODEL_PATH)

    slope, profile = load_raster(config.SLOPE_PATH)
    aspect, _ = load_raster(config.ASPECT_PATH)
    rainfall, _ = load_raster(config.RAINFALL_TOTAL_PATH)
    elevation, _ = load_raster(config.DEM_CLIPPED_UTM)
    stream_distance, _ = load_raster(config.STREAM_DISTANCE_PATH)
    soil_code, _ = load_raster(config.SOIL_CODE_PATH)

    assert (
        slope.shape
        == aspect.shape
        == rainfall.shape
        == elevation.shape
        == stream_distance.shape
        == soil_code.shape
    ), "Raster shapes don't match! Re-check alignment in steps 1, 1b, 1c, 2."

    valid_mask = (
        ~np.isnan(slope)
        & ~np.isnan(aspect)
        & ~np.isnan(rainfall)
        & ~np.isnan(elevation)
        & ~np.isnan(stream_distance)
        & (soil_code > 0)
        & (elevation > -1000)
    )
    print(f"Valid pixels: {valid_mask.sum()} out of {valid_mask.size}")

    rows, cols = np.where(valid_mask)
    X_predict = pd.DataFrame(
        np.column_stack(
            [
                slope[valid_mask],
                aspect[valid_mask],
                rainfall[valid_mask],
                elevation[valid_mask],
                stream_distance[valid_mask],
                soil_code[valid_mask],
            ]
        ),
        columns=FEATURE_COLS,
    )

    batch_size = 500_000
    probabilities = np.zeros(X_predict.shape[0], dtype="float32")
    for i in range(0, X_predict.shape[0], batch_size):
        batch = X_predict.iloc[i : i + batch_size]
        probabilities[i : i + batch_size] = model.predict_proba(batch)[:, 1]
        print(
            f"Predicted {min(i + batch_size, X_predict.shape[0])} / {X_predict.shape[0]} pixels"
        )

    prob_raster = np.full(slope.shape, np.nan, dtype="float32")
    prob_raster[rows, cols] = probabilities
    print(
        f"Probability min/max/mean: {np.nanmin(prob_raster):.3f} / {np.nanmax(prob_raster):.3f} / {np.nanmean(prob_raster):.3f}"
    )

    prob_profile = profile.copy()
    prob_profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(config.PROBABILITY_RASTER, "w", **prob_profile) as dst:
        dst.write(prob_raster, 1)

    # Classify zones
    zone_raster = np.full(slope.shape, 255, dtype="uint8")
    zone_raster[valid_mask & (prob_raster > config.RED_ZONE_THRESHOLD)] = 2
    zone_raster[
        valid_mask
        & (prob_raster > config.YELLOW_ZONE_THRESHOLD)
        & (prob_raster <= config.RED_ZONE_THRESHOLD)
    ] = 1
    zone_raster[valid_mask & (prob_raster <= config.YELLOW_ZONE_THRESHOLD)] = 0

    red = (zone_raster == 2).sum()
    yellow = (zone_raster == 1).sum()
    green = (zone_raster == 0).sum()
    total = red + yellow + green
    print(
        f"Red: {red} ({100 * red / total:.1f}%)  Yellow: {yellow} ({100 * yellow / total:.1f}%)  Green: {green} ({100 * green / total:.1f}%)"
    )

    zone_profile = profile.copy()
    zone_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(config.ZONE_RASTER, "w", **zone_profile) as dst:
        dst.write(zone_raster, 1)

    # Vectorize
    results = (
        {"properties": {"zone_class": int(v)}, "geometry": s}
        for s, v in shapes(
            zone_raster, mask=(zone_raster != 255), transform=profile["transform"]
        )
    )
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=profile["crs"])
    print(f"Raw vectorized polygons: {len(gdf)}")

    # Clean: filter small noise, simplify edges
    gdf["area_sqm"] = gdf.geometry.area
    gdf_filtered = gdf[gdf["area_sqm"] >= config.MIN_POLYGON_AREA_SQM].copy()
    gdf_filtered["geometry"] = gdf_filtered.geometry.simplify(
        config.SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    print(f"After area filter + simplify: {len(gdf_filtered)}")

    # Dissolve same-class polygons, then explode back into distinct regions
    dissolved = gdf_filtered.dissolve(by="zone_class", as_index=False)
    exploded = dissolved.explode(index_parts=False).reset_index(drop=True)
    exploded["area_sqkm"] = exploded.geometry.area / 1_000_000
    exploded = exploded[exploded["area_sqkm"] >= config.MIN_FINAL_ZONE_AREA_SQKM]

    exploded.to_file(config.RED_ZONES_FINAL_GEOJSON, driver="GeoJSON")
    print(f"Final polygon count: {len(exploded)}")
    print(exploded.groupby("zone_class")["area_sqkm"].agg(["count", "sum"]))
    print(f"Saved: {config.RED_ZONES_FINAL_GEOJSON}")
    print("=== STEP 6 COMPLETE ===\n")


if __name__ == "__main__":
    main()
