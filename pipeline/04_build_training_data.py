"""
STEP 4: Build Training Dataset

Input:  Processed slope/aspect/rainfall/DEM rasters + landslide points (from steps 1-3)
Output: data/processed/training_data.csv

No new manual input needed here — this step is fully automatic given steps 1-3 ran.
"""

import glob
import os

import config
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point


def find_boundary_file():
    candidates = glob.glob(
        os.path.join(config.BOUNDARY_RAW_DIR, "*.geojson")
    ) + glob.glob(os.path.join(config.BOUNDARY_RAW_DIR, "*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No boundary file found in {config.BOUNDARY_RAW_DIR}")
    return candidates[0]


def sample_raster_at_points(raster_path, points_gdf, column_name):
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
        values = [val[0] for val in src.sample(coords)]
    points_gdf[column_name] = values
    return points_gdf


def main():
    print("=== STEP 4: Build Training Dataset ===")

    required = [
        config.SLOPE_PATH,
        config.ASPECT_PATH,
        config.RAINFALL_TOTAL_PATH,
        config.DEM_CLIPPED_UTM,
        config.LANDSLIDE_POINTS_UTM,
        config.STREAM_DISTANCE_PATH,
        config.SOIL_CODE_PATH,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}. Run earlier steps first."
            )

    # ---- Load positive (landslide) points ----
    gdf_utm = gpd.read_file(config.LANDSLIDE_POINTS_UTM)
    gdf_utm = sample_raster_at_points(config.SLOPE_PATH, gdf_utm, "slope")
    gdf_utm = sample_raster_at_points(config.ASPECT_PATH, gdf_utm, "aspect")
    gdf_utm = sample_raster_at_points(config.RAINFALL_TOTAL_PATH, gdf_utm, "rainfall")
    gdf_utm = sample_raster_at_points(config.DEM_CLIPPED_UTM, gdf_utm, "elevation")
    gdf_utm = sample_raster_at_points(
        config.STREAM_DISTANCE_PATH, gdf_utm, "stream_distance"
    )
    gdf_utm = sample_raster_at_points(config.SOIL_CODE_PATH, gdf_utm, "soil_code")

    gdf_utm = gdf_utm[
        (~gdf_utm["slope"].isna())
        & (~gdf_utm["rainfall"].isna())
        & (~gdf_utm["stream_distance"].isna())
        & (~gdf_utm["soil_code"].isna())
        & (gdf_utm["slope"] > -1e10)
        & (gdf_utm["rainfall"] > -1e10)
        & (gdf_utm["soil_code"] > 0)  # 0 = no soil data assigned, drop those points
    ]
    print(f"Valid positive samples: {len(gdf_utm)}")

    # ---- Generate negative (background) points ----
    boundary = gpd.read_file(find_boundary_file()).to_crs(config.UTM_EPSG)
    boundary_geom = (
        boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    )
    minx, miny, maxx, maxy = boundary.total_bounds

    n_negative = len(gdf_utm) * 2
    np.random.seed(42)
    negative_points = []
    attempts, max_attempts = 0, n_negative * 20

    while len(negative_points) < n_negative and attempts < max_attempts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        pt = Point(x, y)
        if boundary_geom.contains(pt):
            negative_points.append(pt)
        attempts += 1

    print(f"Generated {len(negative_points)} random background points")

    neg_gdf = gpd.GeoDataFrame(geometry=negative_points, crs=config.UTM_EPSG)

    landslide_buffer = gdf_utm.geometry.buffer(100)
    landslide_buffer = (
        landslide_buffer.union_all()
        if hasattr(landslide_buffer, "union_all")
        else landslide_buffer.unary_union
    )
    neg_gdf = neg_gdf[~neg_gdf.geometry.within(landslide_buffer)].reset_index(drop=True)
    print(f"Background points remaining after buffer exclusion: {len(neg_gdf)}")

    neg_gdf = sample_raster_at_points(config.SLOPE_PATH, neg_gdf, "slope")
    neg_gdf = sample_raster_at_points(config.ASPECT_PATH, neg_gdf, "aspect")
    neg_gdf = sample_raster_at_points(config.RAINFALL_TOTAL_PATH, neg_gdf, "rainfall")
    neg_gdf = sample_raster_at_points(config.DEM_CLIPPED_UTM, neg_gdf, "elevation")
    neg_gdf = sample_raster_at_points(
        config.STREAM_DISTANCE_PATH, neg_gdf, "stream_distance"
    )
    neg_gdf = sample_raster_at_points(config.SOIL_CODE_PATH, neg_gdf, "soil_code")

    neg_gdf = neg_gdf[
        (~neg_gdf["slope"].isna())
        & (~neg_gdf["rainfall"].isna())
        & (~neg_gdf["stream_distance"].isna())
        & (~neg_gdf["soil_code"].isna())
        & (neg_gdf["slope"] > -1e10)
        & (neg_gdf["rainfall"] > -1e10)
        & (neg_gdf["soil_code"] > 0)
    ]
    print(f"Valid negative samples: {len(neg_gdf)}")

    # ---- Combine, shuffle, save ----
    pos_df = gdf_utm[config.FEATURE_COLS].copy()
    pos_df["label"] = 1
    neg_df = neg_gdf[config.FEATURE_COLS].copy()
    neg_df["label"] = 0

    training_data = pd.concat([pos_df, neg_df], ignore_index=True)
    training_data = training_data.dropna()

    # Shuffle so downstream cross-validation folds aren't accidentally grouped by class
    training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(training_data)}")
    print(training_data["label"].value_counts())

    training_data.to_csv(config.TRAINING_DATA_CSV, index=False)
    print(f"Saved: {config.TRAINING_DATA_CSV}")
    print("=== STEP 4 COMPLETE ===\n")


if __name__ == "__main__":
    main()
