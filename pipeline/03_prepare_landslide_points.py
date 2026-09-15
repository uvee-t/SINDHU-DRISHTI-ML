"""
STEP 3: Prepare Landslide Inventory Points

Input:  A landslide inventory .csv or .xlsx file in data/raw/landslide_inventory/
Output: data/processed/landslide_points_utm.geojson

Drop your landslide inventory spreadsheet into data/raw/landslide_inventory/.
This script auto-detects .csv or .xlsx, cleans it, filters to the target district,
and reprojects to UTM.
"""

import glob
import os

import config
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def find_inventory_file():
    candidates = glob.glob(os.path.join(config.LANDSLIDE_RAW_DIR, "*.csv")) + glob.glob(
        os.path.join(config.LANDSLIDE_RAW_DIR, "*.xlsx")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .csv or .xlsx file found in {config.LANDSLIDE_RAW_DIR}"
        )
    return candidates[0]


def main():
    print("=== STEP 3: Prepare Landslide Points ===")

    inventory_path = find_inventory_file()
    print(f"Using inventory file: {inventory_path}")

    if inventory_path.endswith(".csv"):
        df = pd.read_csv(inventory_path)
    else:
        df = pd.read_excel(inventory_path)

    print(f"Total rows loaded: {len(df)}")

    # Drop duplicated header row if present (common in govt exports)
    if (
        "Sl.No." in df.columns
        and len(df) > 0
        and str(df.iloc[0].get("Sl.No.", "")) == "Sl.No."
    ):
        df = df.drop(index=0).reset_index(drop=True)

    df.columns = df.columns.str.strip()

    # Filter to target district
    district_slides = df[
        df["District"].str.contains(config.DISTRICT_NAME, case=False, na=False)
    ].copy()

    district_slides["Latitude"] = pd.to_numeric(
        district_slides["Latitude"], errors="coerce"
    )
    district_slides["Longitude"] = pd.to_numeric(
        district_slides["Longitude"], errors="coerce"
    )
    district_slides = district_slides.dropna(subset=["Latitude", "Longitude"])

    print(f"Valid numeric {config.DISTRICT_NAME} records: {len(district_slides)}")

    # Sanity check bounds
    out_of_bounds = district_slides[
        (district_slides["Latitude"] < config.DISTRICT_LAT_MIN)
        | (district_slides["Latitude"] > config.DISTRICT_LAT_MAX)
        | (district_slides["Longitude"] < config.DISTRICT_LON_MIN)
        | (district_slides["Longitude"] > config.DISTRICT_LON_MAX)
    ]
    print(f"Suspicious out-of-bounds records: {len(out_of_bounds)}")
    if len(out_of_bounds) > 0:
        print(out_of_bounds[["Slide_Name", "Latitude", "Longitude"]])

    # Convert to GeoDataFrame
    geometry = [
        Point(xy)
        for xy in zip(district_slides["Longitude"], district_slides["Latitude"])
    ]
    gdf = gpd.GeoDataFrame(district_slides, geometry=geometry, crs=config.WGS84_EPSG)

    gdf_utm = gdf.to_crs(config.UTM_EPSG)
    gdf_utm.to_file(config.LANDSLIDE_POINTS_UTM, driver="GeoJSON")

    print(f"Saved {len(gdf_utm)} landslide points: {config.LANDSLIDE_POINTS_UTM}")
    print("=== STEP 3 COMPLETE ===\n")


if __name__ == "__main__":
    main()
