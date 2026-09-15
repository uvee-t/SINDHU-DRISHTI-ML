"""
STEP 7: Prepare Population + Villages

Input:  A WorldPop .tif in data/raw/population/, a village/block shapefile in data/raw/villages/
Output: data/outputs/villages_with_population.geojson

Requires: pip install rasterstats
"""

import glob
import os
import subprocess

import config
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats


def find_file(folder, extensions):
    for ext in extensions:
        candidates = glob.glob(os.path.join(folder, f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No file with extensions {extensions} found in {folder}")


def find_boundary_file():
    candidates = glob.glob(
        os.path.join(config.BOUNDARY_RAW_DIR, "*.geojson")
    ) + glob.glob(os.path.join(config.BOUNDARY_RAW_DIR, "*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No boundary file found in {config.BOUNDARY_RAW_DIR}")
    return candidates[0]


def main():
    print("=== STEP 7: Prepare Population + Villages ===")

    if not os.path.exists(config.DEM_CLIPPED_UTM):
        raise FileNotFoundError(
            f"{config.DEM_CLIPPED_UTM} not found. Run 01_prepare_dem.py first."
        )

    # Find input files
    population_path = find_file(config.POPULATION_RAW_DIR, ["tif"])
    villages_path = find_file(config.VILLAGES_RAW_DIR, ["shp", "geojson"])
    print(f"Using population raster: {population_path}")
    print(f"Using village file: {villages_path}")

    # Clip population to district boundary
    boundary = gpd.read_file(find_boundary_file())

    with rasterio.open(population_path) as src:
        boundary_reproj = boundary.to_crs(src.crs)
        out_image, out_transform = mask(
            src, boundary_reproj.geometry, crop=True, nodata=-99999, filled=True
        )
        out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -99999,
        }
    )
    clipped_path = os.path.join(config.PROCESSED_DIR, "population_clipped.tif")
    with rasterio.open(clipped_path, "w", **out_meta) as dst:
        dst.write(out_image)
    print("Clipped population raster saved")

    # ---- Reproject to UTM at NATIVE resolution (do NOT force onto the DEM's fine grid) ----
    # Population is an "extensive" variable (a count, not a measurement like elevation/rainfall).
    # Resampling count data to a much finer grid without adjusting values inflates totals,
    # since bilinear interpolation copies the value into many more, smaller pixels without
    # dividing it. Keep this at its own native ~100m resolution instead.
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            config.UTM_EPSG,
            "-tr",
            "100",
            "100",
            "-r",
            "bilinear",
            "-overwrite",
            "-dstnodata",
            "-99999",
            clipped_path,
            config.POPULATION_UTM_PATH,
        ],
        check=True,
    )
    print(
        f"Aligned population raster saved (native ~100m resolution): {config.POPULATION_UTM_PATH}"
    )

    # Load villages, filter to district, reproject
    villages = gpd.read_file(villages_path)
    print(f"Available columns in village file: {villages.columns.tolist()}")

    if config.VILLAGE_DISTRICT_COLUMN not in villages.columns:
        raise KeyError(
            f"Column '{config.VILLAGE_DISTRICT_COLUMN}' not found. "
            f"Set config.VILLAGE_DISTRICT_COLUMN to one of: {villages.columns.tolist()}"
        )

    district_villages = villages[
        villages[config.VILLAGE_DISTRICT_COLUMN].str.contains(
            config.VILLAGE_DISTRICT_FILTER, case=False, na=False
        )
    ].copy()
    print(
        f"Villages/blocks found for {config.VILLAGE_DISTRICT_FILTER}: {len(district_villages)}"
    )

    district_villages_utm = district_villages.to_crs(config.UTM_EPSG)

    # Zonal statistics: sum population within each village polygon
    stats = zonal_stats(
        district_villages_utm, config.POPULATION_UTM_PATH, stats=["sum"], nodata=-99999
    )
    district_villages_utm["population"] = [
        s["sum"] if s["sum"] is not None else 0 for s in stats
    ]

    print("\nPopulation summary:")
    print(district_villages_utm["population"].describe())

    total_population = district_villages_utm["population"].sum()
    print(f"\nTotal population across all villages: {total_population:,.0f}")
    print(
        "Sanity check: Darjeeling district's real population is roughly 1.8-2.2 million."
    )
    if total_population > 3_000_000 or total_population < 1_000_000:
        print(
            "WARNING: total is well outside the expected range — check for resampling/clipping issues."
        )

    district_villages_utm.to_file(config.VILLAGES_WITH_POP_PATH, driver="GeoJSON")
    print(f"\nSaved: {config.VILLAGES_WITH_POP_PATH}")
    print("=== STEP 7 COMPLETE ===\n")


if __name__ == "__main__":
    main()
