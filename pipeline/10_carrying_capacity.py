"""
STEP 10: Carrying Capacity for Relocation Sites

Input:  data/outputs/red_zones_final.geojson (for green-zone site polygons)
        data/processed/slope.tif
        data/processed/stream_distance.tif
Output: data/outputs/relocation_sites.geojson

For each green-zone (safe) polygon, estimates:
  - buildable_pct: fraction of the site with slope below the buildable threshold
  - avg_stream_distance_m: average distance to water within the site
  - suitability_score: weighted combination of the above (0-1)
  - estimated_household_capacity / estimated_population_capacity

LIMITATION (stated explicitly, not hidden): this version does not account for
road/connectivity access, since no road network data was available. Suitability
reflects terrain buildability and water proximity only. Road distance is a
natural v2 addition using the same distance-transform pattern as stream_distance.
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

import config

def buildable_fraction_stat(masked_array):
    """Custom zonal stat: fraction of pixels with slope below the buildable threshold."""
    valid = masked_array.compressed()
    if valid.size == 0:
        return 0.0
    return float((valid < config.BUILDABLE_SLOPE_MAX_DEG).sum()) / valid.size


def main():
    print("=== STEP 10: Carrying Capacity for Relocation Sites ===")

    for path in [config.RED_ZONES_FINAL_GEOJSON, config.SLOPE_PATH, config.STREAM_DISTANCE_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}. Run earlier steps first.")

    use_real_landuse = os.path.exists(config.LANDUSE_PATH)
    if use_real_landuse:
        print("Land use data found — computing real developable-land fraction "
              "instead of the flat DEVELOPABLE_LAND_FRACTION assumption.")
        with rasterio.open(config.SLOPE_PATH) as src:
            slope_data = src.read(1)
            profile_for_mask = src.profile
        with rasterio.open(config.LANDUSE_PATH) as src:
            landuse_data = src.read(1)

        buildable = slope_data < config.BUILDABLE_SLOPE_MAX_DEG
        developable_landuse = np.isin(landuse_data, config.LANDUSE_DEVELOPABLE_CLASSES)
        developable_mask = (buildable & developable_landuse).astype("uint8")

        profile_for_mask.update(dtype="uint8", nodata=None)
        with rasterio.open(config.DEVELOPABLE_MASK_PATH, "w", **profile_for_mask) as dst:
            dst.write(developable_mask, 1)
        print(f"Developable mask saved: {config.DEVELOPABLE_MASK_PATH} "
              f"({developable_mask.sum()} developable pixels of {developable_mask.size})")
    else:
        print("No land use data found — falling back to flat DEVELOPABLE_LAND_FRACTION "
              f"assumption ({config.DEVELOPABLE_LAND_FRACTION}). Run 13_prepare_landuse.py "
              "for a more rigorous, land-use-based estimate.")

    zones = gpd.read_file(config.RED_ZONES_FINAL_GEOJSON)
    green_sites = zones[zones["zone_class"] == 0].copy()
    print(f"Green-zone candidate sites: {len(green_sites)}")

    green_sites["area_sqkm"] = green_sites.geometry.area / 1_000_000
    green_sites = green_sites[green_sites["area_sqkm"] >= config.MIN_SITE_AREA_SQKM].reset_index(drop=True)
    print(f"Sites after minimum-size filter ({config.MIN_SITE_AREA_SQKM} sqkm): {len(green_sites)}")

    if len(green_sites) == 0:
        raise ValueError("No candidate sites remain after filtering. Lower MIN_SITE_AREA_SQKM in config.py.")

    # ---- Buildable fraction via custom zonal stat on slope raster ----
    slope_stats = zonal_stats(
        green_sites,
        config.SLOPE_PATH,
        add_stats={"buildable_frac": buildable_fraction_stat}
    )
    green_sites["buildable_pct"] = [s["buildable_frac"] for s in slope_stats]

    # ---- Average distance to water via zonal stat on stream_distance raster ----
    water_stats = zonal_stats(green_sites, config.STREAM_DISTANCE_PATH, stats=["mean"])
    green_sites["avg_stream_distance_m"] = [s["mean"] if s["mean"] is not None else config.WATER_PROXIMITY_CAP_M
                                             for s in water_stats]

    # ---- Water proximity score: 1 = right next to water, 0 = at or beyond the cap distance ----
    green_sites["water_score"] = 1 - (
        green_sites["avg_stream_distance_m"].clip(upper=config.WATER_PROXIMITY_CAP_M)
        / config.WATER_PROXIMITY_CAP_M
    )

    # ---- Composite suitability score ----
    green_sites["suitability_score"] = (
        config.CAPACITY_WEIGHT_BUILDABLE * green_sites["buildable_pct"] +
        config.CAPACITY_WEIGHT_WATER * green_sites["water_score"]
    )

    # ---- Estimated capacity ----
    green_sites["buildable_area_sqm"] = green_sites["buildable_pct"] * (green_sites["area_sqkm"] * 1_000_000)

    if use_real_landuse:
        developable_stats = zonal_stats(green_sites, config.DEVELOPABLE_MASK_PATH, stats=["mean"], nodata=255)
        green_sites["developable_pct"] = [s["mean"] if s["mean"] is not None else 0.0
                                           for s in developable_stats]
        green_sites["developable_area_sqm"] = green_sites["developable_pct"] * (green_sites["area_sqkm"] * 1_000_000)
        print("\nUsing real land-use-derived developable_pct (see column) instead of flat assumption.")
    else:
        green_sites["developable_pct"] = green_sites["buildable_pct"] * config.DEVELOPABLE_LAND_FRACTION
        green_sites["developable_area_sqm"] = green_sites["buildable_area_sqm"] * config.DEVELOPABLE_LAND_FRACTION

    green_sites["estimated_household_capacity"] = (
        green_sites["developable_area_sqm"] / config.SQM_PER_HOUSEHOLD
    ).round().astype(int)
    green_sites["estimated_population_capacity"] = (
        green_sites["estimated_household_capacity"] * config.AVG_HOUSEHOLD_SIZE
    ).round().astype(int)

    green_sites["site_id"] = [f"site_{i+1}" for i in range(len(green_sites))]

    green_sites = green_sites.sort_values("suitability_score", ascending=False).reset_index(drop=True)

    print("\nSuitability score summary:")
    print(green_sites["suitability_score"].describe())

    print("\nTotal estimated capacity across all sites:")
    print(f"  Households: {green_sites['estimated_household_capacity'].sum():,}")
    print(f"  Population: {green_sites['estimated_population_capacity'].sum():,}")
    if use_real_landuse:
        print(f"  (Using real land-use-derived developable fraction — see landuse.tif)")
    else:
        print(f"  (Using flat DEVELOPABLE_LAND_FRACTION={config.DEVELOPABLE_LAND_FRACTION} — see config.py)")

    total_capacity = green_sites['estimated_population_capacity'].sum()
    if total_capacity > 2_000_000:
        print("WARNING: total capacity exceeds Darjeeling's entire district population (~2M). "
              "Consider lowering DEVELOPABLE_LAND_FRACTION in config.py.")

    print("\nTop 10 sites by suitability:")
    print(green_sites[["site_id", "area_sqkm", "buildable_pct", "avg_stream_distance_m",
                        "suitability_score", "estimated_population_capacity"]].head(10))

    # Reproject to WGS84 for consistency with your other output files (villages, red_zones)
    green_sites_wgs84 = green_sites.to_crs(config.WGS84_EPSG)
    green_sites_wgs84.to_file(config.RELOCATION_SITES_PATH, driver="GeoJSON")

    print(f"\nSaved: {config.RELOCATION_SITES_PATH}")
    print("NOTE: suitability reflects terrain + water proximity only (no road data yet).")
    print("=== STEP 10 COMPLETE ===\n")


if __name__ == "__main__":
    main()