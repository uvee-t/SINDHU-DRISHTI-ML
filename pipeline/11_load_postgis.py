"""
STEP 11: Load Outputs into PostGIS

Input:  data/outputs/red_zones_final.geojson, villages_with_risk.geojson,
        relocation_sites.geojson (from steps 6, 8, 10)
Output: Tables in PostGIS: red_zones, villages, relocation_sites

Automatically reprojects each file to WGS84 (EPSG:4326) if it isn't already,
then loads it via ogr2ogr with -overwrite, so re-running the full pipeline
always refreshes the database with the latest results — no manual ogr2ogr
commands needed.

Requires: the `ogr2ogr` command-line tool (comes with GDAL, already used
elsewhere in this pipeline) and a reachable PostgreSQL/PostGIS database
matching the settings in config.py.
"""

import os
import subprocess
import geopandas as gpd

import config


def ensure_wgs84(geojson_path):
    """
    Checks the CRS of a geojson file. If it's already WGS84, returns the path
    unchanged. Otherwise reprojects and saves a '_wgs84' version, returning
    that path instead.
    """
    gdf = gpd.read_file(geojson_path)

    if gdf.crs is not None and str(gdf.crs).upper() in ("EPSG:4326", "EPSG:4326 "):
        return geojson_path

    print(f"  Reprojecting {os.path.basename(geojson_path)} from {gdf.crs} to EPSG:4326...")
    gdf_wgs84 = gdf.to_crs(config.WGS84_EPSG)

    base, ext = os.path.splitext(geojson_path)
    wgs84_path = f"{base}_wgs84{ext}"
    gdf_wgs84.to_file(wgs84_path, driver="GeoJSON")
    return wgs84_path


def load_table(geojson_path, table_name):
    pg_connection = (
        f"PG:host={config.DB_HOST} dbname={config.DB_NAME} "
        f"user={config.DB_USER} password={config.DB_PASSWORD} port={config.DB_PORT}"
    )

    result = subprocess.run([
        "ogr2ogr", "-f", "PostgreSQL",
        pg_connection,
        geojson_path,
        "-nln", table_name,
        "-overwrite"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR loading {table_name}:")
        print(f"  {result.stderr}")
        return False

    return True


def get_table_count(table_name):
    """Quick row count check using ogrinfo, so we don't need a Python Postgres driver."""
    pg_connection = (
        f"PG:host={config.DB_HOST} dbname={config.DB_NAME} "
        f"user={config.DB_USER} password={config.DB_PASSWORD} port={config.DB_PORT}"
    )
    result = subprocess.run(
        ["ogrinfo", "-al", "-so", pg_connection, table_name],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "Feature Count" in line:
            return line.split(":")[-1].strip()
    return "unknown"


def main():
    print("=== STEP 11: Load Outputs into PostGIS ===")
    print(f"Target database: {config.DB_NAME}@{config.DB_HOST}:{config.DB_PORT}")

    any_failed = False

    for table_name, config_key in config.POSTGIS_LOAD_TARGETS:
        geojson_path = getattr(config, config_key)

        if not os.path.exists(geojson_path):
            print(f"\nSKIPPED '{table_name}': {geojson_path} does not exist yet "
                  f"(run the earlier pipeline step that generates it first).")
            any_failed = True
            continue

        print(f"\nLoading '{table_name}' from {os.path.basename(geojson_path)}...")
        final_path = ensure_wgs84(geojson_path)

        success = load_table(final_path, table_name)
        if not success:
            any_failed = True
            continue

        count = get_table_count(table_name)
        print(f"  Loaded '{table_name}': {count} features")

    print()
    if any_failed:
        print("=== STEP 11 COMPLETED WITH WARNINGS (see above) ===\n")
    else:
        print("=== STEP 11 COMPLETE — all tables loaded ===\n")


if __name__ == "__main__":
    main()