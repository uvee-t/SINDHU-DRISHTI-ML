"""
Shared configuration for the entire pipeline.
Change values here once — every script reads from this file.
"""

import os

# ============================================================
# PROJECT SETTINGS
# ============================================================

DISTRICT_NAME = "Darjeeling"
UTM_EPSG = "EPSG:32645"  # UTM zone 45N — correct for 88°E region. Change if you move districts.
WGS84_EPSG = "EPSG:4326"

# ============================================================
# FOLDER STRUCTURE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEM_RAW_DIR = os.path.join(RAW_DIR, "dem")
RAINFALL_RAW_DIR = os.path.join(RAW_DIR, "rainfall")
LANDSLIDE_RAW_DIR = os.path.join(RAW_DIR, "landslide_inventory")
BOUNDARY_RAW_DIR = os.path.join(RAW_DIR, "boundary")
SOIL_RAW_DIR = os.path.join(RAW_DIR, "soil")
POPULATION_RAW_DIR = os.path.join(RAW_DIR, "population")
VILLAGES_RAW_DIR = os.path.join(RAW_DIR, "villages")
ROADS_RAW_DIR = os.path.join(RAW_DIR, "roads")
LANDUSE_RAW_DIR = os.path.join(RAW_DIR, "landuse")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Create folders automatically if they don't exist yet
for folder in [
    DEM_RAW_DIR,
    RAINFALL_RAW_DIR,
    LANDSLIDE_RAW_DIR,
    BOUNDARY_RAW_DIR,
    SOIL_RAW_DIR,
    POPULATION_RAW_DIR,
    VILLAGES_RAW_DIR,
    ROADS_RAW_DIR,
    LANDUSE_RAW_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# FILE PATHS (processed/intermediate files — auto-generated, don't edit by hand)
# ============================================================

DEM_CLIPPED_UTM = os.path.join(PROCESSED_DIR, "dem_utm.tif")
DEM_MERGED_PATH = os.path.join(PROCESSED_DIR, "dem_merged.tif")
DEM_MERGED_UTM_PATH = os.path.join(PROCESSED_DIR, "dem_merged_utm.tif")
SLOPE_PATH = os.path.join(PROCESSED_DIR, "slope.tif")
ASPECT_PATH = os.path.join(PROCESSED_DIR, "aspect.tif")
RAINFALL_TOTAL_PATH = os.path.join(PROCESSED_DIR, "total_monsoon_rainfall.tif")
STREAM_DISTANCE_PATH = os.path.join(PROCESSED_DIR, "stream_distance.tif")
SOIL_CODE_PATH = os.path.join(PROCESSED_DIR, "soil_code.tif")
POPULATION_UTM_PATH = os.path.join(PROCESSED_DIR, "population_utm.tif")
VILLAGES_WITH_POP_PATH = os.path.join(OUTPUT_DIR, "villages_with_population.geojson")
VILLAGES_WITH_RISK_PATH = os.path.join(OUTPUT_DIR, "villages_with_risk.geojson")
ROAD_DISTANCE_PATH = os.path.join(PROCESSED_DIR, "road_distance.tif")
LANDUSE_PATH = os.path.join(PROCESSED_DIR, "landuse.tif")
DEVELOPABLE_MASK_PATH = os.path.join(PROCESSED_DIR, "developable_mask.tif")
RELOCATION_SITES_PATH = os.path.join(OUTPUT_DIR, "relocation_sites.geojson")

# ============================================================
# FLOOD (HAND) SETTINGS
# ============================================================
# Reuses files already produced by 01b_prepare_stream_distance.py — no recomputation needed
DEM_MERGED_FILLED_PATH = os.path.join(PROCESSED_DIR, "dem_merged_filled.tif")
FLOW_ACCUM_FULL_PATH = os.path.join(PROCESSED_DIR, "flow_accum_full.tif")
HAND_FULL_PATH = os.path.join(PROCESSED_DIR, "hand_full.tif")
HAND_CLIPPED_PATH = os.path.join(PROCESSED_DIR, "hand.tif")
FLOOD_ZONES_FINAL_GEOJSON = os.path.join(OUTPUT_DIR, "flood_zones_final.geojson")

# HAND thresholds in meters — below HIGH_RISK = likely to flood, above LOW_RISK = safe.
# These are common starting points from HAND literature; refine with real flood
# extent data if available.
HAND_HIGH_RISK_MAX_M = 5
HAND_MODERATE_RISK_MAX_M = 15
SOIL_MAPPING_JSON = os.path.join(PROCESSED_DIR, "soil_code_mapping.json")
LANDSLIDE_POINTS_UTM = os.path.join(PROCESSED_DIR, "landslide_points_utm.geojson")
TRAINING_DATA_CSV = os.path.join(PROCESSED_DIR, "training_data.csv")

# ============================================================
# SOIL SETTINGS
# ============================================================
# HWSD2 is a raster (.bil) where pixel values are Soil Mapping Unit IDs —
# used directly as categorical codes, no attribute column needed.

# ============================================================
# STREAM EXTRACTION SETTINGS
# ============================================================
# Flow accumulation threshold (in number of upstream cells) above which a pixel
# is considered part of the stream network. Lower = denser stream network.
# Tune this after first run by checking how the extracted streams look vs real rivers.
STREAM_FLOW_ACCUM_THRESHOLD = 1000

# ============================================================
# FULL FEATURE LIST (single source of truth used by steps 4, 5, 6)
# ============================================================
FEATURE_COLS = [
    "slope",
    "aspect",
    "rainfall",
    "elevation",
    "stream_distance",
    "soil_code",
]

# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

MODEL_PATH = os.path.join(OUTPUT_DIR, "landslide_rf_model.pkl")
PROBABILITY_RASTER = os.path.join(OUTPUT_DIR, "landslide_probability.tif")
ZONE_RASTER = os.path.join(OUTPUT_DIR, "red_zone_classification.tif")
RED_ZONES_FINAL_GEOJSON = os.path.join(OUTPUT_DIR, "red_zones_final.geojson")

# ============================================================
# MODEL / ZONE THRESHOLDS
# ============================================================

RED_ZONE_THRESHOLD = 0.7
YELLOW_ZONE_THRESHOLD = 0.4
MIN_POLYGON_AREA_SQM = 10000  # 1 hectare — drop noise polygons smaller than this
SIMPLIFY_TOLERANCE = 15  # meters, for smoothing jagged pixel edges
MIN_FINAL_ZONE_AREA_SQKM = 0.5  # drop slivers after dissolve

# ============================================================
# VILLAGE / POPULATION SETTINGS
# ============================================================
# Column name in your village/block shapefile that identifies the district.
# CHANGE THIS to match your actual downloaded file — check with:
#   python -c "import geopandas as gpd; print(gpd.read_file('yourfile.shp').columns.tolist())"
VILLAGE_DISTRICT_COLUMN = "District"

# Column name for the village/block's own name (for display purposes)
VILLAGE_NAME_COLUMN = "Vill_name"

# The exact district name string as spelled in YOUR village file — this can differ
# from DISTRICT_NAME above (e.g. official LGD data spells it "Darjiling", not "Darjeeling")
VILLAGE_DISTRICT_FILTER = "Darjeeling"

# ============================================================
# DATABASE (PostGIS) SETTINGS
# ============================================================
# Reads from environment variables if set, otherwise falls back to these defaults.
# For anything beyond local development, set these as real env vars instead of
# editing the fallback values directly.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "redzone_db")
DB_USER = os.environ.get("DB_USER", "redzone_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "0829")
DB_PORT = os.environ.get("DB_PORT", "5432")

# Which output files get loaded into which PostGIS tables (loader checks CRS
# automatically and reprojects to WGS84 if needed — this list is what maps to what).
POSTGIS_LOAD_TARGETS = [
    ("red_zones", "RED_ZONES_FINAL_GEOJSON"),
    ("villages", "VILLAGES_WITH_RISK_PATH"),
    ("relocation_sites", "RELOCATION_SITES_PATH"),
    ("flood_zones", "FLOOD_ZONES_FINAL_GEOJSON"),
]

# ============================================================
# DISTRICT BOUNDS (sanity-check filter only, not used for clipping)
# ============================================================
DISTRICT_LAT_MIN, DISTRICT_LAT_MAX = 26.4, 27.3
DISTRICT_LON_MIN, DISTRICT_LON_MAX = 87.9, 88.9

# ============================================================
# PRIORITY TIER THRESHOLDS
# ============================================================
# Fraction of a village's area that must fall in the RED zone to trigger each tier.
# Tune these based on how the distribution actually looks after first run.
IMMEDIATE_RED_PCT_THRESHOLD = 0.5  # >=50% of village area in red zone
SHORT_TERM_RED_PCT_THRESHOLD = 0.2  # >=20% of village area in red zone
MEDIUM_TERM_RED_PCT_THRESHOLD = 0.05  # >=5% of village area in red zone

# ============================================================
# CARRYING CAPACITY SETTINGS
# ============================================================
BUILDABLE_SLOPE_MAX_DEG = 15  # slope below this = considered buildable
SQM_PER_HOUSEHOLD = 200  # planning norm: house + basic amenities
MIN_SITE_AREA_SQKM = 0.05  # ignore green-zone fragments smaller than this (~5 hectares)
AVG_HOUSEHOLD_SIZE = 4.6  # approx average household size, West Bengal
WATER_PROXIMITY_CAP_M = 2000  # distances beyond this get the same (lowest) water score
CAPACITY_WEIGHT_BUILDABLE = 0.7  # suitability score weighting: terrain buildability
CAPACITY_WEIGHT_WATER = 0.3  # suitability score weighting: water proximity

# What fraction of "buildable" (by slope) land could realistically be allocated to
# NEW relocation housing, given that most such land is already agricultural, forest,
# tea garden, or existing settlement. Without land-use data to exclude these directly,
# this conservative multiplier prevents wildly unrealistic capacity estimates.
# 0.08 = assume ~8% of slope-suitable land could actually be developed for relocation.
# This is a stated, adjustable assumption — not a precise figure — and should be
# refined with real land-use/land-cover data in a future iteration.
DEVELOPABLE_LAND_FRACTION = 0.08
# NOTE: road/connectivity is intentionally excluded from this version — no OSM road
# data available yet. Suitability here reflects terrain + water only. Flagged as a
# known limitation / future addition, not silently ignored.

# ESA WorldCover class codes considered "potentially developable" for new relocation
# housing — i.e. not already forest, cropland, built-up, water, or wetland.
# 30 = Grassland, 60 = Bare/sparse vegetation.
# This directly replaces the old flat DEVELOPABLE_LAND_FRACTION guess with an actual
# per-site computed value from real land cover data.
LANDUSE_DEVELOPABLE_CLASSES = [30, 60]
