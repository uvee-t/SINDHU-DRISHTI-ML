"""
STEP 8: Village <-> Red Zone Spatial Join

Input:  data/outputs/villages_with_population.geojson (from step 7)
        data/outputs/red_zones_final.geojson (from step 6)
Output: data/outputs/villages_with_risk.geojson

For each village, computes what fraction of its area falls in Red / Yellow / Green
zones, estimates how many people are affected, and assigns a priority tier:
  IMMEDIATE   - large fraction of village area in red zone
  SHORT_TERM  - moderate fraction in red zone
  MEDIUM_TERM - small fraction in red zone
  MONITOR     - no meaningful red zone overlap
"""

import os
import geopandas as gpd
import pandas as pd

import config


def main():
    print("=== STEP 8: Village <-> Red Zone Spatial Join ===")

    if not os.path.exists(config.VILLAGES_WITH_POP_PATH):
        raise FileNotFoundError(f"{config.VILLAGES_WITH_POP_PATH} not found. Run 07_prepare_population.py first.")
    if not os.path.exists(config.RED_ZONES_FINAL_GEOJSON):
        raise FileNotFoundError(f"{config.RED_ZONES_FINAL_GEOJSON} not found. Run 06_generate_redzone_map.py first.")

    villages = gpd.read_file(config.VILLAGES_WITH_POP_PATH)
    zones = gpd.read_file(config.RED_ZONES_FINAL_GEOJSON)

    print(f"Villages: {len(villages)}")
    print(f"Zone polygons: {len(zones)}")

    # ---- Match CRS (both should be UTM already, but confirm) ----
    if villages.crs != zones.crs:
        print(f"CRS mismatch (villages={villages.crs}, zones={zones.crs}) — reprojecting zones to match villages")
        zones = zones.to_crs(villages.crs)

    villages = villages.copy()
    villages["village_area_sqm"] = villages.geometry.area
    villages["_row_id"] = range(len(villages))

    # ---- Overlay: intersect villages with zones ----
    # This splits each village into pieces based on which zone(s) it overlaps
    overlay = gpd.overlay(
        villages[["_row_id", "geometry"]],
        zones[["zone_class", "geometry"]],
        how="intersection"
    )
    overlay["overlap_area_sqm"] = overlay.geometry.area

    # ---- Aggregate: for each village, sum overlap area per zone class ----
    pivot = overlay.pivot_table(
        index="_row_id", columns="zone_class", values="overlap_area_sqm", aggfunc="sum", fill_value=0
    )
    pivot.columns = [f"area_zone_{int(c)}" for c in pivot.columns]  # zone_0, zone_1, zone_2
    for col in ["area_zone_0", "area_zone_1", "area_zone_2"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    villages = villages.merge(pivot, on="_row_id", how="left")
    for col in ["area_zone_0", "area_zone_1", "area_zone_2"]:
        villages[col] = villages[col].fillna(0.0)

    # ---- Compute percentages (guard against divide-by-zero for zero-area geometries) ----
    villages["pct_green"] = villages["area_zone_0"] / villages["village_area_sqm"].replace(0, pd.NA)
    villages["pct_yellow"] = villages["area_zone_1"] / villages["village_area_sqm"].replace(0, pd.NA)
    villages["pct_red"] = villages["area_zone_2"] / villages["village_area_sqm"].replace(0, pd.NA)
    villages[["pct_green", "pct_yellow", "pct_red"]] = villages[["pct_green", "pct_yellow", "pct_red"]].fillna(0.0)

    # ---- Estimated affected population (population * fraction in red zone) ----
    villages["estimated_affected_population"] = (villages["population"] * villages["pct_red"]).round().astype(int)

    # ---- Priority tier ----
    def assign_tier(pct_red):
        if pct_red >= config.IMMEDIATE_RED_PCT_THRESHOLD:
            return "IMMEDIATE"
        elif pct_red >= config.SHORT_TERM_RED_PCT_THRESHOLD:
            return "SHORT_TERM"
        elif pct_red >= config.MEDIUM_TERM_RED_PCT_THRESHOLD:
            return "MEDIUM_TERM"
        else:
            return "MONITOR"

    villages["priority_tier"] = villages["pct_red"].apply(assign_tier)

    villages = villages.drop(columns=["_row_id"])

    # ---- Summary ----
    print("\nPriority tier breakdown:")
    print(villages["priority_tier"].value_counts())

    print("\nEstimated total affected population by tier:")
    print(villages.groupby("priority_tier")["estimated_affected_population"].sum())

    at_risk = villages[villages["priority_tier"].isin(["IMMEDIATE", "SHORT_TERM"])]
    print(f"\nTotal villages needing near-term attention (IMMEDIATE + SHORT_TERM): {len(at_risk)}")
    print(f"Total estimated affected population (IMMEDIATE + SHORT_TERM): "
          f"{at_risk['estimated_affected_population'].sum():,}")

    print("\nTop 10 highest-priority villages by estimated affected population:")
    top10 = villages.sort_values("estimated_affected_population", ascending=False).head(10)
    print(top10[[config.VILLAGE_NAME_COLUMN, "population", "pct_red", "estimated_affected_population", "priority_tier"]])

    villages.to_file(config.VILLAGES_WITH_RISK_PATH, driver="GeoJSON")
    print(f"\nSaved: {config.VILLAGES_WITH_RISK_PATH}")
    print("=== STEP 8 COMPLETE ===\n")


if __name__ == "__main__":
    main()