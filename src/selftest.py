"""
Saturated self-test for the flood hazard pipeline.

Builds a synthetic delta with a known geometry — a levee, a basin, and a coastline —
and asserts the connectivity logic reconstructs the truth. No data needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connectivity import (
    flooded_mask,
    flooded_mask_naive,
    connectivity_diagnostics,
    marginal_compound_area,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL))
    print(f"[{PASS if condition else FAIL}] {name}{' — ' + detail if detail else ''}")


def synthetic_delta():
    """A 60x80 synthetic delta: coast on the west, a levee along col 25–27,
    a sealed interior basin behind it connected ONLY through a single breach.

    Elevations:
      0.5  coastal plain (submerged below the 2.0 m design water surface)
      1.2  tidal flats
      1.8  inland coastal plain, also below the design level
      2.0  ground above the design level -- not submerged (strict less-than)
      3.5  levee
      0.6  interior basin, enclosed by the 2.0 m ring except through the breach
      0.8  river channel at the far east (col 58–60), blocked from the basin
           by the 2.0 m ground between col 45 and 57
      8.0  high plateau
    """
    rows, cols = 60, 80
    dem = np.full((rows, cols), 2.0, dtype=float)
    dem[:, :10] = 0.5
    dem[:, 10:16] = 1.2
    dem[:, 16:25] = 1.8
    dem[:, 25:28] = 3.5            # levee
    dem[20:41, 28:45] = 0.6       # interior basin behind the levee
    dem[28:33, 25:29] = 1.0       # single breach in the levee, joining plain to basin
    dem[:, 58:61] = 0.8           # river channel far east
    dem[0:15, 50:] = 8.0          # plateau

    coast = np.zeros_like(dem, dtype=bool)
    coast[:, 0] = True
    river = np.zeros_like(dem, dtype=bool)
    river[0, 58:61] = True

    return dem, coast, river


def main():
    print("=" * 60)
    print("Flood hazard pipeline — self tests")
    print("=" * 60)
    dem, coast, river = synthetic_delta()
    river_source = river.copy()

    design_level = 2.0  # metres, e.g. a 1-in-100 with surge

    # Coastal plain in the synthetic delta sits at 2.0 m — EXACTLY the design
    # level. An elevation-only flood will not cross it (needs strict <), so the
    # coastal front stops at the plain boundary. The basin is only reachable via
    # the river, which is the physically correct compound behaviour.
    ws_coast = np.full(dem.shape, design_level)
    ws_river = np.full(dem.shape, design_level)
    ws_compound = np.fmax(ws_coast, ws_river)

    # --- 1. coastal flood: must enter the basin only via the breach
    flooded_coast = flooded_mask(dem, ws_coast, source_mask=coast, connectivity=4)
    check("coast flood reaches basin through breach",
          bool(flooded_coast[30, 30]),
          f"cell (30,30) flooded={bool(flooded_coast[30,30])}")

    # River is disconnected from the basin: 2.0 m ground ring blocks it.
    flooded_river = flooded_mask(dem, ws_river, source_mask=river_source, connectivity=4)
    check("river is blocked from basin by dry ground",
          not bool(flooded_river[30, 34]),
          f"cell (30,34) flooded={bool(flooded_river[30,34])}")

    # Compound (coast + river) still reaches the basin, via the coastal path.
    both = coast | river_source
    flooded_compound = flooded_mask(dem, ws_compound, source_mask=both, connectivity=4)
    check("compound footprint reaches the basin",
          bool(flooded_compound[30, 30]))

    # The high plateau must never flood.
    check("high plateau stays dry", not bool(flooded_compound[5, 60]))

    # A cell in the interior basin must be submerged but UNREACHABLE once sealed.
    dem_sealed = dem.copy()
    dem_sealed[28:33, 25:29] = 3.5
    flooded_sealed = flooded_mask(dem_sealed, ws_coast, source_mask=coast, connectivity=4)
    check("sealed levee blocks the basin", not bool(flooded_sealed[31, 30]))

    # A naive elevation-only mask would still flood the sealed basin.
    naive = flooded_mask_naive(dem_sealed, ws_coast)
    check("naive mask does flood the sealed basin", bool(naive[31, 30]))

    # --- 2. diagnostics: naive elevation-only mask floods the sealed basin,
    #        connected mask does not. The delta between the two is the error avoided.
    dem_sealed2 = dem.copy()
    dem_sealed2[28:33, 25:29] = 3.5
    diag = connectivity_diagnostics(dem_sealed2, ws_coast, source_mask=coast,
                                    connectivity=4, cellsize_m=30.0)
    print("      diagnostics (sealed):", diag)
    check("naive predicts more area than connected",
          diag["naive_area_km2"] > diag["connected_area_km2"])
    check("disconnected area is positive",
          diag["disconnected_area_km2"] > 0)

    # --- 3. seal the breach and confirm the basin disconnects entirely
    flooded_sealed = flooded_mask(dem_sealed2, ws_coast, source_mask=coast, connectivity=4)
    check("sealed levee blocks the basin", not bool(flooded_sealed[31, 30]))

    # --- 4. compound water surface: max superposition, not sum
    from hazard import compound_water_surface
    fluv = np.array([[3.0, 1.0]])
    coas = np.array([[1.5, 4.0]])
    ws_max = compound_water_surface(fluv, coas, mode="max")
    ws_sum = compound_water_surface(fluv, coas, mode="sum")
    check("compound max picks the higher surface", list(ws_max[0]) == [3.0, 4.0])
    check("compound sum would double-count (known-bad mode kept for comparison)",
          list(ws_sum[0]) == [4.5, 5.0])

    # --- 5. marginal compound area accounting
    fluvial = np.zeros_like(dem, dtype=bool); fluvial[0:20, 58:61] = True
    coastal = np.zeros_like(dem, dtype=bool); coastal[:, :10] = True
    compound = fluvial | coastal
    marg = marginal_compound_area(fluvial, coastal, compound, cellsize_m=30.0)
    check("marginal compound area non-negative",
          marg["marginal_over_largest_single"] >= 0)

    print()
    n_fail = sum(1 for _, s in results if s == FAIL)
    print("=" * 60)
    print(f"{len(results) - n_fail} passed, {n_fail} failed, {len(results)} total")
    print("=" * 60)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
