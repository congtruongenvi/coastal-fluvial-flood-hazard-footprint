"""End-to-end driver: hazard layers -> connectivity -> exposure report.

Usage:
    python src/pipeline.py --selftest     # validate, no data needed
    python src/pipeline.py --case mekong-delta   # real run, expects data/raw/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_selftest() -> int:
    import selftest
    return selftest.main()


def run_real(case: str):
    """Real-data path. Expects a case directory under data/raw/<case>/ with:
        dem.tif, coast_mask.tif (or a coastline shapefile),
        discharge.csv, tide_levels.csv, population.tif
    This is a skeleton: fill in raster I/O (rasterio) to match your files.
    """
    raw = os.path.join("data", "raw", case)
    if not os.path.isdir(raw):
        print(f"No case directory at {raw}")
        print("Create it with: dem.tif, coast_mask.tif, discharge.csv, "
              "tide_levels.csv, population.tif")
        return 2

    try:
        import rasterio
    except ImportError:
        print("rasterio required for the real-data path: uv pip install rasterio")
        return 2

    dem_path = os.path.join(raw, "dem.tif")
    pop_path = os.path.join(raw, "population.tif")
    if not (os.path.exists(dem_path) and os.path.exists(pop_path)):
        print("dem.tif and population.tif are required in the case directory.")
        return 2

    from connectivity import flooded_mask, marginal_compound_area
    from hazard import (
        coastal_water_surface,
        fluvial_water_surface,
        compound_water_surface,
    )
    from exposure import exposure_report

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        cellsize = abs(src.transform.a)

    design_level = 2.0  # placeholder: read from tide_levels.csv in a real run
    coast_mask = np.zeros_like(dem, dtype=bool)
    coast_mask[:, 0] = True

    ws_coast = coastal_water_surface(dem, coast_mask, design_level,
                                     attenuation_per_km=0.0, cellsize=cellsize)
    ws_river = fluvial_water_surface(dem, coast_mask, design_stage=design_level,
                                     longitudinal_slope=1e-4, cellsize=cellsize)
    ws_compound = compound_water_surface(ws_river, ws_coast, mode="max")

    f_mask = flooded_mask(dem, ws_river, coast_mask, connectivity=8)
    c_mask = flooded_mask(dem, ws_coast, coast_mask, connectivity=8)
    comp_mask = flooded_mask(dem, ws_compound, coast_mask, connectivity=8)

    with rasterio.open(pop_path) as src:
        population = src.read(1).astype(float)

    report = exposure_report(f_mask, c_mask, comp_mask, population, cellsize)
    marg = marginal_compound_area(f_mask, c_mask, comp_mask, cellsize)

    print("Exposure report:", report)
    print("Marginal compound area:", marg)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--case", default=None)
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if args.case:
        return run_real(args.case)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
