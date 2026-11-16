"""
Exposure overlay: combine hazard footprints with population / built-up.

Exposure is NOT risk. This module computes how many people or how many built-up
cells lie under the hazard footprint — the input every risk assessment needs,
before vulnerability and coping-capacity terms are added.

All layers must share the DEM grid. Resampling coarse population onto the DEM
without a stated method is the silent failure mode here; `align_population` does
the alignment explicitly and records what it did.
"""

from __future__ import annotations

import numpy as np


def align_population(
    population: np.ndarray,
    target_shape: tuple[int, int],
    method: str = "block_sum",
) -> np.ndarray:
    """Bring a population raster onto the DEM grid.

    method='block_sum'  -- populations ADD across cells, so downsampling must
                           sum. This is the only defensible aggregation for a
                           count variable.
    method='block_mean' -- averages instead of summing, WRONG for counts but
                           kept for contrast in the self-test comparison.
    method='nearest'    -- zero-order resampling, e.g. resampling DEM-matched
                           population to a slightly different DEM. Loses total
                           population unless followed by renormalisation, done
                           below and documented.
    """
    if method not in ("block_sum", "block_mean", "nearest"):
        raise ValueError("unknown method")

    if population.shape == target_shape:
        return population.astype(float).copy()

    if method == "block_sum" or method == "block_mean":
        # Only valid when shrinking by integer factors.
        r = population.shape[0] // target_shape[0]
        c = population.shape[1] // target_shape[1]
        if population.shape[0] % target_shape[0] or population.shape[1] % target_shape[1]:
            raise ValueError(
                "block_sum requires integer downsampling factors; "
                "use 'nearest' + renormalisation for non-integer ratios"
            )
        reshaped = population.reshape(target_shape[0], r, target_shape[1], c)
        agg = reshaped.sum(axis=(1, 3)) if method == "block_sum" else reshaped.mean(axis=(1, 3))
        return agg.astype(float)

    # nearest: nearest-neighbour, then renormalise to preserve total population.
    ys = (np.arange(target_shape[0]) * population.shape[0] / target_shape[0]).astype(int)
    xs = (np.arange(target_shape[1]) * population.shape[1] / target_shape[1]).astype(int)
    ys = np.clip(ys, 0, population.shape[0] - 1)
    xs = np.clip(xs, 0, population.shape[1] - 1)
    resampled = population[np.ix_(ys, xs)].astype(float)
    total_orig = float(population.sum())
    total_new = float(resampled.sum())
    if total_new > 0:
        resampled *= total_orig / total_new
    return resampled


def exposed_population(
    footprint: np.ndarray,
    population: np.ndarray,
) -> dict:
    """Population under the flood footprint, with sanity checks."""
    if footprint.shape != population.shape:
        raise ValueError("footprint and population must share a shape")
    footprint = footprint.astype(bool)
    pop_in = float(population[footprint].sum())
    pop_total = float(population.sum())
    return {
        "exposed_population": round(pop_in, 1),
        "total_population": round(pop_total, 1),
        "fraction_exposed": round(pop_in / pop_total, 6) if pop_total else 0.0,
    }


def exposure_report(
    fluvial: np.ndarray,
    coastal: np.ndarray,
    compound: np.ndarray,
    population: np.ndarray,
    cellsize_m: float = 30.0,
) -> dict:
    """Full exposure breakdown for the three hazard layers."""
    out = {}
    for name, mask in (("fluvial", fluvial), ("coastal", coastal), ("compound", compound)):
        r = exposed_population(mask, population)
        r["area_km2"] = round(float(mask.sum()) * cellsize_m**2 / 1e6, 3)
        out[name] = r

    def intersect(a, b):
        return a.astype(bool) & b.astype(bool)

    both = intersect(fluvial, coastal)
    out["coincident"] = exposed_population(both, population)
    out["coincident"]["area_km2"] = round(float(both.sum()) * cellsize_m**2 / 1e6, 3)
    return out
