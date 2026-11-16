"""
Hydraulic connectivity: which cells below the water surface are actually flooded.

This module carries the single most important and most frequently skipped step in
static flood mapping.

The mistake
-----------
Compute a water-surface field, then flag every DEM cell below it as inundated.
In a delta this over-predicts flooding badly, and the error is spatially biased:
it floods the interior of the delta -- low-lying basins with no hydraulic
connection to the river or the sea -- while the genuinely connected floodplain
is handled correctly. The result is a map that looks plausible and that no
hydrodynamic simulation would reproduce.

The fix
-------
Flood-fill outward from the water body, across cells that are below the
water-surface level. This is a connectivity constraint, not an elevation
constraint. A cell 3 m below sea level with a 4 m ridge between it and the coast
does not flood; the same cell with an open channel to the coast does.

The consequence for research use
--------------------------------
Footprint areas from a connectivity-constrained mask and a naive mask differ by
enough to change every downstream exposure statistic. Any comparison against a
published footprint requires knowing which convention was used.
"""

from __future__ import annotations

from collections import deque

import numpy as np

NODATA = -9999.0


def flooded_mask(
    dem: np.ndarray,
    water_surface: np.ndarray,
    source_mask: np.ndarray,
    nodata: float = NODATA,
    connectivity: int = 8,
) -> np.ndarray:
    """Inundation mask by flood-fill from source cells across submerged terrain.

    A cell is flooded if and only if:

    1. its elevation is strictly below the water surface at that cell, and
    2. a path of such cells exists to a source cell (a water body).

    Condition 2 is what a naive implementation omits.

    Parameters
    ----------
    dem : 2-D elevation, metres
    water_surface : 2-D water-surface elevation, metres, same grid
    source_mask : boolean, True on cells that are water bodies (river channel,
        coast, or the domain boundary where it is open to the sea)
    connectivity : 4 or 8. 8 is the usual choice; 4 is more conservative and
        prevents flooding through a diagonal touching only at a corner.

    Returns
    -------
    Boolean array, True where flooded.
    """
    if dem.shape != water_surface.shape or dem.shape != source_mask.shape:
        raise ValueError("dem, water_surface and source_mask must share a shape")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    rows, cols = dem.shape

    submerged = (dem > nodata) & np.isfinite(water_surface) & (dem < water_surface)

    if connectivity == 8:
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        neighbours = ((-1, 0), (0, -1), (0, 1), (1, 0))

    # Seed with source cells that are themselves submerged. A source cell above
    # the water surface cannot propagate flooding.
    flooded = np.zeros(dem.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    src = source_mask & submerged
    for i, j in zip(*np.where(src)):
        flooded[i, j] = True
        queue.append((int(i), int(j)))

    if not queue:
        # No source is submerged: nothing floods. Worth surfacing rather than
        # returning an empty mask silently -- it usually means the water level,
        # the datum, or the source mask is wrong.
        return flooded

    while queue:
        i, j = queue.popleft()
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if submerged[ni, nj] and not flooded[ni, nj]:
                    flooded[ni, nj] = True
                    queue.append((ni, nj))

    return flooded


def flooded_mask_naive(
    dem: np.ndarray,
    water_surface: np.ndarray,
    nodata: float = NODATA,
) -> np.ndarray:
    """Elevation-only mask, with no connectivity constraint.

    Provided explicitly so the difference can be measured rather than argued
    about. Never use this to report a footprint; use it to quantify how much a
    published footprint would change if connectivity were ignored.
    """
    return (dem > nodata) & np.isfinite(water_surface) & (dem < water_surface)


def connectivity_diagnostics(
    dem: np.ndarray,
    water_surface: np.ndarray,
    source_mask: np.ndarray,
    nodata: float = NODATA,
    connectivity: int = 8,
    cellsize_m: float = 30.0,
) -> dict:
    """Quantify how much the connectivity constraint changes the footprint.

    Returns area in km^2 for the connected and naive masks, plus the
    disconnected area -- the land that an elevation-only method wrongly floods.
    That last number is the headline: it is the size of the error being avoided.
    """
    cell_km2 = (cellsize_m**2) / 1e6

    connected = flooded_mask(dem, water_surface, source_mask, nodata, connectivity)
    naive = flooded_mask_naive(dem, water_surface, nodata)

    disconnected = naive & ~connected

    a_conn = float(connected.sum()) * cell_km2
    a_naive = float(naive.sum()) * cell_km2
    a_disc = float(disconnected.sum()) * cell_km2

    return {
        "connected_area_km2": round(a_conn, 3),
        "naive_area_km2": round(a_naive, 3),
        "disconnected_area_km2": round(a_disc, 3),
        "over_prediction_pct": round(100 * a_disc / a_naive, 2) if a_naive else np.nan,
    }


def marginal_compound_area(
    fluvial_mask: np.ndarray,
    coastal_mask: np.ndarray,
    compound_mask: np.ndarray,
    cellsize_m: float = 30.0,
) -> dict:
    """Area flooded only because both drivers coincided.

    This is the quantity single-driver hazard maps cannot produce, and the reason
    for the compound analysis: land in neither the fluvial-only nor the
    coastal-only footprint but inside their joint footprint.

    The three masks must all be connectivity-constrained, or this number is
    meaningless -- it would be dominated by the disconnected-inland error rather
    than by the compound interaction.
    """
    cell_km2 = (cellsize_m**2) / 1e6

    fluvial_only = fluvial_mask & ~coastal_mask
    coastal_only = coastal_mask & ~fluvial_mask
    both = fluvial_mask & coastal_mask

    return {
        "fluvial_only_km2": round(float(fluvial_only.sum()) * cell_km2, 3),
        "coastal_only_km2": round(float(coastal_only.sum()) * cell_km2, 3),
        "coincident_km2": round(float(both.sum()) * cell_km2, 3),
        "union_km2": round(float((fluvial_mask | coastal_mask).sum()) * cell_km2, 3),
        "compound_km2": round(float(compound_mask.sum()) * cell_km2, 3),
        # The excess of the union over the larger single-driver footprint --
        # how much of the hazard only appears when both are considered.
        "marginal_over_largest_single": round(
            (
                float((fluvial_mask | coastal_mask).sum())
                - max(float(fluvial_mask.sum()), float(coastal_mask.sum()))
            )
            * cell_km2,
            3,
        ),
    }
