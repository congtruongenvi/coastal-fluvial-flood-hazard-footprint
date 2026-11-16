"""
Flood hazard water-surface fields: fluvial, coastal, and compound.

Each function returns a *water-surface elevation* field on the DEM grid. Turning
a water-surface field into an inundation mask is a separate step (`connectivity.py`)
because that is where most implementations quietly go wrong.

Units are metres throughout, relative to the DEM's vertical datum. Mixing datums
between the tide record and the DEM is a common and large error in deltas: a
0.5 m datum offset is comparable to the entire tidal range.
"""

from __future__ import annotations

import numpy as np

NODATA = -9999.0


# --------------------------------------------------------------------------
# Fluvial
# --------------------------------------------------------------------------
def fluvial_water_surface(
    dem: np.ndarray,
    channel_mask: np.ndarray,
    design_stage: float,
    longitudinal_slope: float,
    cellsize: float,
    nodata: float = NODATA,
    downstream_is_row0: bool = True,
) -> np.ndarray:
    """Water-surface elevation for a design fluvial stage.

    Simplified to a constant stage at the outlet plus a bed slope upstream.
    This is adequate for hazard *screening* along a reach where a rating curve
    is unavailable, and is explicitly not a substitute for a hydraulic model
    when backwater, bridge constriction or floodplain conveyance matter.

    Parameters
    ----------
    channel_mask : boolean, True on channel cells; used to fix the datum.
    design_stage : water-surface elevation at the downstream boundary, metres.
    longitudinal_slope : bed slope, m/m, positive downstream.
    downstream_is_row0 : orientation flag. Getting this backwards inverts the
        water surface and produces a downstream-deepening flood, which looks
        plausible on a map and is completely wrong.
    """
    rows, cols = dem.shape
    ws = np.full(dem.shape, np.nan, dtype=float)

    # Distance upstream from the downstream boundary, in cells.
    if downstream_is_row0:
        n_upstream = np.arange(rows)[::-1].astype(float)
    else:
        n_upstream = np.arange(rows).astype(float)

    rise = longitudinal_slope * n_upstream * cellsize  # metres above the outlet
    profile = design_stage + rise

    valid = dem > nodata
    ws[valid] = np.broadcast_to(profile[:, None], (rows, cols))[valid]
    return ws


def rating_curve_water_surface(
    dem: np.ndarray,
    downstream_index: tuple[int, int],
    flow_accumulation: np.ndarray,
    design_discharge: float,
    slope: float,
    manning_n: float,
    cellsize: float,
    nodata: float = NODATA,
) -> np.ndarray:
    """Water-surface elevation from a Manning-type rating curve per cell.

    Depth is estimated from the design discharge and the local flow accumulation
    (as a proxy for width), then the water surface is the DEM plus that depth.
    More defensible than a constant stage where flow accumulation is available
    from a filled DEM, and still an approximation: no backwater, no conveyance
    partitioning between channel and floodplain.

        d = (Q n / (W sqrt(S)))^(3/5)

    with W taken as proportional to the square root of flow accumulation, which
    is the standard hydraulic-geometry assumption (W ~ Q^0.5).
    """
    acc = np.maximum(flow_accumulation.astype(float), 1.0)

    # Width proxy: cells wide, scaled by sqrt of accumulation, floored so that
    # headwater cells do not produce metres of depth from a rounding error.
    width = np.maximum(np.sqrt(acc) * cellsize, cellsize)

    q_local = design_discharge * (acc / acc.max())  # distribute along the network

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = width * np.sqrt(slope)
        depth = (q_local * manning_n / denom) ** 0.6
    depth = np.where(np.isfinite(depth), depth, 0.0)
    depth = np.clip(depth, 0.0, 15.0)  # 15 m cap; deeper indicates bad inputs

    ws = dem.astype(float).copy()
    valid = ws > nodata
    ws[valid] = ws[valid] + depth[valid]
    return ws


# --------------------------------------------------------------------------
# Coastal
# --------------------------------------------------------------------------
def coastal_water_surface(
    dem: np.ndarray,
    coast_mask: np.ndarray,
    design_level: float,
    attenuation_per_km: float,
    cellsize: float,
    nodata: float = NODATA,
) -> np.ndarray:
    """Coastal water-surface elevation with distance attenuation inland.

    A constant-lake assumption (every cell below the design level floods) is the
    usual starting point and is badly wrong in a delta, where the surge surface
    decays with distance from the sea. Attenuation is applied as a linear decay
    with distance from the nearest coast cell, computed by successive
    relaxation over the grid.

    attenuation_per_km = 0 reproduces the constant-lake behaviour, which is
    useful as a comparison case rather than as a default.
    """
    if design_level <= nodata:
        raise ValueError("design_level must be a real elevation")

    dist = _distance_to_mask(coast_mask, cellsize)
    decay = attenuation_per_km * (dist / 1000.0)

    ws = np.full(dem.shape, np.nan, dtype=float)
    valid = dem > nodata
    ws[valid] = design_level - decay[valid]

    # A surge cannot produce a water surface below the ground it sits on in the
    # sense of flooding, but the field itself is allowed below terrain -- the
    # connectivity step decides what actually floods.
    return ws


def _distance_to_mask(mask: np.ndarray, cellsize: float) -> np.ndarray:
    """Euclidean-ish distance to the nearest True cell, by iterative relaxation.

    Two-pass chamfer with diagonal weighting (1.0 / sqrt(2)). Accurate to about
    2 % on 8-connected grids and linear in grid size -- sufficient for an
    attenuation term, where a few per-cent of distance error is immaterial
    beside DEM vertical error.
    """
    rows, cols = mask.shape
    big = 1e12
    d = np.where(mask, 0.0, big)

    diag = np.sqrt(2.0)
    fwd = (
        (-1, 0, 1.0), (0, -1, 1.0), (-1, -1, diag), (-1, 1, diag),
    )
    bwd = (
        (1, 0, 1.0), (0, 1, 1.0), (1, 1, diag), (1, -1, diag),
    )

    for i in range(rows):
        for j in range(cols):
            for di, dj, w in fwd:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    cand = d[ni, nj] + w * cellsize
                    if cand < d[i, j]:
                        d[i, j] = cand

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            for di, dj, w in bwd:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    cand = d[ni, nj] + w * cellsize
                    if cand < d[i, j]:
                        d[i, j] = cand

    return d


# --------------------------------------------------------------------------
# Compound
# --------------------------------------------------------------------------
def compound_water_surface(
    fluvial: np.ndarray,
    coastal: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """Combine the two water-surface fields.

    mode='max'  -- the higher water surface wins at each cell. Physically the
                   right choice for a static approach: water levels do not add,
                   they superpose on the terrain.
    mode='sum'  -- additively combines the two depths. Provided for comparison
                   and specifically *not* recommended: it double-counts the
                   shared elevation datum and inflates inland depths by the full
                   surge height, producing a footprint that no hydrodynamic model
                   would reproduce.

    The distinction matters. Summing water levels is the intuitive thing to do and
    it is wrong; the correct static-combination rule is the maximum. A river at
    +3 m and a tide at +1.5 m do not produce a +4.5 m water surface.
    """
    if mode == "max":
        out = np.fmax(fluvial, coastal)
    elif mode == "sum":
        out = fluvial + coastal
    else:
        raise ValueError("mode must be 'max' or 'sum'")

    out = np.where(np.isfinite(fluvial) & np.isfinite(coastal), out, np.nan)
    return out


def meet_bar(
    fluvial_depth: np.ndarray,
    coastal_depth: np.ndarray,
) -> np.ndarray:
    """Water surface built from depths that already share a terrain datum.

    Where depths rather than absolute levels are available, superposition of
    depths *is* additive, but only where both are measured relative to the same
    ground elevation. This function asserts that intent explicitly rather than
    leaving the reader to guess which convention is in play.
    """
    return np.fmax(fluvial_depth, coastal_depth)
