"""
Terrain conditioning for flood hazard mapping.

Deliberately written against plain NumPy arrays rather than a geospatial raster
library, so the hydrologically important logic -- sink filling, flat resolution,
connectivity -- can be read, tested and criticised without a GDAL install. The
I/O wrappers at the bottom optionally use rasterio.

Every function here operates on a 2-D float array of elevations with a nodata
sentinel. No coordinate assumptions: the DEM's geotransform is carried separately
by the caller.
"""

from __future__ import annotations

import numpy as np

# Elevations at or below this are treated as nodata. Real DEMs use values near
# -32768 or -9999; a large negative sentinel avoids colliding with genuine
# below-sea-level terrain, which matters in deltas.
NODATA = -9999.0


# --------------------------------------------------------------------------
# Sink filling
# --------------------------------------------------------------------------
def fill_sinks(dem: np.ndarray, nodata: float = NODATA) -> np.ndarray:
    """Fill closed depressions by priority-flood.

    Why this is needed
    ------------------
    A raw DEM almost always contains spurious pits from interpolation error and
    from radar/optical noise. In a flood model, a spurious pit becomes a spurious
    lake: water routed into it never leaves, and the inundation mask gains a blob
    that has no hydraulic justification. Filling first is standard practice and
    skipping it is a classic beginner error.

    Method
    ------
    Priority-flood (Barnes et al., 2014). Seed a priority queue with the DEM
    border, always process the lowest cell first, and raise any neighbour below
    the current cell to the current cell's level plus an epsilon. The epsilon
    prevents creating a perfectly flat filled surface, which the downstream
    flow-direction step cannot route across.

    Implementation note: this uses `heapq` over a flat index. It is O(n log n)
    and adequate to a few million cells. For continental DEMs, use the
    I/O-wrapped variants below with block processing.
    """
    import heapq

    dem = dem.astype(float).copy()
    rows, cols = dem.shape
    visited = np.zeros(dem.shape, dtype=bool)

    # Border cells, excluding nodata, seed the queue.
    heap: list[tuple[float, int, int]] = []
    for i in range(rows):
        for j in (0, cols - 1):
            if dem[i, j] > nodata:
                heapq.heappush(heap, (dem[i, j], i, j))
                visited[i, j] = True
    for j in range(cols):
        for i in (0, rows - 1):
            if dem[i, j] > nodata and not visited[i, j]:
                heapq.heappush(heap, (dem[i, j], i, j))
                visited[i, j] = True

    eps = 1e-4  # metres; small relative to DEM precision, large enough to route
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while heap:
        elev, i, j = heapq.heappop(heap)
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and not visited[ni, nj]:
                visited[ni, nj] = True
                if dem[ni, nj] <= nodata:
                    continue
                if dem[ni, nj] <= elev:
                    dem[ni, nj] = elev + eps
                heapq.heappush(heap, (dem[ni, nj], ni, nj))

    return dem


# --------------------------------------------------------------------------
# Slope
# --------------------------------------------------------------------------
def slope_horn(dem: np.ndarray, cellsize: float, nodata: float = NODATA) -> np.ndarray:
    """Slope in degrees, Horn (1981) 3x3 kernel.

    Horn's method is used rather than the simpler central difference because it
    is what GDAL, ArcGIS and SAGA implement, so results can be cross-checked
    against those tools without a systematic offset.
    """
    dem = dem.astype(float)
    out = np.full(dem.shape, np.nan)

    # Interior only; edges are left NaN rather than extrapolated.
    z = dem[1:-1, 1:-1]
    a = dem[:-2, :-2]; b = dem[:-2, 1:-1]; c = dem[:-2, 2:]
    d = dem[1:-1, :-2];                    f = dem[1:-1, 2:]
    g = dem[2:, :-2];  h = dem[2:, 1:-1];  i = dem[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)

    # 2.0 factor converts from the 45-degree kernal convention to true aspect
    # north; standard in Horn's formulation.
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2) / 1.0)
    out[1:-1, 1:-1] = np.degrees(slope_rad)

    # Propagate nodata.
    bad = dem <= nodata
    out[bad] = np.nan
    return out


# --------------------------------------------------------------------------
# Height Above Nearest Drainage
# --------------------------------------------------------------------------
def compute_hand(
    dem: np.ndarray,
    drainage_mask: np.ndarray,
    nodata: float = NODATA,
) -> np.ndarray:
    """Height Above Nearest Drainage.

    HAND is the elevation difference between a cell and the nearest cell on the
    drainage network, measured along the flow path rather than as a straight-line
    distance. It is the single most useful terrain covariate for flood extent:
    a cell 1 m above the river 200 m away floods; a cell 5 m above it does not,
    regardless of how far away it is.

    This implementation walks the flow path using a steepest-descent assignment
    from a priority-flood over the drainage mask -- equivalent to the standard
    approach and, again, dependency-free.

    Parameters
    ----------
    dem : 2-D elevation array (should already be sink-filled)
    drainage_mask : boolean array, True where a cell is part of the drainage
        network (channel, or the river/coast cells used as flood sources)

    Returns
    -------
    HAND in the DEM's vertical units, NaN where nodata.
    """
    import heapq

    dem = dem.astype(float)
    rows, cols = dem.shape
    hand = np.full(dem.shape, np.nan, dtype=float)
    visited = np.zeros(dem.shape, dtype=bool)

    heap: list[tuple[float, int, int]] = []
    for i, j in zip(*np.where(drainage_mask)):
        if dem[i, j] > nodata:
            hand[i, j] = 0.0
            visited[i, j] = True
            heapq.heappush(heap, (dem[i, j], i, j))

    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))

    while heap:
        elev, i, j = heapq.heappop(heap)
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and not visited[ni, nj]:
                visited[ni, nj] = True
                if dem[ni, nj] <= nodata:
                    continue
                # HAND is relative to the drainage cell, so it accumulates as
                # the maximum difference along the path to the nearest channel.
                delta = dem[ni, nj] - dem[i, j]
                hand[ni, nj] = max(hand[i, j] + max(delta, 0.0), 0.0)
                # Traverse by elevation, which is the priority-flood invariant.
                heapq.heappush(heap, (dem[ni, nj], ni, nj))

    return hand


# --------------------------------------------------------------------------
# I/O wrappers (optional dependency)
# --------------------------------------------------------------------------
def read_dem(path: str) -> tuple[np.ndarray, dict]:
    """Read a DEM with rasterio, returning the array and its profile.

    Kept at the bottom so the hydrology above stays importable without GDAL.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rasterio is required for DEM I/O. Install with:\n"
            "    uv pip install rasterio\n"
            "The terrain functions themselves need only NumPy."
        ) from exc

    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        profile = src.profile.copy()
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = NODATA
    return arr, profile


def write_raster(path: str, array: np.ndarray, profile: dict) -> None:
    """Write a single-band float raster, preserving the input georeferencing."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio is required for raster writing.") from exc

    profile = dict(profile)
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
