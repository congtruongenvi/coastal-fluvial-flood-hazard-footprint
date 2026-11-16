# Coastal–Fluvial Flood Hazard Footprint

Joint **coastal and fluvial flood exposure mapping** for deltaic provinces, combining
tide-driven coastal inundation with river-discharge-driven fluvial flooding into a
single hazard-footprint layer that includes the **compound** case where both drivers
coincide — the situation that produces the largest losses and the one that single-driver
hazard maps systematically under-represent.

---

## Status

**Working hazard-mapping pipeline, connectivity-constrained, pre-results stage.** The
flood-fill connectivity logic — the part that separates a defensible footprint from a
naive elevation-below-water mask — is implemented and validated (12 self-tests). Real
layers (DEM, tide, discharge, population) are **not committed**; outputs shown in figures
are generated from synthetic fixtures. Real-data processing requires the case directory
under `data/raw/<case>/` described in `data/README.md`.

This repository demonstrates **hierarchical method implementation and verification**,
not a finished hazard atlas. Do not cite `figures/` outputs as observed inundation.

---

## The problem this addresses

Standard flood hazard mapping treats the two drivers separately:

- **Fluvial maps** assume a design discharge and route it downstream.
- **Coastal maps** assume a design storm surge or tide level.

In a delta, both act on the same land surface at the same time. A river at bankfull
meeting a high tide inverts the water-surface slope, so drainage stops and water
backs up. The joint footprint is larger than either map, and the *marginal* area —
land flooded only because both drivers coincided — is where the unmodelled losses live.

This repository quantifies that marginal area.

---

## Method

| Step | Approach |
|---|---|
| Terrain | DEM conditioning: fill sinks, resolve flats, compute slope and HAND |
| Fluvial hazard | Water-surface elevation from a normal-depth or rating-curve approximation per reach |
| Coastal hazard | Tide + surge elevation propagated inland with a connectivity constraint |
| **Compound** | Union of both, **restricted to terrain hydrologically connected to a water body** |
| Exposure | Overlay with population and built-up layers at matching resolution |
| Footprint | Area statistics in km² by administrative unit, and the marginal compound area |

### The connectivity constraint is not optional

A naive "elevation below water level" mask floods every inland depression that happens
to sit below the design water level — including basins with no hydraulic connection to
the coast or the river. In the Mekong Delta this over-predicts inundation by a large
and *spatially biased* margin.

The fix is a flood-fill from the water body across cells whose elevation is below the
water surface, which is what `connectivity.py` implements. It is the single most
important step in the pipeline and the one most often skipped.

---

## Repository layout

```
.
├── src/
│   ├── terrain.py        # DEM conditioning, slope, HAND
│   ├── hazard.py         # fluvial, coastal, and compound water-surface fields
│   ├── connectivity.py   # flood-fill from source water bodies
│   ├── exposure.py       # population / built-up overlay
│   └── pipeline.py       # end-to-end run
├── notebooks/
│   └── 01_hazard_footprint_workflow.ipynb
├── figures/
├── data/                 # DEM, tide, discharge, population (not committed)
└── requirements.txt
```

---

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# Full run against real layers placed in data/raw/
python src/pipeline.py --case mekong-delta

# Synthetic self-test: no data files needed, validates the connectivity logic
python src/pipeline.py --selftest
```

---

## Data sources

| Layer | Source | Notes |
|---|---|---|
| DEM | Copernicus DEM GLO-30 / ALOS PALSAR | 30 m; blend them where one has voids |
| Tide / surge | National tide-gauge records + regional surge model | Astro tide separated from residual |
| Discharge | Gauge network, or a hydrological model where gauges are sparse | Design quantiles per return period |
| Population | WorldPop / GPW | Dasymetric, better than administrative apportionment |
| Built-up | GHSL | Epoch-matched to the flood scenario |
| Administrative units | GADM | For reporting footprints by province |

---

## Limitations, stated plainly

- **Static inundation, not hydrodynamics.** This is a "bathtub plus connectivity"
  approach. It does not resolve timing, flow velocity, or the duration of inundation.
  Where those matter — and for flood *risk*, not just flood *extent*, they do — a
  hydrodynamic model (MIKE 21, Delft3D, HEC-RAS 2D) is required. This pipeline is
  designed to produce the hazard-footprint layer such a model can be validated against,
  and to do it cheaply across many scenarios where running full hydrodynamics would be
  prohibitive.
- **DEM vertical accuracy dominates the result.** A 2 m vertical error in a delta with
  1 m of tidal range changes the footprint more than any modelling choice downstream.
  Sensitivity to DEM error is reported, not assumed away.
- **Tide–surge interaction is not modelled explicitly.** The design water level is a
  superposition, which is conservative in shallow estuaries and could be either sign
  elsewhere.
- **Exposure is not risk.** Population in the flood footprint is *exposure*. Turning it
  into risk requires vulnerability and adaptive capacity, which are not in this repository.

---

## Author

**Truong Cong Truong** — MSc Environmental Science (VNU-HCM), 15+ years in numerical
hydrology, hydrodynamic modelling (MIKE 11, MIKE SHE, ISIS) and water-resources
assessment across the Mekong Delta and Dong Nai River basin.

*Python · GDAL/rasterio · geopandas · DEM processing · exposure analysis*
