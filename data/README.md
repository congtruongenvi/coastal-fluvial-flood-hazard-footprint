# data/

Nothing in this directory is committed to git.

## raw/<case>/

Expected files for a real run:

| File | Contents |
|---|---|
| `dem.tif` | DEM, 30 m preferred (Copernicus GLO-30 / ALOS PALSAR) |
| `coast_mask.tif` | Boolean raster, True on coastal source cells |
| `discharge.csv` | `time,station,discharge_m3s` — design quantiles |
| `tide_levels.csv` | `time,level_m` — design still-water levels |
| `population.tif` | WorldPop / GPW, same grid as DEM |

Run with:

```bash
python src/pipeline.py --case <case-name>
```

## processed/

See `README.md`. Output footprints and exposure reports land here.
