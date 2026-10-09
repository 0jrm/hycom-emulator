# Loop Current front: 1.5 kt isotach

Code: `src/hycom_emulator/isotach.py`. Comparison: `scripts/isotach_compare.py`.

## Definition

The front is the edge of the set where total surface speed is at least 0.7717 m/s (1.5 kt). Edge pixels are fast pixels with a slow 4-neighbour. They are found on the whole grid and then kept only inside a region. The default region is ocean deeper than 500 m, west of 81W and north of 21.5N. Any boolean mask can replace it.

`front_distance` returns three distances in km between the predicted and true edge pixels:

- `hausdorff_km` is the larger of the two directed maxima.
- `mean_km` is the mean of the pooled directed distances, pred to true and true to pred.
- `p95_km` is the 95th percentile of the pooled distances (the medpy hd95 convention).

When either front is empty, all three are nan and `status` reads `true_empty`, `pred_empty` or `both_empty`.

The grid is Mercator, not a regular lat-lon grid. Longitude steps by 0.04 deg and latitude by 0.04 cos(lat) deg, so cells are square. Their side is 4.23 km at 18N and 3.77 km at 32N. The Euclidean distance transform runs in cell units and is scaled by the cell size at each query pixel. A single mid-latitude constant would be off by up to 6 percent at the domain's edges.

## Surface velocity

Layer u/v are baroclinic, so total = `u_k01 + ubaro`, `v_k01 + vbaro`. Three pieces of evidence support this:

- The store reads the `u-vel.`/`v-vel.` records of the archv snapshot unchanged (`build_store.S00_LAYER`).
- HYCOM-tools `archv2data3z.f` adds `ubaro` to `u` only for `artype.eq.1` (archv) when total velocity is requested ("mean archives already contain total velocity").
- On the abozec_053 pack at 2025-07-13, deeper than 500 m, the thickness-weighted vertical mean of layer u has an rms of 0.006 m/s, against 0.121 m/s for `ubaro`. For v the figures are 0.005 and 0.169 m/s.

u/v stay on the C-grid in the store and the pack. `surface_speed` averages each pair of faces to the p-point.

## Surrogate loss

This follows Karimi and Salcudean (2019):

    p = sigmoid((speed_pred - 0.7717) / tau)
    loss = mean over region of (p - q)^2 * (d_q^alpha + d_p^alpha)

`q` is the hard true mask. `d_q` is the distance in km to the true front, precomputed per target with `isotach_target`. `d_p` is the distance to the front of the detached hard predicted mask, computed with scipy on CPU and carrying no gradient. The defaults are tau = 0.05 m/s and alpha = 2, so the loss is in km^2. The function accepts neural-lam flat `(B, N)` input (with `grid_shape`) or `(B, ny, nx)` input.

## Cost

These are medians per call on skynet CPU (4 threads) for the 385 x 525 grid, batch 1:

| call | ms |
|---|---|
| `front_distance` | 16 |
| `isotach_target` (once per target) | 8 |
| `isotach_loss` forward | 11 |
| `isotach_loss` backward | 1 |

In training, the forward cost scales with batch size, because `d_p` takes one CPU distance transform per sample.

## Persistence baseline (abozec_053 pack)

Truth(t) is used as the forecast of truth(t+L), for every pair of days inside the val (2025-08-06..15) and test (2025-08-21..09-01) splits. That gives 20 samples at L = 1 and 18 at L = 2. Every sample had both fronts present. All values below are medians, with distances in km and the surrogate in km^2.

| region | L | Hausdorff | mean | p95 | surrogate |
|---|---|---|---|---|---|
| Gulf, > 500 m | 1 | 100.8 | 7.3 | 20.3 | 9.5 |
| Gulf, > 500 m | 2 | 114.9 | 11.1 | 29.3 | 21.7 |
| east of 90W | 1 | 69.3 | 6.0 | 16.1 | 4.7 |
| east of 90W | 2 | 71.9 | 9.1 | 24.4 | 13.5 |

Spearman correlation of the surrogate with each distance across samples:

| region | L | Hausdorff | mean | p95 |
|---|---|---|---|---|
| Gulf, > 500 m | 1 | 0.40 | 0.65 | 0.54 |
| Gulf, > 500 m | 2 | 0.42 | 0.81 | 0.76 |
| east of 90W | 1 | 0.71 | 0.83 | 0.68 |
| east of 90W | 2 | 0.03 | 0.84 | 0.85 |

The surrogate tracks the mean and p95 distances. It does not reliably track the maximum. The maximum is set by a few pixels. In the smoke run, the worst pixels were a one-pixel fast patch near 89.7W 24.4N and a speed patch near the threshold on the western boundary current at 97.3W 22.1N, not the Loop Current itself. The mean and p95 shifts of 6 to 11 km/day and 16 to 29 km/day fit Loop Current front motion.

## Reanalysis read-out

`evaluate_rea` scores the front of the model and of persistence at every lead, under `fronts` in its JSON. Each lead holds the medians of `hausdorff_km`, `mean_km` and `p95_km` over the samples where both fronts exist, and a count of each status. The metric is evaluation only. No training loss uses it.

### Surface speed

Surface speed is `hypot(u_0m, v_0m)`. The reanalysis pack (`rea_pack.py`) reads u and v from the product's z-level files (`gomb4_daily_*_3z.nc`), not from archv layers. These values are total velocity at p-points, for three reasons:

- Each daily file is an `ncra` mean of hourly `038_archv.*_3z.nc` files from `archv2ncdf3z` (the `history` attribute). By default (`baclin` 0), archv2data3z adds `ubaro` to archv layer velocity and moves each layer's face velocities to the p-point (`uvp`) before it interpolates to z.
- The files have one Longitude and one Latitude axis for every variable.
- For 2024-04-09, deeper than 1000 m, the depth mean of the 3z `u` matches `u_barotropic_velocity`. The correlation is 0.997, and the rms difference is 0.006 m/s against an rms of 0.074 m/s. For v the figures are 0.997 and 0.006 m/s against 0.073 m/s. A baroclinic layer velocity would have a depth mean near zero.

So `surface_speed` does not apply here: there is no `ubaro` to add and no face to average.

### Grid and region

`front_grid` reads the grid shape and the longitude step from the pack's coordinates. The stride-2 pack is 193 x 263 with a 0.08 deg step. Its cells are 8.46 km at the southern edge and 7.55 km at the northern edge, twice the native size. The region is `gulf_region` of the pack's own static depth, lon and lat: 12931 points, none in the boundary band. `--region` does not change it.

At stride 2, a one-cell shift is about 8 km. This is larger than the 1-day persistence mean at full resolution, so compare a model with persistence on the same pack, not with the table above.

### Smoke result

These figures are for `emu-rea-s2b/scored.ckpt` on the first 24 test samples (January 2024), 4-day rollouts. Every sample had both fronts. Values are medians in km.

| forecast | lead (days) | Hausdorff | mean | p95 |
|---|---|---|---|---|
| model | 1 | 59.1 | 4.8 | 12.8 |
| model | 2 | 73.6 | 7.0 | 17.9 |
| model | 3 | 123.9 | 8.3 | 22.7 |
| model | 4 | 157.1 | 9.9 | 24.0 |
| persistence | 1 | 206.5 | 10.4 | 27.2 |
| persistence | 2 | 281.5 | 14.3 | 34.9 |
| persistence | 3 | 216.2 | 14.6 | 40.7 |
| persistence | 4 | 187.1 | 15.6 | 46.7 |

The persistence mean distances at leads 1 and 2 are about 3 km above the full-resolution baseline. The coarser cells fit that, though the season and the product also differ. The model halves the persistence mean and p95 distances at lead 1, and the gap narrows by lead 4.
