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
