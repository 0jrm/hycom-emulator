# SSH structure score of free runs

> **Caveat (every number below).** The free runs use the true future wind, sunlight and boundary band. Their skill is optimistic, and these structure scores inherit that until issue-time-only forcing lands.

`hycom_emulator.structure` asks whether a free run keeps the eddies and the Loop Current (LC) behaviour of the reanalysis. Pointwise RMSE does not answer this. It is evaluation only and adds no loss term. It ports the method of the persistence-ocean project, where closed-contour eddy detection on SSH is 0-dimensional persistent homology.

## What it computes

The inputs are SSH of model and truth at every lead, from `rollout_rea fields`. Existing rollouts keep only per-lead statistics, so the fields must be written once.

1. **Eddy census.** It computes the superlevel persistence (maxima: anticyclones and the LC) and sublevel persistence (minima: cyclones) of SSH. A feature counts when its persistence is at least the floor. For model and truth, at each lead, it reports the count, the median and p90 persistence, and the summed persistence.
2. **Diagram distance.** It computes W1 and bottleneck distances (L-infinity ground metric) between the diagrams of model and truth, over pairs above the floor only. It computes the same distance from the start state to the truth (persistence) as the baseline.
3. **Loop Current.** It demeans SSH over the Gulf. The seed is the highest region cell in the Yucatan box (W -86.6, E -85.2, S 21.5, N 22.0). On the 0.25° grid there are 10 such cells; a test checks this.
   - The same Kruskal pass gives each cell's neck saddle (the level at which it joins the seed) and its neck depth, min(h, seed-side peak) - saddle.
   - Detachments come from the 17 cm contour. A rejoin counts when saddle - 17 cm stays at or above the LC floor for more than 8 days in a row. A detachment is a separation when no counted rejoin follows within 30 days.
   - Every event records the neck depth from 20 days before detachment to the end of tracking (`events.json`).
4. **Noise floor** (`structure floor`). It runs a correlated-noise null: Gaussian noise of 3.5 cm with a Gaussian covariance of 50 km. The floor is the 95th percentile of the persistence of the noise's features.

Scored water is the fields' region (Gulf, ocean, outside the boundary band) deeper than 500 m. On the shelf, wind set-up and set-down make SSH extrema that are not eddies. On 2022-09-28 (Ian), the shelf added up to ten minima above the floor to one day of truth.

## Grid and floor: regrid to 0.25°

Scores are computed on the DUACS 0.25° cells (centres at k/4 + 1/8). Each cell is the mean of the emulator cells it holds. A cell is kept when at least half of its emulator cells are scored water. The floors are the calibrated persistence-ocean values: 5.5 cm per eddy and 4.3 cm per LC feature. The reasons:

- The floor was calibrated on 0.25° DUACS. On another grid the same null gives a different number. Our null gives 7.9 cm at 0.25° and 6.4 cm on the 0.08° emulator grid (`floor.json`), so a floor moved between grids without recalibration would be wrong by about 20%.
- Our null does not reproduce the reference's 5-6 cm (it gives 7.9 cm). The covariance shape, the quantile or what is pooled must differ from `null_model.py`. Until that code is in hand, a native-grid recalibration would rest on an unchecked null. The 0.25° route needs no null at all.
- The 0.25° grid is also where the LC rules (the ten Yucatan cells, 17 cm) were defined.

`--native` scores on the 0.08° grid with the floor from our null. That is a sensitivity check, not the score.

## First result: revert-fp16, 13 free runs of 60 days

The checkpoint is revert-fp16 (card emu-rea-revert, md5 6192b343). Each of the 13 starts is 20 days before a 2022-2024 truth separation, and 2024-07-01 is added. The run was inference only, on skynet GPU 3, and took 35 s. The fields reproduce the existing `rollout-h15/stats.nc` SSH RMSE at the same starts to a relative 2.4e-7. Outputs are in `skynet:/scratch/jmiranda/hycom-emulator-runs/structure-score/` (`revert-fp16/score-quarter/report.md`, `score-native/`, `truth-2022-2024/`, `floor.json`). The code is 642b903.

Model/truth ratios of means over starts, on the 0.25° grid:

| lead (d) | 1 | 7 | 14 | 21 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|
| maxima count | 1.00 | 0.86 | 0.97 | 0.89 | 0.88 | 0.74 | 0.60 |
| maxima summed persistence | 0.99 | 0.94 | 0.94 | 0.88 | 0.84 | 0.79 | 0.73 |
| maxima W1, model / persistence | 0.86 | 0.93 | 0.91 | 0.84 | 0.83 | 0.90 | 0.91 |
| minima count | 1.02 | 0.88 | 0.82 | 0.80 | 0.80 | 0.70 | 0.75 |
| minima summed persistence | 1.00 | 0.86 | 0.78 | 0.75 | 0.75 | 0.70 | 0.71 |
| minima W1, model / persistence | 0.82 | 0.99 | 0.97 | 0.90 | 0.85 | 0.95 | 0.96 |

- **Variance collapse shows as eddies fading below the floor, cyclones first.** The truth holds 12.5-13.5 cyclones above 5.5 cm. The model falls from 12.6 to about 9.5 by day 21 and stays there; summed cyclone persistence is 0.75 of truth at day 21. Anticyclones hold to about day 30 (0.88), then fall to 0.60 of truth by day 60.
- **Diagram distance** beats persistence only modestly: 0.83-0.93 for maxima and 0.85-0.99 for minima after day 7.
- **Separations inside the runs.** Truth has 15 separations and the model has 10. Nine are matched within 15 days, with median |date error| 4 d and mean -1.6 d. Neck depth at detachment agrees when matched (truth/model, cm: 27/25, 34/33, 26/23, 30/33, 24/16).
  - Of the 8 large truth rings (≥ 40,000 km²), the model reproduces 5 (errors +7, -12, 0, -4, +1 d) and misses 3 (starts 2022-06-15, 2023-09-09, 2024-02-08).
  - The model adds one early separation the truth does not have (start 2023-03-14, lead 2, 31,000 km²).
- **The native grid reads differently for anticyclones.** At 0.08° with the 6.4 cm floor, the model has 1.14-1.19 times the truth's maxima at days 7-21, and its W1 is above persistence (1.08-1.23). Small model maxima exist at grid scale and the 0.25° mean removes them. Cyclones fade the same way on both grids (0.68 of truth at day 21).

## Truth alone, 2022-01 to 2024-08

The 17 cm rule as briefed confirms 14 separations in 32 months (`truth_events.json`). The persistence-ocean work found seven in five years of DUACS. Ring areas at detachment split into two groups. One group is 47,000-100,000 km², with 8 events (2022-01-30, 06-14, 09-07, 12-14, 2023-07-04, 09-28, 2024-02-27, 07-25). The other is 2,000-19,000 km², which are shed fragments. A ring-area threshold between the groups would match the known LC ring rate better. It is not applied here because the reference rule has no such threshold that we know of.

## Unverified

- The engine was written from the brief. It is not yet checked against persistence-ocean's `cubical_h0.py`, `merge_tree.py`, `loop_current.py` and `null_model.py`. Four readings are ours:
  - 4-connectivity.
  - The essential pair dies at its mask piece's minimum.
  - The rejoin margin is saddle - 17 cm, measured at the highest LC cell in the ring's footprint.
  - The null's covariance and quantile.

  The reference repo is on the laptop only. A git bundle (`git -C ~/work/persistence-ocean-lc bundle create /tmp/po.bundle feat/write-up-assets`, then `scp` to RCC) would settle all four.
- Separation dates for model runs come from 60-day runs. Only detachments by lead 21 are always decidable, and the matching window (15 days) is our choice.
- The result comes from one checkpoint, one seed and 13 starts. It is a demonstration of the score, not a verdict on revert-fp16.
- This GPU inference ran without a question card. It was an evaluation of an existing checkpoint at the user's request, and it trained nothing.
- The optional temperature front score is not built.
