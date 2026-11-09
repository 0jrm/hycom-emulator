# SSH structure score of free runs

> **Caveat (every number below).** The free runs use the true future wind, sunlight and boundary band. Their skill is optimistic, and these structure scores inherit that until issue-time-only forcing lands.

`hycom_emulator.structure` asks whether a free run keeps the eddies and the Loop Current (LC) behaviour of the reanalysis. Pointwise RMSE does not answer this. It is evaluation only and adds no loss term.

## Engine: vendored from persistence-ocean

`hycom_emulator.topology` is a minimal port of persistence-ocean (branch `feat/write-up-assets`, commit 205e3c3). It takes `cubical_h0`, `merge_tree`, `boundary`, `null_model` and `loop_current`, and leaves their logic unchanged; numba is optional. The reference repository came over as a git bundle with Jose's approval. It is used read-only outside this repository, and none of its history is pushed.

The port is checked against the reference on the reference's own data. `scripts/structure_duacs_check.py` runs the ported LC17 rules on DUACS ADT 2017-2021 (skynet `/unity/g2/jmiranda/persistence-ocean/duacs_subset`) at the reference setting. It reproduces all seven final separations of `lc_events_summary_L50_snp_central.csv`: the same dates, piece positions and pixel counts, with 10 seed cells and 21 detachments. The skynet files differ in bytes from the reference run's (they are a separate download); the events match all the same. The reference's merge-tree, connection-level and LC17 tests are ported into `tests/test_structure.py`.

## What it computes

The inputs are SSH of model and truth at every lead, from `rollout_rea fields`. Existing rollouts keep only per-lead statistics, so the fields must be written once.

1. **Eddy census.** H0 persistence with 8-connectivity and an open boundary: a component dies when it reaches land or the grid edge. This is the setting of the reference's atlas audit and of its signal-plus-noise floor. Maxima are anticyclones (and the LC); minima are cyclones. A feature counts when its persistence is at least 5.5 cm, the reference's signal-plus-noise floor per eddy in the open Gulf. Per lead it reports the count, the median and p90 persistence, and the summed persistence.
2. **Diagram distance.** W1 and bottleneck distances (L-infinity ground metric) between the model and truth diagrams, over pairs above the floor only. The same distance from the start state (persistence) to the truth is the baseline.
3. **Loop Current.** The reference rules on the 0.25° Gulf:
   - SSH minus its plain Gulf mean.
   - The seed is the highest of the 10 Gulf cells in the Yucatan box.
   - Detachments come from the 17 cm contour. Each piece is followed to its fate.
   - A rejoin counts when its touch depth (connection level to the seed minus 17 cm) is at least 4.324 cm on more than 8 days in a row.
   - A final separation is a piece north of 24 N or west of 87 W that stays off the 17 cm LC for more than 30 days without a counted rejoin.
   - Each event's neck depth, min(h_max, seed-side peak) - h_saddle, is tracked from the separation back to the run's start (at most 180 days) and 30 days forward.

The census leaves out water shallower than 500 m. In daily-mean HYCOM SSH, storm set-up and set-down on the shelf make extrema that mapped DUACS does not show: on 2022-09-28 (Ian), up to ten minima above the floor in one day of truth. The LC keeps the shelf, as the reference does.

## Grid and floor: regrid to 0.25°

Scores are computed on the DUACS 0.25° cells (centres at k/4 + 1/8). Each cell is the mean of the emulator cells it holds. A cell is kept when at least half of its emulator cells are scored water. This is the grid on which every floor and LC rule was set. The calibrated floor is the signal-plus-noise floor, and it exists only on that grid; recalibrating it natively would mean rerunning the reference's signal-plus-noise experiments on the HYCOM grid.

The pure-noise null (the reference's experiment 02, run at a constant 3.5 cm and L = 50 km) does barely depend on resolution: 5.9 cm at 0.25° and 5.8 cm at 0.08° (`floor.json`). So `--native` is a usable sensitivity check, scored with the 5.8 cm pure-noise floor. The 0.25° score stays the primary one.

## Result: revert-fp16, 13 free runs of 60 days

The checkpoint is revert-fp16 (card emu-rea-revert, md5 6192b343). The 13 starts were picked 20 days before detachments found by the first, superseded version of this module, plus 2024-07-01. The run was inference only, on skynet GPU 3, and took 35 s. The fields reproduce the existing `rollout-h15/stats.nc` SSH RMSE at the same starts to a relative 2.4e-7. Outputs are in `skynet:/scratch/jmiranda/hycom-emulator-runs/structure-score/` (`revert-fp16/score-quarter/`, `score-native/`, `truth-2022-2024/`, `floor.json`, `duacs_check.log`).

Model/truth ratios of means over starts, on the 0.25° grid:

| lead (d) | 1 | 7 | 14 | 21 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|
| anticyclone count | 0.96 | 1.02 | 0.95 | 0.87 | 0.83 | 0.82 | 0.75 |
| anticyclone summed persistence | 0.97 | 0.97 | 0.98 | 0.91 | 0.90 | 0.81 | 0.79 |
| anticyclone W1, model / persistence | 0.79 | 0.96 | 0.84 | 0.76 | 0.78 | 0.85 | 0.81 |
| cyclone count | 1.04 | 0.76 | 0.64 | 0.66 | 0.61 | 0.36 | 0.38 |
| cyclone summed persistence | 0.99 | 0.71 | 0.55 | 0.53 | 0.47 | 0.27 | 0.28 |
| cyclone W1, model / persistence | 1.14 | 1.29 | 1.28 | 1.07 | 0.93 | 0.87 | 0.97 |

- **Variance collapse is a cyclone deficit.** Truth keeps about 3.6 cyclones above 5.5 cm. The model falls from 3.7 to 2.5 within a week, to 1.8 by day 30 and to 1.2 by day 60. Summed cyclone persistence is 0.55 of truth at day 14 and 0.27 at day 45.
- **Anticyclones hold for two weeks**, then fade slowly: 0.87 of truth's count at day 21 and 0.75 at day 60.
- **Diagram distance.** For anticyclones the model is closer to truth than persistence at every lead (0.76-0.96). For cyclones it is further than persistence through day 21 (1.07-1.29): keeping the start state's cyclones beats the model's faded ones.
- **The native grid agrees on cyclones** (0.68 of truth at day 14, 0.47 at day 45). At native resolution the model has 1.18 times truth's anticyclones at day 7, small grid-scale maxima that the 0.25° mean removes.
- **LC separations inside the runs.** Truth and model are judged on the same 61-day windows. Truth has 9 final separations and the model 7; 4 match within 15 days, with errors +9, -8, +3 and -3 d. Neck depth at separation agrees when they match (truth/model, cm: 29/30, 42/38, 31/26, 34/39).
  - The model misses 5 truth separations (starts 2022-06-15, 2023-03-14, 2023-09-09, 2024-02-08, 2024-03-19).
  - It adds 3: two at lead 2-3 that truth makes at lead 20 (an early break-off), and one at lead 28 in the 2024-07-01 run.
  - A 61-day window decides "final" more often than the full record does: a piece that reattaches after the window looks final. Truth inside the windows therefore has more separations than the long series.

## Separation timing: 16 runs around the 4 truth separations

These are 60-day revert-fp16 free runs started 10, 20, 30 and 40 days before each 2022–2024 final separation (inference only, GPU 3; `revert-fp16-events/`). The truth row below is the long-series event. Inside a 60-day window, truth can only confirm a separation by lead 29, so the 30- and 40-day runs leave truth censored but still date the model's break-off. Errors are the model's final separation minus the truth date; "none" means no model separation in the window.

| truth separation (ring area) | start 10 d before | 20 d | 30 d | 40 d |
|---|---|---|---|---|
| 2022-09-10 (76,000 km²) | −2 d | −8 d | −28 d | −30 d |
| 2023-04-03 (15,000 km²) | 0 d | −17 d | −18 d | −23 d |
| 2023-09-29 (84,000 km²) | 0 d | +26 d (censored) | −23 d (12,000 km² piece) | none |
| 2024-02-28 (57,000 km²) | none | none | none | −13 d |

- **From 10 days out the model dates 3 of 4 separations within 2 days.** Its ring areas (67,000-75,000 km²) and neck depths at separation (34-46 cm) match truth's.
- **From 20-40 days out the timing is lost.** Seven of the 12 runs separate a ring 8-30 days early, usually within the run's first 2-17 days. One is 26 days late, and three make no separation at all. For 2022-09-10 the error tracks the start offset: the model sheds the ring a fixed few days into the run, not at the event.
- The early rings are of realistic size and neck depth (28-38 cm), so the model makes plausible rings at the wrong time. This points to the Loop Current neck pinching off too soon in free runs. That would fit the cyclone deficit above, since cyclones flanking the neck help set when a ring separates; this link is a hypothesis, not tested.
- The census ratios over these 16 runs repeat the 13-run result: cyclone count 0.60 of truth at day 21 and 0.30 at day 45; anticyclones 0.89 to day 45.

## Truth alone, 2022-01 to 2024-08

With the reference rule there are 4 final separations: 2022-09-10, 2023-04-03, 2023-09-29 and 2024-02-28. One more, on 2024-08-15, is censored by the record's end. That is about 1.5 a year, in line with the reference's 7 in five years of DUACS. The first, superseded version of this module found 14, because it counted every detachment that decayed and had no LCE-region filter.

## Unverified

- The census floor (5.5 cm) and the touch floor (4.324 cm) are DUACS calibrations. They mark what altimetry could resolve, not an error of the reanalysis truth.
- The 13-run starts came from the superseded event list. Four of them happen to fall 20 days before the real truth separations, and the 16-run set above is centred on all four. Only 4 out-of-sample events exist, so the timing result is 16 runs on 4 events. In-sample events (2001–2021 reanalysis) would add events but test a model that saw them in training.
- This is one checkpoint, one seed and 13 starts: a demonstration of the score, not a verdict on revert-fp16.
- This GPU inference ran without a question card. It evaluated an existing checkpoint and trained nothing.
- The optional temperature front score is not built.
