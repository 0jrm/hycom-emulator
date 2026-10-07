# Reanalysis emulator pipeline

This note designs the data and training pipeline for an autoregressive emulator of the public GOMb0.04 reanalysis daily means. It trains on 2001-2024 and is tested, and maybe fine-tuned, on GrASE 2025. The model must stay consistent, unbiased and sharp when it feeds on its own output, at +48 h and well beyond. `hycom_emulator.rea_pack` builds the training pack. Everything after the pack (losses, evaluation, launch script) is a plan here, not code.

## What the source holds

The daily means live on skynet under `/hycom/ftp/pub/BOEM/GOMb0.04/data/daily_netcdf/YYYY/gomb4_daily_YYYY_DDD_{3z,2d}.nc`. Each file is the mean of one day's hourly archives. Per the file history, `ncra` averages that day's hours 12-23, 00-11 and 12 again, with weight 0.92 on the two 12Z files, so a row is centred near 12Z (`MT` reads day + 0.48).

- 3z (about 162 MB, netCDF-4, uncompressed, chunks of 14 depths): `u`, `v` (total velocity), `w_velocity`, `water_temp`, `salinity` on 40 depths (0 to 5000 m) x 385 x 525. Fill value 1.267651e30. Each variable's `valid_range` attribute is that file's own min and max.
- 2d: `ssh`, `mixed_layer_thickness`, `u_barotropic_velocity`, `v_barotropic_velocity`, `wnd_ewd`, `wnd_nwd` (10 m wind).
- The grid is HYCOM's 525 x 385 Mercator p-grid (`topo/regional.grid.b`, `mapflg = 0`), 98W-77.04W, 18.09N-31.96N. 70.2% of points are ocean at the surface, 27.5% at 2000 m. A Mercator cell is square with a side proportional to cos(lat), so area weights go as cos^2(lat): a cell at 31.96N holds 80% of the area of one at 18.09N.
- No heat or freshwater flux exists anywhere in the public set. The hourly `data/YYYY/*_2d.nc` files hold the same six fields as the daily 2d files. The daily files' history shows the surface stresses were deleted (`ncks -x -v surface_e_stress,surface_n_stress`) and the wind was appended from the HYCOM forcing file `wndewd_124g.nc`.

`python -m hycom_emulator.rea_pack inventory` lists the days present per year from file names alone.

| Years | 3z days | Missing days | Experiment |
|---|---|---|---|
| 2001 | 350 | Jan 1-15 (record starts 2001-01-16) | 020 |
| 2002-2017 | full | none | 020, then 031 from 2017-06-02 |
| 2018 | 363 | May 1-2 (2d missing May 2 only) | 031 |
| 2019 | 363 | May 11-12 (2d missing May 12 only) | 031 |
| 2020-2023 | full | none | 031 (2020), 035 (2021-2023) |
| 2024 | 244 | none, ends Aug 31 | 035, 037, 038 (below) |

The experiment is the 3-digit prefix of the hourly archives each daily file was averaged from, read from the daily files' `history`. The global `experiment` attribute says "01.0" in every file and does not tell them apart. The hourly directories hold one prefix per year except 2017 and 2024 (the 2002 and 2019 listings did not finish; the first and last daily files of 2002 name 020, those of 2019 name 031). The changes are:

| Days | 3z source | 2d source |
|---|---|---|
| 2001-01-16 to 2017-05-31 | 020 | 020 |
| 2017-06-01 | 020, hours 19-23 from 031 | same |
| 2017-06-02 to 2020-12-31 | 031 | 031 |
| 2021-01-01 to 2023-12-31 | 035 | 035 |
| 2024-01-01 | 035, hours 19-23 from 037 | 035 |
| 2024-01-02 to 2024-01-05 | 037 | 037 |
| 2024-01-06 to 2024-01-31 | 037 | 035 |
| 2024-02-01 | 037 | 035, hours 19-23 from 037 |
| 2024-02-02 to 2024-04-01 | 037 | 037 |
| 2024-04-02 to 2024-08-31 | 038 | 038 |

In January 2024 the 3z and 2d files of one day come from different experiments, so T, S and u, v disagree with SSH, barotropic velocity and wind on 27 days of the test year. Scores over 2024-01-01 to 2024-02-01 should be read with that in mind, or start the test on 2024-02-02. The two spliced days (2017-06-01, 2024-01-01) mix 5 hours of the next experiment into the mean.

Reading is the slow part. The source is NFSv3 from `u2:/pools/f2/BOEM_GOMb0.04` over a 10 Gb/s link, the shared HYCOM ftp store, and its throughput grows with parallel readers. `dd` of daily 3z files gave 8 MB/s with 1 stream, 13 MB/s with 4, 54 MB/s with 16 and 163 MB/s with 32 (2026-10-06). Fancy-indexing the depth axis is slower than a full read, because every chunk spans 14 depths: 4 variables at 22 depths took 39-54 s, the full variables 17 s. The builder reads the depth range down to the deepest selected level as one slice and never reads `w_velocity`. With 2000 m as the deepest level that touches every depth chunk of 4 of the 5 variables, about 130 MB per day and 1.1 TB for the record. The build defaults to 32 workers. Each worker holds about one day of the selected fields; the build peaked at 5.6 GB RSS in all. Stay under 64 workers: the server is shared. netCDF reads do not reach the `dd` rate. A 33-day stride-4 build with 32 workers read its rows in 77 s (2.3 s per day, about 56 MB/s; 2026-10-07), so one pass over the record takes about 5.5 h, not the 2 h that 160 MB/s would give. The 10-day smoke packs, with about 10 days in flight, took 14 s per day at stride 1 and 9-10 s at strides 2 and 4. The stride changes only what is written; every stride reads the same bytes.

## State: 22 depths down to 2000 m, plus SSH and barotropic velocity

The state holds `temp`, `salin`, `u` and `v` at 22 depths, then `ssh`, `ubaro` and `vbaro`: 91 channels. The default depths are 0, 4, 10, 20, 30, 40, 50, 70, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1250, 1500 and 2000 m. They keep 7 levels in the mixed layer (0-50 m), 9 in the thermocline where the Loop Current and its eddies carry their signal (70-500 m), and 6 below. Below 2000 m the fields change slowly and the barotropic velocity carries the full-column transport for two channels. `w_velocity` and `mixed_layer_thickness` are diagnostics of the state and stay out. `--levels` picks any other subset, or `all`.

The level count is a memory decision. One row of 91 float32 channels on the full grid is 73.6 MB, plus 4.0 MB of forcing. The record has 8629 days (2001-01-16 to 2024-08-31):

| Grid | Depths | State channels | MB per day | All days | Train rows |
|---|---|---|---|---|---|
| 0.04 deg (stride 1) | 40 | 163 | 135.8 | 1172 GB | 1040 GB |
| 0.04 deg (stride 1) | 22 | 91 | 77.6 | 670 GB | 594 GB |
| 0.04 deg (stride 1) | 16 | 67 | 58.2 | 502 GB | 446 GB |
| 0.08 deg (stride 2) | 22 | 91 | 19.5 | 168 GB | 149 GB |
| 0.16 deg (stride 4) | 22 | 91 | 4.9 | 42 GB | 38 GB |

skynet has 678 GB free on `/scratch` (the same disk as `/`), a 504 GB `/dev/shm` with 446 GB free, and 1 TB of RAM of which the live E runs already use about 420 GB. No full-resolution pack of the whole record fits in RAM, and the 22-level one would fill `/scratch`. GPU memory points the same way. On the full grid an A100 holds `graph_lam` h128 at batch 4 for a 2-step rollout (69 GB), and a 3-step rollout ran out of memory. A 4-day rollout on the full grid fits only at batch 1 or 2 (about 39 GB and 69 GB by linear extrapolation, unmeasured).

So the plan trains in two resolutions:

1. Pretrain at stride 2 (0.08 deg, 263 x 193) on all of 2001-2024: 168 GB, which fits in `/dev/shm` once the E runs end, or reads in place from `/scratch` with `STAGE=0`. A 4-day rollout at batch 4 fits on one A100 by the same extrapolation.
2. Fine-tune at stride 1 on 2017-2024 (2800 days, 217 GB), the experiments closest to the GrASE system, with 2- and 4-day rollouts at batch 1-2. `graph_lam`'s MLPs are shared across nodes and edges, so the weights carry over if both graphs use the same mesh hierarchy. A smoke run must confirm that before the full fine-tune.

Below the sea floor and on land, every state channel holds one constant: its mean over real points on the start day. Those points then standardize near 0 and never change, so they add almost nothing to the loss. The first smoke used the deepest valid value above for below-floor points and 0 on land. With per-level statistics that put standardized values near 2800 below the floor (a shelf salinity copied into the 2000 m channel, whose std is 0.013), and below-floor points made 98% of the persistence `wmse` on the stride-1 smoke pack (99.9% at stride 4). Over real points alone the persistence loss was 0.90 (0.74 at stride 4). The per-level validity mask is stored as `level_ocean` (grid_index, level) in `meta.zarr`. Statistics and scores use it. The loss does not yet (see open questions).

## Forcing: wind and the calendar

The forcing per row is the day's mean `wnd_ewd` and `wnd_nwd`, then three calendar channels: `sin_doy`, `cos_doy` (phase 2 pi (doy - 0.5) / days in the year) and `insolation`, the daily-mean top-of-atmosphere insolation from latitude and day of year. Hour of day is constant for daily means and is left out. neural-lam stacks the forcing of rows t-1, t and t+1 around each target, so no channel is shifted in the pack.

Insolation alone cannot tell spring from autumn: 20 April and 22 August get the same sun, but the upper ocean lags the sun by about two months and is much warmer in August. The sine and cosine carry that phase. With 24 annual cycles in the record, day of year is no longer confounded with the date the way it was in the single Mar-Sep season of the E runs.

The missing forcing is heat and freshwater. The pack has no air temperature, humidity, radiation, precipitation or river discharge. The model must infer the surface heat flux from the state, the wind and the calendar. Expect three consequences. First, the seasonal heating and cooling become a learned climatology. Second, cold-air outbreaks and hurricane heat loss beyond what the wind implies go unforced, so SST and mixed-layer errors grow over days to weeks in those events. Third, Mississippi and Atchafalaya discharge is unforced, so shelf salinity drifts toward its mean. Wind stress is also only approximated: the daily mean of `|U| u` cannot be rebuilt from the daily mean wind. The hourly 2d files hold hourly wind, so a `|U| u` channel costs a second pass over 24 small files per day.

## Static fields

`depth` (from `topo/regional.depth.a`), `lon`, `lat`, `coriolis`, `ocean` (valid surface temperature) and `gulf`. The Gulf mask is the ocean region that holds 25N 90W after two sections are cut out: the Yucatan Channel (Cabo Catoche to Cabo San Antonio) and the Florida Straits at 81.1W. Scores are reported over the Gulf, so the Caribbean and Atlantic parts of the domain do not dilute them.

## Boundary: a 20-cell band from the open edges is given

The reanalysis is nested in GOFS through relaxation at the open south and east edges. The boundary mask marks land and every ocean point within 20 native cells of an ocean cell in the outer two rows or columns. It is not B00's mask. B00's mask, derived from the relaxation e-folding time, shares 57% of its ocean points with this band on the same grid (Jaccard index, stride-1 smoke). It reaches 19.38N at the south edge, where this band stops at 18.89N, and it covers about 5000 Atlantic points east of Florida (around 26.8N 77.8W) that touch no open edge. The public set does not hold the reanalysis' relaxation mask, so `--band` stays a guess until those zones are known. neural-lam overwrites boundary points with the truth at each step and excludes them from the loss, so the band acts as given forcing. In GrASE use the band comes from GOFS or the GrASE run itself.

## Splits

- Train: 2001-01-16 to 2021-12-31 (7655 days).
- Validation: 2022-01-01 to 2023-12-31 (730 days).
- Test: 2024-01-01 to 2024-08-31 (244 days).
- External test: GrASE 2025 (Apr-Sep), as its own pack from the GrASE daily means.

The four missing days in 2018 and 2019 are filled by linear interpolation between neighbouring days and flagged in `time_filled`. They are excluded from the statistics, and scoring must skip windows that touch them. Gaps longer than 3 days stop the build.

## Normalization

Every channel is standardized by its mean and standard deviation over train rows and its valid points: each depth uses its own `level_ocean` mask, surface and forcing channels use the surface ocean. One-day changes count only between consecutive real train rows. The change std has the same floor as B00 (0.05 x the state std). neural-lam weights each channel's loss by `1/(state_diff_std/state_std)^2`.

## Autoregressive training

The curriculum is 1, then 2, then 4 days. Each stage loads the best checkpoint of the stage before. neural-lam averages the loss over the rollout steps, which is the sum over leads divided by the rollout length, so the gradient reaches every lead through the whole unrolled chain. Validation uses a 4-step rollout in every stage (`--ar_steps_eval 4`). The checkpoint scores then compare across stages, which `train_b00.sh`'s `lowest()` relies on. The val loss of a 1-step stage is therefore already a 4-day score.

The loss starts as neural-lam's `wmse`. The reweight loss that won in track E (equal weight per field, thickness weighting) has a z-level counterpart: weight each depth by its layer thickness in the selected set. Build it once the baseline runs.

## Evaluation per lead

The reanalysis emulator has no increment channels. Every score is a free forecast from the reanalysis state at t0, given the true daily wind and the true boundary band over the whole forecast. Both are future information at t0, so each score carries the label "free forecast, true wind and boundary band". The truth itself is an assimilating reanalysis, so the t0 state is an analysis. This follows the labels of the workspace note `docs/research/2026-10-07-b00-increment-replay.md`.

Score leads 1 to 10 days from every test start, over interior Gulf points, with area (cos^2 lat) x layer-thickness weights. Score against persistence and against a day-of-year climatology of the train years, which becomes the baseline to beat beyond a few days.

- RMSE per field and depth, and for SSH.
- Bias per field: the mean error per lead, as a Gulf mean and as a map.
- Sharpness: the band-limited power ratio of prediction to truth for 10-50 km scales (T at 0 and 100 m, SSH), and the kinetic-energy deficit.
- Loop Current front: the Hausdorff distance between predicted and true 1.5 kt (0.77 m/s) surface isotachs, from `feature/isotach`.
- Conservation, as diagnostics only: the error of Gulf-mean T, S and SSH, and of column heat content (rho c_p times the integral of T over depth) and salt content.

A long-rollout check runs 365 days free from several starts (2022-01-01, 2023-01-01, 2024-01-01). It tracks Gulf-mean T, S and SSH, total kinetic energy and the maximum speed. It passes when nothing blows up and the Gulf means stay inside the train years' envelope for that day of year.

## Throughput against skynet's measured limits

The loader, not the GPU, limited B00: about 4 GB/s of copies between processes with 8 forked workers (1.2 batches/s at batch 4). One 4-day sample is 6 state rows plus forcing windows. At stride 2 that is about 0.12 GB per sample, so the copy ceiling (about 8 batches/s at batch 4) sits above any plausible GPU rate. At stride 1 a sample is about 0.49 GB, so batch 2 stays near 4 batches/s, also above the GPU rate. The GPU sets the pace in both stages.

## Build a pack

```
PYTHONPATH=<worktree>/src:/conda/jmiranda/pylib-netcdf4 \
	/conda/jmiranda/venvs/hycom-emulator/bin/python -m hycom_emulator.rea_pack build <out> \
	--start 2001-01-16 --end 2024-08-31 --train-end 2021-12-31 --stride 2
```

The skynet venv has no netCDF4. It was installed apart from the venv, with `uv pip install --target /conda/jmiranda/pylib-netcdf4 --no-deps netCDF4 cftime certifi`, so the live runs' venv stays untouched. The build is resumable: rerun the same command after a crash and only the unwritten rows are read. A rerun with a different plan into the same folder stops with an error.

## Open questions for the user

- Storage: the plan needs 168 GB (stride 2) plus 217 GB (stride 1, 2017-2024) on `/scratch`, which has 678 GB free. A full-resolution pack of the whole record (670 GB) needs another disk, float16 storage, or fewer depths.
- Heat and freshwater forcing: is it worth fetching the reanalysis' HYCOM forcing files (Alex) or ERA5 daily fluxes and river discharge? Without them the emulator cannot follow weather-driven heat loss.
- Below-floor points: they hold a constant, but `wmse` still counts them, which dilutes the loss by the real fraction (about 0.71 of interior points). A loss masked by `level_ocean` would remove them. It needs a custom loss in `physics.py`.
- GrASE 2025 daily means: the Blueback runs hold 12Z daily means. They must be made on the same 00-23Z window and the same 40 depths before GrASE can serve as a test.
