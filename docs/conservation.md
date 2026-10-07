# Domain-mean drift of the emulators and conservation loss terms

This note explains the upward SSH drift of the reanalysis emulator and of track E's E2, defines the conservation diagnostics in `hycom_emulator.conservation`, and designs loss terms for a later card. Nothing here was trained. Every number comes from the interim checkpoint of `emu-rea-s2b` (stage-2 best, `ckpt_s2_best.ckpt`; stage-1 best where noted) and from the stride-2 pack, measured on skynet GPU 1 on 2026-10-07. The probe folder is `/scratch/jmiranda/hycom-emulator-runs/conservation-probe/`. `scripts/ssh_drift_analysis.py` prints every table below from its outputs (`analysis.txt` there).

All forecasts are free forecasts given the true daily wind and the true boundary band. "Gulf" is the static Gulf mask outside the band. "Interior" is every ocean point outside the band, as `evaluate_rea --region interior`. Area weights are cos^2(lat). Depth weights are the depth interval of each level, on points real at that level.

## Diagnostics

`python -m hycom_emulator.conservation forecast <nlam.yaml> <ckpt> <prefix> [--split test] [--limit n]` runs the model and writes `<prefix>.json` and `<prefix>.npz`. Each quantity is a linear functional of the state, so its error is the functional of the error.

| Quantity | Definition | Units |
|---|---|---|
| `ssh_mean`, `ubaro_mean`, `vbaro_mean` | area mean | m, m/s |
| `temp_mean`, `salin_mean`, `u_mean`, `v_mean` | volume mean to 2000 m | degC, psu, m/s |
| `heat_content` | area mean of rho0 cp times the integral of T dz (rho0 1025 kg/m3, cp 3990 J/(kg K)) | J/m2 |
| `salt_content` | area mean of rho0 times the integral of S/1000 dz | kg/m2 |

The JSON gives, per region, quantity and lead, the mean and standard deviation of the error over forecasts, the share of forecasts with a positive error, and the mean one-day change of truth and model. It also splits the SSH error into its area-mean offset and the RMSE of the rest (the pattern), as the E2 report did. The npz keeps every forecast's values, the mean SSH error map, the pointwise column heat and salt content error, and the per-channel bias over the loss domain. `conservation series <nlam.yaml> <prefix>` computes the same quantities on every truth row of the pack, and the natural size of their one-day changes over the train years.

A heat-content error that grows by X J/m2 per day is the error a surface flux of X / 86400 W/m2 would make. That conversion is used below.

## Measured drift (2024 test, 238 forecasts, s2 checkpoint)

| Gulf, mean error at lead 1 / 2 / 3 / 4 days | 1 | 2 | 3 | 4 | Forecasts with error > 0 at lead 1 |
|---|---|---|---|---|---|
| SSH mean (cm) | +0.58 | +1.14 | +1.65 | +2.12 | 69% |
| Volume-mean T (mK) | +5.1 | +10.8 | +16.1 | +20.7 | 79% |
| Heat content (10^7 J/m2) | +2.3 | +4.8 | +7.1 | +9.1 | 79% |
| Volume-mean S (10^-3 psu) | 0.0 | -0.05 | -0.17 | -0.34 | 64% |
| Salt content (kg/m2) | 0.00 | -0.06 | -0.19 | -0.37 | 64% |
| Volume-mean v (cm/s) | -0.20 | -0.49 | -0.83 | -1.17 | 1% |
| Volume-mean u (cm/s) | -0.07 | -0.17 | -0.28 | -0.39 | 8% |
| ubaro mean (cm/s) | +0.16 | +0.21 | +0.22 | +0.22 | 86% |
| vbaro mean (cm/s) | -0.24 | -0.51 | -0.76 | -1.00 | 7% |

The heat-content error at lead 1 is the error of a 263 W/m2 surface flux over the Gulf. The truth's own mean one-day change on these days is +0.02 cm of SSH and +1.2 mK of T. The model's is +0.60 cm and +6.4 mK.

The same diagnostics on other splits (150 forecasts spread over each split):

| Gulf, lead-1 mean error | train 2001-2021 | val 2022-2023 | test 2024 |
|---|---|---|---|
| SSH mean (cm) | +0.13 | +0.28 | +0.58 |
| Volume-mean T (mK) | +0.7 | +1.6 | +5.1 |
| Volume-mean v (cm/s) | -0.17 | -0.18 | -0.20 |
| vbaro mean (cm/s) | -0.21 | -0.20 | -0.24 |
| ubaro mean (cm/s) | +0.13 | +0.17 | +0.16 |

Two kinds of drift show up. The velocity means drift by the same amount on every split, train included, and on nearly every forecast. That is a bias the training did not remove. SSH and T drift little on train and grow out of sample.

## Root cause of the SSH drift

### What it is not

- **Not the normalization or the reconstruction.** neural-lam rebuilds the next state as `X_t + (net * diff_std + diff_mean)` (`GraphLAMBase.forward` in the skynet venv), so it does add `state_diff_mean`. For SSH that is 7.8e-6 m per day in `meta.zarr`, or 0.0008 cm/day, 700 times smaller than the drift. The train data's Gulf-mean SSH change is +0.0001 cm/day. Its monthly means range from -0.22 (December) to +0.16 (April) cm/day, against a day-to-day standard deviation of 1.3 cm/day.
- **Not the boundary band.** The mean SSH error at lead 1 is +0.55, +0.52, +0.51 and +0.60 cm at 0-5, 5-15, 15-40 and over 40 cells from the band, and positive at 99% of Gulf points. Mass entering through the band would make the error largest near it. The error is a uniform offset.
- **Not the SSH level.** The lead-1 error does not correlate with the initial Gulf-mean SSH anomaly against the train climatology of its month (r = 0.02 on test).
- **Not a pull toward the train years' SSH-to-density relation.** Gulf-mean SSH is 89% explained by a linear fit on Gulf-mean T and S over the train years. In 2021-2024 SSH sits 2-6 cm below that fit, and the source experiment changes come with one-day SSH jumps (+7.4 cm on 2021-01-01, -4.5 cm on 2024-01-02, -9.8 cm on 2024-02-02, -5.2 cm on 2024-04-02). But model and truth relax the residual at the same rate (one-day change = -0.092 and -0.089 times the residual, 538 forecasts pooled), and the error does not correlate with it (r = -0.01).
- **Not daily means as such.** E2 shows the same symptom on 00Z snapshots.

### What it is

**The domain-mean SSH change is one weakly constrained degree of freedom, and the model fills it with a bias that grows out of sample.**

1. The model barely predicts the truth's Gulf-mean SSH change. Its lead-1 error correlates at -0.92 with the true change (-0.87 on train). On each forecast the Gulf-mean error has a standard deviation of 1.41 cm at lead 1, about the size of the natural day-to-day change (1.31 cm). The drift (+0.58 cm) is the mean of that spread.
2. The loss hardly sees it. `wmse` averages the squared error over points, then over the 91 channels. Splitting the SSH channel into offset and pattern, the per-forecast Gulf-mean offset is 45% of SSH's squared error at lead 1 on train, but SSH is 1/91 of the loss. The offset is about 0.6% of the lead-1 loss (0.28). The drift part of it is 0.007% on train (+0.13 cm) and 0.14% at the test drift (+0.58 cm). A uniform v offset of 0.17 cm/s, against change stds of 0.9 to 11 cm/s per level, costs 7e-4 in `wmse`, 0.3% of the loss.
3. So the mean tendency is set by whatever the network happens to learn, and it moves from checkpoint to checkpoint. From the stage-1 to the stage-2 best checkpoint the Gulf-mean v drift flips from +0.09 to -0.20 cm/s per day and the vbaro drift from +0.29 to -0.24 cm/s per day. The SSH drift is +0.43 then +0.58 cm/day.
4. It grows out of sample, and most in experiment 038. Lead-1 SSH error by source experiment: 020 +0.14 cm (117 train forecasts), 031 -0.08 (26), 035 +0.28 (150 val), 037 +0.27 (91 test), 038 +0.76 (147 test, April-August 2024). T follows: 038 +9.7 mK per day, every other experiment within +-2.3. The same calendar months in the train years give +0.22 cm and +1.9 mK. Something in the 038 states moves the free mean to a large value. These diagnostics cannot say what.

So there is no normalization or reconstruction bug to fix. The cause is in the objective. The loss gives the domain means almost no weight, and nothing ties the model's mean SSH change to the transports and fluxes that set it.

### What the mean costs in skill

The SSH offset is a large share of the SSH error, larger than the drift alone. A uniform shift changes only the offset, so the Gulf SSH RMSE of a forecast with a corrected mean can be computed exactly from the diagnostics:

| Gulf SSH RMSE (cm), lead 1 / 2 / 3 / 4 | test | val | train |
|---|---|---|---|
| model | 2.16 / 3.16 / 4.10 / 4.95 | 1.70 / 2.67 / 3.48 / 4.19 | 1.84 / 2.76 / 3.53 / 4.15 |
| perfect mean (offset removed) | 1.53 / 2.20 / 2.82 / 3.37 | 1.27 / 1.91 / 2.53 / 3.05 | 1.37 / 2.06 / 2.65 / 3.17 |
| Gulf mean held at its initial value | 2.18 / 3.12 / 3.97 / 4.70 | 1.80 / 2.84 / 3.68 / 4.41 | 1.95 / 2.97 / 3.81 / 4.45 |
| initial value plus the train change of the month | 2.19 / 3.12 / 3.97 / 4.70 | 1.78 / 2.83 / 3.65 / 4.38 | 1.95 / 2.95 / 3.77 / 4.40 |
| model minus the val offset per lead | 2.11 / 3.00 / 3.87 / 4.64 | | |

At lead 4 the offset is 54% of the test SSH MSE and 42% on train. On train and val the model's mean change has skill, and holding the mean fixed costs 5-7%. On test it has none past lead 1, and holding the mean gains 1-5%. Subtracting the val offset gains 3-6% on test, but it calibrates on 035 and is applied to 037 and 038, so it would not transfer to another system such as GrASE.

## Loss terms for a later card

Notation: e = prediction - truth in physical units at step k, I the loss's interior points (neural-lam's `interior_mask`), a = cos^2(lat), F = 91 channels. neural-lam's training loss is `wmse` = (1/F) sum over channels of mean over I of (e / sigma_f)^2, with sigma_f the channel's one-day change std, averaged over the K rollout steps. Its lead-1 value is 0.28 at the s2 checkpoint (val, stage 2), and the 4-step mean about 0.9.

### A. Domain-mean penalty per quantity and step

L_A = lambda_A (1/K) sum_k sum_q ( Q_q(e_k) / s_q )^2

Q_q is a `conservation.functionals` quantity on the interior: SSH, ubaro and vbaro area means, and T, S, u and v volume means. s_q is the train standard deviation of the one-day change of that quantity (from `conservation series`, interior): SSH 1.05 cm, ubaro 0.31 cm/s, vbaro 0.46 cm/s, T 15 mK, S 1.3e-3 psu, u 0.18 cm/s, v 0.096 cm/s.

- **Size at the measured errors.** On 150 train forecasts the sum over quantities is 6.5 at lead 1, 28 at lead 2, 68 at lead 3 and 124 at lead 4: 57 averaged over a 4-day rollout. The volume-mean v is 45% of it at lead 1 and 63% at lead 4. The 4-day `wmse` averages about 0.9 at this checkpoint (val 0.28 at lead 1, 0.69 at lead 2, 1.51 at lead 4; lead 3 is not logged). lambda_A = 0.0016 makes L_A a tenth of it.
- **Against what `wmse` already does.** `wmse` already charges a uniform offset b by b^2 times (1/F) sum over channels of (real-point share / sigma_f^2). That is 11.8 per m^2 for SSH, 252 per (m/s)^2 for v, 254 for u, 411 per degC^2 for T and 5386 per psu^2 for S. A adds lambda_A / s_q^2. At lambda_A = 0.0016 that is 1.2 times the existing weight for SSH, 2 times for u and 7 times for v, but only 0.02 times for T and 0.2 times for S. The deep levels' small change stds already make a uniform T or S offset expensive in `wmse`, and T still drifts on 038, so weight alone may not stop an out-of-sample drift. A second arm at lambda_A = 0.006 gives SSH 4.6 times and v 26 times.
- **Cost.** One weighted sum over (batch, interior, channels) per step with fixed weights, about the cost of `wmse` itself. No extra memory beyond one temporary of the state's size.
- **Per-channel weighting.** It is added to `wmse`, not folded into it, so the per-channel `1/(diff_std/state_std)^2` weights stay. It works on destandardized errors (`e_std * state_std`), so its scale does not depend on the channel normalization.
- **Boundary band.** Sum over I only, like `wmse`. The band holds the truth, so its error is zero and adds nothing. The term does not constrain what crosses the band. It does fix the interior's mean, which the band does not.
- **Below-floor points.** The volume means use `level_ocean` weights, so fill points do not count. `wmse` counts them, and they carry no error, so the per-channel bias it sees is the real-point bias times the real share of the interior (0.62 at 500 m, 0.38 at 2000 m; measured on train). L_A gives the real-point bias its full weight.
- **Prediction.** It should remove the train-split velocity drifts (in sample, every forecast the same sign) almost entirely. Whether it shrinks the 038 SSH drift is open. A model that must get the Gulf-mean change right on every train forecast may also learn what sets it.

### B. Column heat and salt content per point

L_B = lambda_B (1/K) sum_k mean over I of a ( (H(e_T) / s_H)^2 + (S(e_S) / s_S)^2 ) / mean(a)

H(e_T) = rho0 cp sum over real levels of e_T dz, and S(e_S) the same with rho0 / 1000. s_H and s_S are the area RMS one-day change of column heat and salt content over train days (from `conservation series`): 4.2e8 J/m2 and 12.7 kg/m2 over the interior.

- **Size at the measured errors.** The RMS column error over the RMS one-day change is 0.63, 1.16, 1.63 and 2.02 for heat and 0.62, 1.04, 1.39 and 1.67 for salt at leads 1-4 (train, interior). The squared sum averages 3.7 over a 4-day rollout. lambda_B = 0.025 makes L_B a tenth of the 4-day `wmse`. The ratios are almost the same on val and test (heat 0.58-0.65 at lead 1), so the pointwise column error is not where the test drift shows; the drift is in the area mean.
- **What it adds.** `wmse` treats the 22 levels as independent, so a small error of one sign through the whole column costs little. Column content weighs it by depth interval and sums it, which is what a heat budget sees. Its area mean is the domain-mean heat and salt term of A, so B contains A's T and S terms and adds their pointwise part.
- **Cost, weighting, band.** Like A: two weighted sums over levels per point and step.

### C. SSH split into mean and pattern

L_C = (1/F) [ var_I(e_ssh) / sigma_ssh^2 + mu ( mean_I(e_ssh) / s_ssh )^2 ], replacing SSH's `wmse` entry mean_I(e_ssh^2) / sigma_ssh^2.

mu = 1 with s_ssh = sigma_ssh is `wmse` unchanged. With s_ssh = 1.05 cm, mu = 1 already weighs the SSH offset (3.05 / 1.05)^2 = 8.4 times more than now. It is A restricted to SSH, with the existing offset part taken out. It costs nothing. It does not reach the velocity drifts.

### D. Volume budget consistency (not recommended yet)

The interior-mean SSH change should equal the net volume flux through the band edge, -(1/A) times the edge integral of H u.n dt, plus E - P. A term on the difference needs no truth and ties the SSH mean to the transports. It needs the edge geometry and H, and the truth itself does not close it, since altimetry and IAU increments add volume in the reanalysis. Measure the truth's budget residual before any arm.

## Recommendation

No normalization or reconstruction change removes the drift. The reconstruction is correct, and the data's mean tendency is near zero. The cause is in the objective, so a loss term is the fix, not a workaround.

**Try A first**, with lambda_A at 0.0016 and 0.006, against a control fine-tune of the same length from the same checkpoint (4-day rollouts, train splits as `emu-rea-s2b`). Reasons:

- It acts on the velocity means, which drift in sample on every forecast and which `wmse` weighs least relative to their natural variability. A prediction-time correction cannot fix them without a target.
- The SSH offset is 42-54% of the SSH MSE at lead 4. A is the term that targets the whole offset, not just its mean.
- It is cheap, and its weights come from `conservation.functionals`, so the diagnostics and the loss measure the same thing.

Arm read-out: `conservation forecast` on val and test (Gulf and interior mean errors per lead, v and vbaro drift, SSH offset and pattern), `evaluate_rea` RMSE per field against the control (no field more than 1% worse), and the isotach Hausdorff distance. If A halves the val offset RMS but leaves the 038 drift, the drift is a generalization limit and B or D are the next arms. A prediction-time projection (hold the mean, or remove a fitted offset) stays a fallback. It gained 1-6% on test but cost 5-7% on train and val.

Two data-side changes belong with the arm. Mask the training samples that straddle an experiment change (the 2017-06-01 splice, and 2021-01-01 with a one-day Gulf-mean SSH jump of 7.4 cm), because those jumps are not dynamics. And report test scores by source experiment (037, 038), since 038 carries most of the test drift.

## Track E (frozen, for the record)

E2's offset (+1.6 cm at +48 h on 05.3) has the same signature. It is a uniform offset, with a better pattern than E1's. The E2 report's guess, that the cycled twins rise faster, does not hold on their normalization statistics. The one-day srfhgt change mean is 0.102 cm/day in 05.3 and 0.070-0.102 in the twins (0.085 in the E2 stack). `diff_mean` adds 0.17 cm over 48 h, a tenth of the offset. The same diagnostics and term A apply to B00 unchanged in form: srfhgt / g as the area mean, thickness-weighted T and S as the volume means. That would test whether the B00 offset is the same free mean, but track E is frozen.

## What remains uncertain

- Why experiment 038 moves the free mean. The T-S residual and the SSH level are ruled out; the source of the shift is not found.
- Whether A generalizes to the out-of-sample drift, or only removes the in-sample velocity drifts.
- These numbers are from an interim checkpoint; stage 3 (4-day rollouts) was still training. The velocity drift changed sign between stages 1 and 2, so the final checkpoint's values will differ.
- 150 forecasts per split on train and val, spread evenly; 238 on test. No significance test.
