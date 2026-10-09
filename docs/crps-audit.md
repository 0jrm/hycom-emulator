# Audit of the crps arm (2026-10-09)

The question: "something is very wrong in the crps implementation, it's not learning at all." Short answer: the code
computes what the design says (loss, members, noise, gradients, flag routing, compile, checkpointing all check out),
and the design cannot learn skill from scratch. Trained from random weights, the two-member afCRPS puts the error into
spread within a few hundred steps and the deterministic skill stops improving, while the wmse control on the same
start keeps improving. Fine-tuned from the deterministic checkpoint at lr 1e-4 it keeps the skill and halves the CRPS
gap; at lr 1e-3 it destroys the skill within 400 steps. Two defects around the design did make the runs worse than
they needed to be and are fixed on this branch: validation drew fresh noise (1% swings in the number that selects
checkpoints and stops stages) and the noise multiplied the mesh state before every processor layer (activations
200-300x the control's, fp16 overflows). Line numbers are of `feat/rea-scratch-round` at fd3f082 unless the file is
named with a branch.

## What the probes measured

All probes run on skynet GPU 2 or 3 from `/scratch/jmiranda/hycom-emulator-runs/crps-audit/` with the frozen clone
`/scratch/jmiranda/hycom-emulator-code/787d8d5d1cee1cfe2fd3a6384c98a6817a5c6cc4` (the code of the runs) or
`.../9f2141def6947ea8b3a561bee21b3430e0642c70` (this branch). Scripts and logs are in `scripts/skynet/crps_audit/` on
this branch; the logs are also on skynet next to the scripts.

`probe_learn.py`: 600 AdamW steps (betas 0.9/0.95, 1-day rollouts, batch 4, fp32, seed 0, h128, 4 processor layers,
the multiscale graph) of the control (`graph_lam`, wmse) and the ensemble (`crps_graph_lam`, afcrps, 2 members) from
the same random weights or from `emu-rea-s2b/scored.ckpt`, scored every 100 steps on 24 fixed val samples (2022-2023)
with the same yardsticks for both: wmse of the forecast (control) or of the z = 0 member, of the 8-member mean and
of one member; afCRPS of 2 and 8 members; spread/skill of 8 members. wmse and afCRPS are summed over the 91 channels
as in training.

From scratch (all arms start at wmse 0.799):

| step | control wmse | crps z=0 wmse | crps 8-mean wmse | crps member wmse | spread/skill | afCRPS (8) | FiLM \|W\| |
|---|---|---|---|---|---|---|---|
| 100 | 0.629 | 0.708 | 0.700 | 0.775 | 0.38 | 3.41 | 1.35 |
| 200 | 0.593 | 0.711 | 0.709 | 0.984 | 0.60 | 3.21 | 1.69 |
| 300 | 0.579 | 0.711 | 0.872 | 1.566 | 0.97 | 3.37 | 2.01 |
| 400 | 0.580 | 0.711 | 0.760 | 1.408 | 0.92 | 3.23 | 2.20 |
| 500 | 0.561 | 0.807 | 0.771 | 1.367 | 0.96 | 3.54 | 2.35 |
| 600 | 0.553 | 0.655 | 0.715 | 1.330 | 1.01 | 3.09 | 2.51 |

Variants of the ensemble arm from scratch (z = 0 wmse / 8-mean wmse / member wmse at step 600): afcrps with the
noise off (members identical, so the loss is wmae) 0.594 / 0.594 / 0.594, close to the control; 4 members
0.630 / 0.675 / 1.452; FiLM on the layer update (the fix below) 0.682 / 0.782 / 1.128; afcrps plus 1.0 x wmse of the
z = 0 forecast 0.636 / 0.729 / 1.407; this branch's code 0.666 / 0.693 / 0.943. Every noisy variant stalls between
0.63 and 0.75 while the control reaches 0.553 and the noise-free afcrps 0.594. Logs: `learn_control.log`, `learn_crps.log`, `learn_crps_z0.log`, `learn_crps_m4.log`,
`learn_crps_update.log`, `learn_crps_meanw1.log`, `fixed_scratch.log` (this branch's code).

Fine-tune from `emu-rea-s2b/scored.ckpt` (wmse 0.301 on the 24 samples), 2 members:

| step | lr 1e-4: z=0 / 8-mean / member wmse | spread/skill | afCRPS (8) | lr 1e-3: z=0 / 8-mean / member wmse | spread/skill | afCRPS (8) |
|---|---|---|---|---|---|---|
| 0 | 0.301 / 0.301 / 0.301 | 0.00 | 2.73 | 0.301 / 0.301 / 0.301 | 0.00 | 2.73 |
| 100 | 0.288 / 0.295 / 0.323 | 0.40 | 2.18 | 0.345 / 0.363 / 0.547 | 0.74 | 2.21 |
| 300 | 0.290 / 0.314 / 0.397 | 0.83 | 2.02 | 0.410 / 0.423 / 0.810 | 1.17 | 2.28 |
| 600 | 0.289 / 0.331 / 0.440 | 1.12 | 2.03 | 0.501 / 0.503 / 1.033 | 1.26 | 2.46 |

This branch's code at lr 1e-4 (`fixed_ft_lr1e-4.log`): 0.290 / 0.313 / 0.455 at step 600, spread/skill 0.96, afCRPS
1.95. Logs: `ft_crps_lr1e-4.log`, `ft_crps_lr1e-3.log`, `ft_crps_update_lr1e-4.log`.

`probe_valnoise.py`: the 4-day, 2-member val_mean_loss of `emu-rea-scratch-crps-d1/min_val_loss.ckpt` over the full
val split (724 samples) with three seeds: 4.813, 4.765, 4.834 (`valnoise_d1.log`).

`probe_film.py`: FiLM weight norms of the trained checkpoints and the spread of the `emu-rea-scratch-crps-d8` model on
one val sample, eager, compiled and under fp16 autocast (`probe_film.log`).

`probe_nan.py`: forward hooks on the predictor of `emu-rea-scratch-crps-d8` and of `emu-rea-scratch-control-d8` over
8-day training batches under fp16 autocast, reporting the max |activation| after each processor layer
(`nan_d8.log`, `nan_control_d8.log`).

`mlflow_curves.py` (CPU): the train_loss_step, train_loss_epoch, val_mean_loss, val_loss_unroll1/4, val_spread_skill
and lr series of every run in an mlflow.db, with the count of non-finite train losses.

## Findings

### F1. The design cannot learn skill from scratch with 2 members (explains "not learning"; no code defect)

Evidence: the table above. The control's wmse falls 0.799 -> 0.553 in 600 steps. The ensemble arm's z = 0 forecast is
flat at 0.71 from step 100 to 400, its 8-member mean sits at 0.70-0.87, and a single member gets worse than the
untrained model (0.775 -> 1.57 at step 300) while spread/skill climbs to 1.0 by step 300 and the FiLM norms grow
linearly (1.35 -> 2.51). The objective itself is not the obstacle: with the noise off the same loss (then wmae) reaches
0.594, within 7% of the control. Four members, the bounded FiLM and a wmse term on the z = 0 forecast all stall the
same way. The mlflow curves of the from-scratch run show the same thing at full scale: the 1-day stage's
val_loss_unroll1 went 2.87, 2.66, 2.61, 2.75, 2.66 over its five epochs (`emu-rea-scratch-crps/mlflow.db`) while the
control's went 0.49 -> 0.37 over its first five and 0.27 over thirty; the 8-day stage's train loss rose through the
stage (chunk medians 6.1 -> 6.9) and its val_mean_loss rose 6.79 -> 6.96.

Mechanism. With 2 members the almost-fair CRPS per grid point is the distance from the truth to the interval between
the members plus 1.25% of the interval's width (`ensemble.py:81-96`, correct against Lang et al.). Its gradient pulls
only the member nearest the truth, with weight 0.99, and the other with 0.01; a bracketed point gives both members
0.01. Widening the spread lowers the loss at every point the members do not yet bracket, so the FiLM weights grow
until spread/skill is about 1 (300 steps here, 1-2 epochs in the runs). From then on half the points are bracketed and
contribute nothing to the skill gradient, and at the rest the pull on the mean is the sign of a noisy member error:
for a Gaussian ensemble of spread s and mean error e the expected pull is 2 Phi(e/s) - 1, about 0.8 e/s, where wmse
gives 2 e with no noise. Because z is one 32-vector per member, the member perturbation is coherent over the whole
domain, so this noise does not average out over the 50 759 grid points. AIFS-CRPS and NeSPReSO both avoid this by
training the mean first (see the comparison below).

Severity: high; this is the whole gap between the arm and its control.

### F2. Validation drew fresh noise, so checkpoint selection and the plateau rule reacted to draws (fixed)

`ensemble.py:115-122` scored validation with 2 freshly sampled members; `train_model.py:547-553` selects
`min_val_loss.ckpt` on `val_mean_loss` and `rea_train.py:175-212` (Plateau) decays the learning rate after 2 and stops
the stage after 3 validations that do not beat the best by 0.5%. Evidence: three seeds on the full val split give
4.813, 4.765, 4.834 for one checkpoint, a 1.4% range, three times the plateau threshold. The 1-day stage of the
from-scratch run stopped after 5 epochs on the sequence 5.21, 4.72, 4.73, 4.85, 4.76 (the control ran its 30), and
every later stage stopped after 4 epochs on 2-3% swings. The fine-tune's val_mean_loss moved 4.42 -> 4.28 with
swings of 2.5% between epochs (`emu-rea-arms-crps/mlflow.db`), so the selected epoch is partly a draw.

Fix: `EnsembleForecasterModule.on_validation_start/end` and the test hooks seed the RNG with `EVAL_SEED` for the pass
and restore the training state after (`ensemble.py` on this branch, commit 753bcc8;
`tests/test_ensemble.py::test_validation_draws_the_same_noise_every_time_and_leaves_the_training_stream_alone`).
Severity: medium. It does not explain the learning curve, it explains why stages stopped early and which epoch won.

### F3. The noise multiplied the mesh state before every processor layer (fixed)

`ensemble.py:63-71`: `mesh_rep = mesh_rep * (1 + scale) + shift` before each of the four InteractionNets, whose own
updates are layer-normed but whose residual stream is not. The trained scale rows reach norm 1.0 (`probe_film.log`:
rms scale 0.35 after the 1-day stage, 0.65-0.70 from the 4-day stage on; the fine-tune stays at 0.07), so each layer
multiplies the stream by 1 + N(0, 0.5-1) and the four multiply. Evidence: max |mesh_rep| after processor layers 1-4
is 37 / 133 / 441 / 1427 under fp16 autocast and 33 / 107 / 426 / 2282 in fp32 for `emu-rea-scratch-crps-d8`,
against 5.0 / 5.4 / 6.3 / 7.4 for `emu-rea-scratch-control-d8` (`nan_d8.log`, `nan_control_d8.log`). Under
16-mixed the run logged non-finite train losses on 6 of 7641 steps in the 4-day stage, 45 of 7629 in the 8-day stage
and 18 of 914 in the 16-day stage; the control logged none, nor did the fp32 fine-tune (`mlflow_curves.py`). Lightning's
GradScaler skips those steps and halves the loss scale each time.

Fix: the noise modulates each layer's layer-normed update, `mesh_rep + (net(mesh_rep) - mesh_rep) * (1 + scale) +
shift` (commit 9f2141d; `tests/test_ensemble.py::test_noise_modulates_the_layer_update_not_the_mesh_state`). z = 0 still
reproduces GraphLAM, the state-dict keys do not change, and the fine-tune probe behaves the same (0.290 / 0.313 / 0.455
and afCRPS 1.95 at step 600 against 0.289 / 0.331 / 0.440 and 2.03). Severity: medium. It does not change the learning curve from scratch
(`learn_crps_update.log`), it removes the crps-only overflow and the exponential growth.

### F4. The learning rate of the CRPS stage decides whether skill survives (recommendation)

Fine-tuned from s2b at lr 1e-3 the z = 0 forecast's wmse goes 0.301 -> 0.433 in 400 steps and the member's to 1.5;
at lr 1e-4 the z = 0 forecast stays at 0.29 for 600 steps, the 8-member mean drifts 0.301 -> 0.331 and afCRPS falls
2.73 -> 2.03. The card's fine-tune did use lr 1e-4 (`emu-rea-arms-crps` params) and its 8-member mean ended 2-5%
behind the control at 21 days, which matches this drift. The from-scratch script restarts every stage at lr 1e-3
(`train_rea.sh:61-67`), which for this arm resets the damage each stage.

### F5. The z = 0 member is the arm's best deterministic forecast and nobody scores it

In every probe the z = 0 forecast beats the 8-member mean (0.289 against 0.331 after the lr 1e-4 fine-tune). The
members are not symmetric around z = 0 through the nonlinear processor, so averaging them adds error.
`evaluate_ens.py` reports the mean, members, spread and CRPS but not the z = 0 forecast, and the card's "ensemble mean
2-5% worse than control" never saw the arm's cheapest product.

### F6. Checked and found correct

- afCRPS (`ensemble.py:81-96`): the sorted-pair identity, the almost-fair coefficient, division by `per_var_std` (as
  `wmae`, `metrics.py:185-236`), reduction mean over interior points and sum over channels
  (`metrics.py:37-84`), `.mean(0)` over the batch to `(T,)` and neural-lam's mean over steps
  (`module.py:411-412`). `tests/test_ensemble.py` checks it against the direct double sum in float64.
- Members: `repeat_interleave` on the batch axis, every member gets its own z per step
  (`torch.randn(batch_size, 1, 32)`, `ensemble.py:65`), the boundary band overwrites all of them, two forward calls
  differ, z = 0 gives the deterministic model (`probe_film.log`: members differ by up to 3.5 standardized units at
  day 1, by 7e-3 at z = 0, which is the GPU scatter nondeterminism).
- Gradient flow: the FiLM gradients are non-zero from the first step (`tests/test_ensemble.py::test_training_moves_the_film...`)
  and the weights move in every run (`probe_film.log`).
- torch.compile: compiled and eager draws differ between calls and between members and stay finite under fp16
  autocast (`probe_film.log`). Recomputed checkpointed steps draw the same noise: eager through
  `preserve_rng_state` (`tests/test_ensemble.py::test_checkpointed_steps_give_the_same_loss_and_gradients`), compiled
  through AOTAutograd's `functionalize_rng_ops`, which the partitioner applies when a checkpointed region contains
  random ops (`torch/_functorch/partitioners.py:1428-1432`, torch 2.12.1). The 16-day stage is the only one that used
  both and ran 914 steps; no probe of it.
- Flag routing (`nlam.py:132-134`): `--members`, `--init_from`, `--checkpoint_steps` go to the ensemble module,
  `--pushforward`, `--train_first_step`, `--compile`, `--nondeterministic`, `--plateau` to rea_train; nothing is
  dropped, and `--checkpoint_steps` is honoured by `CRPSGraphLAM.forward` (`ensemble.py:57-62`) rather than
  ReaForecaster, which is why the log says `checkpoint_steps False`.
- Batch semantics: the loss is a mean over the B samples after the members are reduced; Lightning logs with
  `batch_size=B` (`module.py:415`, `ensemble.py:121`).
- Loss scale: afCRPS summed over 91 channels is 2-7, wmse 0.3-1.5; AdamW is scale-free.

## Comparison with NeSPReSO's CRPS

NeSPReSO's A_CRPS (`ISAS20_project/NeSPReSO2_onTemplate/model/loss.py:534-566`, `evalphys/calibration.py:33-41`) is
a different object: a closed-form Gaussian CRPS on a predicted (mu, sigma) per PCA coefficient, no members and no
noise. Every output gets a dense gradient at every point (d CRPS / d mu = 2 Phi(z) - 1, d CRPS / d sigma from the
same formula), validation is deterministic, and the spread is a parameter with a bias initialised at the PC RMS
(`loss.py:798-802`). Its training protocol (`scripts/train_prob_twostage.py:1-5,113,156-159`,
`scripts/launch_matrix.py:30-33`): stage 1 trains mu with MSE and sigma frozen for 60 epochs, stage 2 unfreezes sigma
and switches to CRPS at 0.1 x the learning rate (Adam 1e-3 -> 1e-4, batch 512), early-stopping on val ENCE
(calibration) rather than val loss, patience 40. What transfers to the emulator: mean first, then CRPS at a tenth of
the learning rate, a deterministic validation, and a stopping rule on a calibration quantity (spread/skill) rather
than on the noisy score. What does not transfer: the emulator's uncertainty is a field with spatial structure, which
is why it samples members instead of predicting a sigma; the price is the sampled-gradient variance of F1, and
AIFS-CRPS pays it with 8 members after a deterministic pre-training.

## Recommendations for the next crps card

1. Start from the deterministic checkpoint (`--init_from`, or `--load` of the control's stage checkpoint); do not
   train the arm from scratch. Evidence: F1; every from-scratch variant stalls at 0.63-0.75 against 0.55.
2. Run the CRPS stage at lr 1e-4 and never restart a stage at 1e-3; `train_rea.sh` needs `LR=1e-4` for this arm, so
   the stage restarts (twice the previous end, capped at LR) stay at or below 1e-4. Evidence: F4.
3. Validate with the seeded noise of this branch and, if the stage budget allows, with more members than training
   (`--members` is one number for both today; a separate `--val_members` would be a small change). Set the plateau
   threshold to the seeded val noise, which is now zero, and give the stage a patience of 3-5 epochs instead of 2-3.
   Evidence: F2.
4. Score the z = 0 forecast in `evaluate_ens.py` and in the card's read-out, next to the mean and the members.
   Evidence: F5.
5. Members: 4 with `--checkpoint_steps` where memory allows (the 4-day batch-4 run costs 20.6 GiB checkpointed,
   `rea-arm-crps.md`). It did not rescue the from-scratch run (F1) but halves the gradient variance of the fine-tune.
6. Precision: with the bounded FiLM the crps-only overflow path is gone; if 16-mixed is kept, log the GradScaler's
   skipped steps, or use bf16-mixed, which cannot overflow at these magnitudes.
7. Pushforward: the 8-day stage with `--pushforward 4 --train_first_step` trained on steps 1 and 6-8 and its loss rose
   through the stage; with a fine-tuned start and lr 1e-4 retest it before keeping it in the arm.
8. If a from-scratch stochastic model is wanted later, the NeSPReSO-style schedule inside one run is the thing to
   build: FiLM frozen (noise off) for the first stages, unfrozen at a tenth of the learning rate. The probes say a
   wmse term alone (`learn_crps_meanw1.log`) is not enough.

## Artifacts the fixes invalidate

The FiLM weights of every crps checkpoint trained before commit 9f2141d parametrise a different function (they
multiplied the mesh state; now they multiply the layer updates). They load, but their noise is not what was trained:

- `/scratch/jmiranda/hycom-emulator-runs/emu-rea-arms-crps/runs/emu-rea-arms-crps/checkpoints/{min_val_loss,last}.ckpt`
  and everything scored from them: `emu-rea-arms-crps-test` in that mlflow.db, the card emu-rea-arms crps read-out
  (`score/crps/ens_gulf_ar21.json`, `score/crps/rollout-h15`), and the crps rows of its report.
- `/scratch/jmiranda/hycom-emulator-runs/emu-rea-scratch-crps/runs/emu-rea-scratch-crps-d{1,2,4,8,16}/checkpoints/*`
  and the card emu-rea-scratch crps curves (`scr/emu-rea-scratch-crps/{mlflow.db,train.log}`) as a measure of the
  arm; they remain valid as evidence for F1-F3.
- `docs/rea-arm-crps.md`: the model description (updated on this branch); its memory and speed table measured the
  old modulation, same cost, still valid.
- The numbers in this audit's probes that used the old modulation are labelled as such.
- Any val_mean_loss comparison across epochs of the old runs: those were draws, see F2.
