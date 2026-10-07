# Arm 2: a stochastic GraphLAM trained with an ensemble CRPS

Deterministic `wmse` training rewards the mean of the possible futures. In a free rollout that mean blurs fronts and eddies, and by a few days it loses their phase. Arm 2 fine-tunes the `emu-rea-s2b` GraphLAM as a stochastic model. Every step draws noise, every training sample is forecast by M members, and the loss is the CRPS of the members against the truth. A model that wins that loss keeps each member sharp and puts the uncertainty in the spread between members.

## Model

`crps_graph_lam` (`hycom_emulator.ensemble`) is GraphLAM with one change in the processor. Each step draws z ~ N(0, I_32) per member. Before each of the processor layers the mesh state h becomes h (1 + W_s z) + W_b z. W_s and W_b are linear maps without bias, one pair per layer, and they start at zero. Consequences:

- A `graph_lam` checkpoint loads strictly (missing FiLM weights are filled with zeros), and the fine-tune starts exactly at that model.
- z = 0 gives the deterministic GraphLAM of the current weights at any point in training: a control member for free.
- z is one vector for the whole domain. The processor turns it into spatially structured perturbations, as in FGN (Alet et al. 2025). Nothing perturbs single mesh nodes independently.

## Loss

Members ride on the batch axis: each sample is repeated M times and neural-lam's ARForecaster unrolls them unchanged, so the true boundary band overwrites every member. The loss is the almost-fair CRPS of AIFS-CRPS (Lang et al. 2024) per grid point and channel:

    afCRPS = mean_j |x_j - y| - (1 - (1 - alpha) / M) / (2 M (M - 1)) sum_{j != k} |x_j - x_k|

`--loss afcrps` has alpha = 0.95, `--loss fcrps` alpha = 1 (fair CRPS). With alpha = 1 and M = 2 any pair that brackets the truth scores 0 however far apart it is. The 5% plain-CRPS share removes that flat direction. Each entry is divided by neural-lam's per-channel `per_var_std` (the one-day change std, as `wmae` does) and reduced like `wmse`: mean over interior points, sum over the 91 channels, mean over the rollout. Validation and test metrics (`val_mse`, `test_mse`, `test_mae`) see the ensemble mean. `val_spread_skill` and `test_spread_skill` log the spread-skill ratio.

## Flags

Pass through `EXTRA_ARGS`:

    --model crps_graph_lam --loss afcrps --members 2 --checkpoint_steps [--init_from <graph_lam.ckpt>]

- `--members` (default 2): forecasts per sample in training, validation and neural-lam's test.
- `--init_from`: weights only, training starts at epoch 0. `--load` also takes a `graph_lam` checkpoint and keeps its usual meaning (weights and epoch, fresh optimizer).
- `--checkpoint_steps`: recompute each rollout step in the backward pass. The checkpoint restores the RNG state, so the recomputed step draws the same noise and the gradients match the plain ones.

`hycom_emulator.nlam` strips these before neural-lam parses the rest. `exclude_source_changes` (Arm 1's datastore key) filters train samples in the data module, so it applies to this model unchanged. Term A (`rea_wmse`) stays off: the ensemble module accepts only `afcrps` or `fcrps` and refuses any other loss.

## Ensemble scores

    python -m hycom_emulator.evaluate_ens <nlam.yaml> <ckpt> <out.json> --members 8 --ar-steps 21 [--region gulf] [--limit n]

Per channel, per column aggregate (T, S, u, v to 2000 m) and per lead, it reports the RMSE of the ensemble mean and of single members, the spread sqrt((M + 1)/M var) (Fortin et al. 2014), spread/skill, fair CRPS and persistence RMSE, on `evaluate_rea`'s points and weights. Spread/skill near 1 is calibrated. Below 1 the ensemble is overconfident. `ensemble_forecast` returns the members of one sample in physical units for any other diagnostic.

## GPU memory and speed

`scripts/skynet/probe_crps.py`, stride 2 (263 x 193), `emu-rea-s2b` interim checkpoint (h128, 4 processor layers), 2 members, fp32, A100 80 GB (GPU 1), 8 optimizer steps, median of the last 6. 2026-10-07.

| Rollout | Batch | Checkpointed steps | Peak allocated | Reserved | GPU s/step | Data wait s/step |
|---|---|---|---|---|---|---|
| 4 d | 2 | no | 37.2 GiB | 39.5 GiB | 0.37 | 0.06 |
| 4 d | 4 | no | 72.2 GiB | 76.6 GiB | 0.69 | 0.13 |
| 8 d | 2 | no | 74.3 GiB | 77.3 GiB | 0.73 | 0.09 |
| 8 d | 4 | no | out of memory | | | |
| 4 d | 4 | yes | 20.6 GiB | 23.7 GiB | 0.90 | 0.11 |
| 8 d | 4 | yes | 22.1 GiB | 25.3 GiB | 1.78 | 0.17 |
| 21 d | 2 | yes | 13.9 GiB | 16.1 GiB | 2.48 | 0.22 |
| 21 d | 4 | yes | 27.0 GiB | 31.4 GiB | 4.65 | 0.42 |

Without checkpointing, memory grows by about 2.3 GiB per member, sample and rollout day. Checkpointed steps cost 30% more time and make memory nearly flat in the rollout length. The data wait stays under a tenth of the step, reading the pack in place from `/scratch`.

At batch 4 the 7655-day train split is about 1900 steps: an epoch takes about 1 h with 8-day and about 2.5 h with 21-day rollouts.

## Smoke

CPU, stride-4 smoke pack (2026-10-07). A tiny `graph_lam` parent trained 1 epoch, then `train_rea.sh` with `EXTRA_ARGS="--model crps_graph_lam --loss afcrps --members 2 --init_from <parent>"` ran its three stages (1, 2, 4 days, 1 epoch each) and the test pass, then `evaluate_ens` scored 4 members. A second run loaded the parent with `--load`, used `--checkpoint_steps` and `fcrps` with 3 members. The checkpoints hold the FiLM weights, `members` and the loss. The splits overlap, so this proves the path, not skill.
