# Plan: an RCAN pyramid around the GraphLAM mesh (draft, 2026-11-04, revised the same day: one seed)

Draft for discussion. Nothing here has run. The proposal is the user's: downscale the inputs with an RCAN-style convolutional encoder, run the graph network at the coarse level, upscale with an RCAN-style decoder, with channel attention, residual-in-residual groups and skip connections. This note says what the skynet RCAN project holds, why the idea fits the emulator's open failure, what to build (dimensions, parameters, loss, schedule), what it costs on 1-2 A100s with 500 GB of host RAM, and what the 1 km reanalysis on AWS would take. The implementation plan is at the end.

Code facts are read from hycom-emulator `origin/main` at 7bd966a (PR 42, the stride-2 U-Net). The RCC checkout is on `main` at that commit; new code branches from it. Revision of 2026-11-04 (user): every run in this plan is one seed for now; seeds 2 and 3 follow only if the one-seed result is promising. The first build covers steps 1-7 of section 6; the 1 km line (step 8) waits for the stride-2 result and the user's AWS decision.

## 1. What the skynet RCAN project is

`/unity/g2/jmiranda/RCAN` (skynet NFS home) holds `RCAN.py` (317 lines, 2024-02-01), two notebooks, a 685-pair PNG dataset (`dataset/{train,val}/{low_res,high_res}`), `best_RCAN.pth` (15 MB) and two PCA pickles from the Argo work. `RCAN.py` is a reduced RCAN for 3-channel images:

- `ShallowFeatureExtraction`: one 3x3 conv, in_channels to num_features.
- `ResidualChannelAttentionBlock`: conv3x3, ReLU, conv3x3, residual. Despite the name it has no channel attention ("simplified structure").
- `ResidualGroup`: num_blocks RCABs with a group skip. `RIRModule`: num_groups groups, without the long skip conv of Zhang et al. (2018).
- `UpscaleModule`: nearest upsample then conv3x3 (not sub-pixel convolution). `ReconstructionLayer`: one conv3x3.
- Training: MSE, Adam 1e-4, `DataParallel`, on synthetic sinusoid images (`dummy_dataloader`); the run used num_features 64, 5 groups x 10 blocks, scale 4.
- `RCAN_chatGPT.ipynb` (63 cells) is the tutorial the script was reduced from; it also defines a `ChannelAttention` (global average pool, two 1x1 convs, sigmoid) and a `DetailedRCAB` that uses it, the Zhang et al. form. `RCAN.ipynb` is `RCAN.py` with its outputs.

There is nothing to port. The real ingredients of RCAN that the emulator needs are already in `src/hycom_emulator/convnet.py` (`ResBlock`: pre-norm LayerNorm, conv-SiLU-conv, RCAN's squeeze-excite channel attention with reduction 16, zero-initialised second conv so a block starts as the identity; a stride-s coarse level with a learned `down` conv and bilinear up). What the current `ConvGraphLAM` lacks against RCAN, and what this plan adds: residual groups (two levels of skips), the residual-in-residual long skip, sub-pixel upsampling (pixel shuffle) instead of bilinear, space-to-depth downsampling instead of a mean-initialised strided conv, more than one conv level, per-block activation checkpointing and tiled execution of the finest level, and level growth so a checkpoint trained at one grid spacing initialises a model at a finer one.

## 2. Why this architecture, from the evidence

The reanalysis emulator (graph_lam multiscale, hidden 128, 4 processor layers, 867 K parameters, stride-2 pack 263 x 193 = 50,759 points, 91 state channels) has one open failure: blur from day 1 (card emu-rea-phys0). In free runs its kinetic energy ratio is 0.71 at day 21 and its 100 m temperature variance at 50-100 km is 0.5 at day 7 (control of emu-rea-sharpstab, 2024 starts). GraphLAM's grid path is pointwise: each grid point is embedded by an MLP of its own features and exchanges only with the mesh, whose finest level is 81 x 81 nodes at 0.26 deg (about 28 km). Change below the mesh scale is a pointwise function of local features, so fronts come out smooth.

What has been tried on this failure:

| Card | Lever | Result |
|---|---|---|
| emu-b00-053-conv (B00) | option A: 3+3 full-resolution conv blocks, no pre-norm, fine-tune | sharpened, but SSH RMSE +2.1%, inversions +19% |
| emu-b00-053-unet-norm (B00) | option B: stride-4 conv level with pre-norm, identity start, fine-tune | KE deficit 6.3% to 0.9%, RMSE better on every field; the only clean win |
| emu-b00-053-sharp `grad` | L1 on error gradients | blurred T, SSH RMSE +29.5% |
| emu-rea-sharpstab / sharp-s12 | adjusted MSE (amse 1, 32-day rollouts) | KE +0.040, variance +0.049 (asked +0.05), RMSE +2.5% |
| emu-rea-sharpstab `stab` | static-stability penalty 0.137 | inversions down 5x at no RMSE cost (now the default) |
| emu-rea-unet2 | option B at stride 2 on the reanalysis, fine-tuned from revert-fp16 at lr 2.5e-5 | KE +0.014, variance +0.010, S and SSH RMSE +2.5%, day-1 S +6.7%: fails |
| emu-rea-forcing | atmospheric forcing channels, from scratch | no gain on blur; "next lever is the model or the target" |

Two readings of emu-rea-unet2 matter here. First, the fine-tune started 12% worse in one-step loss than its base (a stride-2 cell average replaces the pointwise read of the mesh) and a low learning rate never recovered it; the card's own conclusion is "the U-Net from scratch, not a fine-tune". Second, the U-Net was small: 3 blocks per stage, one level, mean-pool down and bilinear up. The RCAN pyramid is the from-scratch successor the card names, with more capacity where the B00 evidence says it helps (a coarse conv level that sits between the grid and the mesh) and the pieces that keep sub-cell phase information through the down and up steps.

The same structure is what a 1 km model needs for a different reason: a full-field 1 km sample (3.23 M points) cannot go through GraphLAM's grid path on one GPU, and the graph should not run on 3.23 M nodes anyway. Convolutions at 1 km that downscale by 4 to the 0.04 deg cells, where the existing mesh coupling and processor already work, are the cheap and natural fit.

## 3. The model

### 3.1 Structure

A pyramid of conv levels in index space on the pack's Mercator grid (cells are square, 4.2 km at 18N to 3.8 km at 32N at 0.04 deg), from the grid spacing down to the mesh-coupled level, around the unchanged GraphLAM mesh:

    x (B, 203, nx, ny)                           grid input: 2 states x 91, 3 forcing windows x 5, 6 static
    h0 = head(x)                                 conv3x3 -> F channels (replaces nothing: grid_embedder stays for the mesh path at the coupled level)
    e0 = RIR_enc0(h0)                            level 0, groups G0 x blocks B0, long skip
    h1 = S2D(e0)                                 space-to-depth by s1, 1x1 conv (s1^2 F -> F), initialised as the cell mean
    e1 = RIR_enc1(h1)                            level 1
    ...
    eK = RIR_encK(hK)                            mesh-coupled level: cells of 0.16 deg (B00 option B's size; every mesh node has a sender)
    cn = nodes(eK); rep = cn + encoding_grid_mlp(cn)
    mesh = g2m_gnn(cn, mesh_emb); mesh = process_step(mesh); msg = m2g_gnn(mesh, rep) - rep     GraphLAM, unchanged
    dK = RIR_decK(image(rep + msg))
    d(K-1) = RIR_dec(K-1)(e(K-1) + PS(dK))       pixel shuffle up by sK (conv3x3 F -> sK^2 F, PixelShuffle), level skip by addition
    ...
    d0 = RIR_dec0(e0 + PS(d1))
    out = output_map(nodes(d0))                  GraphLAM's output MLP F -> 91 (or 182 with --output_std)
    new_state = prev_state + out * diff_std + diff_mean   (GraphLAM's residual update; the rea clamp if any)

Every RCAB is `convnet.ResBlock` with the boundary mask as an extra input channel: pre-norm LayerNorm over channels, conv3x3 (F+1 -> F), SiLU, conv3x3 (F -> F, zero-initialised), squeeze-excite (reduction 16), residual. A residual group is B blocks plus a group skip; a RIR is G groups plus a zero-initialised conv3x3 on the long skip. With the zero-initialisations every RIR starts as the identity, S2D starts as the cell mean and the pixel shuffle starts as nearest replication, so a model at init reproduces GraphLAM's step up to the cell average at the coupled level (the 12% one-step gap measured in emu-rea-unet2, which training removes). The forcing windows and static fields enter through the head at level 0 and through `grid_embedder` at the coupled level, so neither path is starved.

Why these choices and not others:

- Pre-norm is required. Without it a spatially uniform offset passes each 3x3 conv multiplied by the sum of its taps and the stream blows up (emu-b00-053-conv `unet` arm, loss frozen at 1.1094).
- Space-to-depth and pixel shuffle keep sub-cell phase. Mean pooling discards it and bilinear up cannot invent it; that is the part of RCAN that matters for fronts, not the depth.
- Addition for level skips, not concatenation: it keeps F fixed and lets the identity start hold.
- Index-space convs are fine on this grid because Mercator cells are square; the 10% size change across the domain is below what the kernels resolve. Lat enters as a static channel.
- No attention over space, no transformer: at 50 K-3 M points per field and 16-day rollouts the memory goes to activations, and the mesh already carries the long range.

### 3.2 Configurations

Levels are named by physical cell size, so one checkpoint loads into any pack stride: a level present in the checkpoint loads its weights, a new finer level starts at identity. F = 128 everywhere, equal to GraphLAM's hidden dim, so `encoding_grid_mlp`, `output_map` and the mesh MLPs are unchanged.

| Name | Pack | Levels (cell, encoder groups x blocks, decoder groups x blocks) | Conv blocks | Conv params | Total params |
|---|---|---|---|---|---|
| `rcan_s2` | stride 2 (0.08 deg) | 0.08: 2x3, 2x3; 0.16 (coupled): 1x3, 1x3 | 18 | about 5.6 M | about 6.5 M |
| `rcan_s2_light` | stride 2 | 0.08: 1x3, 1x3; 0.16: 1x3, 1x3 | 12 | about 3.7 M | about 4.6 M |
| `rcan_s1` | stride 1 (0.04 deg) | 0.04: 1x3, 1x3 (new); then `rcan_s2`'s levels | 24 | about 7.5 M | about 8.4 M |
| `rcan_1km` | 1/100 deg | 0.01: 1x3, 1x3 (new, S2D by 4); then `rcan_s1`'s levels | 30 | about 9.4 M | about 10.3 M |

One RCAB at F = 128 is 0.30 M parameters (two 3x3 convs of 128 x 129 x 9 and 128 x 128 x 9, plus the 2 K of the squeeze-excite); groups and long skips add a 3x3 conv each. `rcan_s2_light` has the U-Net's block count with RCAN's structure and is the probe size. `rcan_s2` is the card size. Processor depth is cheap (measured); 6 processor layers instead of 4 is a reasonable second arm later, not now.

### 3.3 Memory and time arithmetic

Reference points (A100 80 GB, fp32 unless stated; from `docs/rea-compute.md` and memory `skynet-training-pipeline`):

| Measured | Value |
|---|---|
| graph_lam, stride 2, 1-day, batch 8, 4 workers | 0.306 s/step |
| graph_lam, stride 2, 16-day, batch 8, `--checkpoint_steps`, 16-mixed | 3.07-3.4 s/step, 22 GB |
| unet_norm (12 blocks: 6 at the grid, 6 at stride 2), same stage | 4.67 s/step (1.52x) |
| graph_lam, B00 full grid (202,125 points), batch 4 x 2 steps | 0.75 s/step, 64.7 GiB: about 7.5 GB and 0.094 s per sample-step |
| ConvGraphLAM stride 1 (6 full-grid blocks) with block checkpointing, same | 1.43 s/step, 69.3 GiB |
| graph_lam full recipe (1-2-4-8-16 days) per seed | 13.2-13.4 GPU-h |
| host memory per trainer, 4 loader workers | +54 GB (shared memory from the loader) |

From the unet_norm point, one grid-level RCAB costs 1.7 ms per sample-step at stride 2 (about 0.4 ms per 10,000 points); the memory-bound LayerNorm, squeeze-excite and concatenation are inside that number. One activation tensor at F = 128 is 26 MB per sample at stride 2, 103 MB at stride 1, 1.65 GB at 1 km (fp32).

| Configuration | Grid-block equivalents | s/step (16-day, batch 8, stride 2 units) | x graph_lam | Peak GPU, 16-day, `--checkpoint_steps` |
|---|---|---|---|---|
| `rcan_s2_light` | 6 + 6/4 = 7.5 | 4.7 | 1.5 | about 30 GB (batch 8) |
| `rcan_s2` | 12 + 6/4 = 13.5 | 6.0 | 1.9 | about 35 GB (batch 8) |
| `rcan_s1` (per sample-step at stride 1) | 6 + 12/4 + 6/16 = 9.4 | 0.21 s per sample-step (0.16 + checkpointing) | | about 30 GB at batch 2, 60 GB at batch 4 |
| `rcan_1km` (per sample-step) | 6 x 16 + 9.4 = 105 | 0.3-0.6 s on H100 bf16 (see section 5) | | needs block checkpointing and lazy targets (section 5) |

Peaks are one step's activations times the batch plus the rollout states, since `--checkpoint_steps` keeps one step live. These are estimates; the probe in step 4 of the implementation plan replaces them.

## 4. Loss, schedule, data

- **Loss.** `rea_wmse` with `--mean_penalty 0 --stability 0.137` (the emu-rea-forcing base recipe, which is the from-scratch control with 3 seeds). wmse weights each channel by 1/(diff_std/state_std)^2 with the 0.05 floor, averaged over the rollout steps, so every lead is trained. No gradient term (the `grad` arm). Adjusted MSE (`--amse 1`) is a fine-tune stage of the winner, not part of the architecture card, since it costs 2.5% RMSE for +0.04 KE and would confound the comparison. The isotach surrogate loss (`isotach.isotach_loss`, Karimi and Salcudean 2019, on `feature/isotach`) is a later arm at a small weight once the architecture is chosen; it tracks the mean and p95 front distances, not the Hausdorff maximum.
- **Schedule.** The revert-fp16 recipe, as `train_rea.sh` runs it: stages 1-2-4 days at constant lr 1e-3, 16-mixed, `--compile --nondeterministic`; 8 days at 1e-4 with `--plateau 2 3`, `--pushforward 4 --train_first_step`; 16 days from the 8-day end rate with `--checkpoint_steps`; the rate never rises. Epoch caps 30/15/10/8/6. Seed 1 only for now. Scored checkpoint: the 16-day stage's min_val_loss.
- **Data.** `rea_s2.pack` (2001-2024, 168 GB, read in place or staged to `/dev/shm`), splits train 2001-01-16..2021-12-31, val 2022-2023, test 2024-01-01..08-31, `EXCLUDE_SOURCE_CHANGES=1`. Judge on 2022-2023 (in regime) with 2024 beside, per emu-rea-regime.
- **Read-out.** emu-rea-forcing's: `evaluate_rea` Gulf ar 21 on val and test with the isotach fronts, `rollout_rea` 60-day free runs from the 23 val starts and the 31 phys0 test starts, `scorecard_rea`, `summarize.py` in the run folder.
- **Hypotheses for the card (same form as emu-rea-unet2; one rcan seed against the forcing base arm's three seeds).** H1: KE ratio at day 21 and 100 m T variance ratio at 50-100 km at day 7 above the control's seed mean by at least 0.05 and above the top of the control's seed range. H2: Gulf column RMSE averaged over days 8-21 within 2% of the control's seed mean on each of T, S, u, v, SSH. H3: front distance over persistence at day 21 no worse than the control's seed mean plus range. One seed cannot resolve margins under about 5% at 16-21 days (emu-rea-revert: two runs of one recipe differ 3-6% on T and SSH), so a pass on H1 by a margin inside the control's range is "promising, add seeds 2 and 3", not a verdict. Partner metrics: day-1 RMSE per field, inversion share, Gulf-mean drift, surface vorticity noise (zeta_0m_s15), corr_change.

## 5. Estimates

### 5.1 Skynet, or any host with 1-2 A100 80 GB and 500 GB of RAM

Host memory is the binding constraint on skynet (1007 GB total, foreign tomcat processes of 63-135 GB, a 650 GB stop). The 500 GB figure below is what this project may count on.

| Work | GPU | GPU-hours | Host RAM (pack + trainers) | Wall on 1 A100 | Wall on 2 A100 |
|---|---|---|---|---|---|
| Probes: smoke on a 2-month window, 50-step timing and memory probe, `rcan_s2_light` and `rcan_s2` | 1 | 1 | 168 + 60 | 1 h | 1 h |
| Card emu-rea-rcan: `rcan_s2` from scratch, seed 1, cap 30 h, read-out 1.5 h | 1 | 25-32 | 168 + 60 | 1.3 days | 1.3 days |
| Optional second arm `rcan_s2_light` (seed 1, cap 24 h) | | 20-25 | same | 1 day | runs beside the first |
| Sharp fine-tune of the winner (`--amse 1`, 32-day, 3 epochs, seed 1; emu-rea-sharpstab's 9 h cap x 1.9) | | about 15 | 168 + 95 | 0.7 day | 0.7 day |
| Card emu-rea-rcan-s1: `rcan_s1` fine-tune on `rea_s1_2017.pack` (2017-2024, 217 GB), batch 4, stages 4 (3 epochs), 8 (4), 16 (4), seed 1, cap 15 h | | 10-15 | 217 + 55 = 272 | 0.6 day | 0.6 day |
| Seeds 2 and 3 of whichever of the above is promising | | x 2 of its row | two trainers: 290 (stride 2), 327 (stride 1) | | |

Per-seed cost of `rcan_s2` is graph_lam's 13.3 h times 1.9, about 25 h, before plateau stops; the 30 h cap matches the forcing card's 27 h times the ratio. The stride-1 epoch on 2017-2021 (1826 samples) at 16 days and batch 4 is about 1.4 h (1826 x (12 x 0.21 + 4 x 0.07) s), validation over 730 starts about 10 min. The loader at stride 1 needs 1.3 GB per 16-day sample, 4 per 3.4 s step, 1.5 GB/s, under the 4 GB/s ceiling.

Two trainers fit in 500 GB at either stride. Three do not with the stride-1 pack. A full-record stride-1 pack (670 GB in float32) does not fit in 500 GB; stored as int16 fixed point of the standardised value (scale 1/2000, range +-16 std, resolution 0.0005 std, which is 1% of the smallest one-day change std at the 0.05 floor) it is 335 GB and fits with one trainer. float16 is the worse choice: near +-4 std its resolution is 0.004 std, 8% of that change std. The int16 pack is also what the 1 km line needs, so it is on the implementation list.

What the stride-1 fine-tune buys is unknown and is its own card: the 0.04 deg level is new, the rest loads from the stride-2 winner, and the question is whether sharpness at 10-25 km improves at the same RMSE. The handoff of 2026-10-26 gated the 1 km line on this (Q9), and that gate stands: a model that blurs at 25 km on the stride-2 grid gains nothing from 1 km data.

### 5.2 The 1 km reanalysis on AWS

**Data in hand.** The 1/100 deg reanalysis is on skynet at `/hycom/ftp/datasets/GOMb0.01/reanalysis/data/{2001..2024,daily_netcdf}`, files `0NN_archv.YYYY_DDD_00_{2d,3z}.nc`, 2001-01-16 to 2024-09-01, hourly plus daily means, 2101 x 1537 grid (3,229,237 points, 16 x the 0.04 deg grid), 40 depths, 2.67 GB per day. The pack at 22 depths and 91 channels is 1.175 GB per day in float32 and 0.59 GB in int16:

| Span | Days | int16 pack | float32 |
|---|---|---|---|
| 2017-2024 (experiments nearest GrASE) | 2800 | 1.65 TB | 3.3 TB |
| 2001-2024 (whole record) | 8629 | 5.1 TB | 10.1 TB |

Build the pack on skynet, where the source is: `rea_pack` reads 2.67 GB per day from the ftp NFS store at 56-160 MB/s (measured 2026-10-06/07 with 32 workers; netCDF decode limits it), so 2017-2024 is 7.5 TB read in 13-37 h and the whole record 23 TB in 40-115 h, resumable. `rea_pack` needs the 1 km file layout, its `regional.depth.a` and a check that the 2d files hold ssh, ubaro and vbaro and the 3z files the same 40 depths (unverified: the directory listing on 2026-11-04 timed out on the NFS store before the header was read). Upload to S3 with `aws s3 cp` from skynet or RCC; at 100 MB/s the 2017-2024 pack takes 5 h.

**Model.** `rcan_1km`: a new 0.01 deg level (1x3 encoder, 1x3 decoder), space-to-depth by 4 to 0.04 deg cells (2101/4 pads to 526 x 385, one cell wider than the 0.04 deg grid), then the `rcan_s1` levels and mesh loaded from the skynet checkpoint. The multiscale mesh is built once from the 0.04 deg cell centres (`coarse_graph` rebuilds g2m and m2g from cell centres with neural-lam's rules; it must take the mesh from an existing graph folder instead of running `create_graph` on 3.2 M nodes). 203 input channels at 1 km is 2.6 GB per sample in float32; the head conv reads it once.

**Per-GPU memory (bf16 autocast).** One 1 km activation tensor is 0.83 GB per sample. Six 1 km RCABs keep about five tensors each: 25 GB per sample-step without block checkpointing, 5 GB saved plus 4 GB recompute peak with it. The coarse levels and the graph add about 4 GB. So about 13 GB per sample-step with block checkpointing, and with `--checkpoint_steps` the 16-day stage peaks near one step: batch 4 fits an 80 GB H100 for the activations alone. The rollout states do not: neural-lam keeps the whole target window on the GPU in float32, 18 rows x 1.175 GB x batch 4 = 85 GB. Two changes make the full field fit, and both are code, not hardware: (a) keep the window on the host in int16 and move one target row to the GPU per step ("lazy targets", an `ARForecaster` change in `rea_train.install`), and (b) run the 0.01 deg level on tiles with a halo of 16 cells under checkpointing (exact: 6 blocks x 2 convs x 1 cell plus the head is 13 cells of receptive field), so its activations never exceed one tile. With both, batch 4 per 80 GB GPU; without (a), batch 1-2, or H200 (141 GB, p5e/p5en). The earlier reviewed plan (artifact section 16, 2026-10-13) reached the same shape: tiles with halos and a coarse-model context.

**Time.** The 1 km level dominates: 6 blocks x 16 x 1.7 ms = 0.16 s per sample-step in A100 fp32 units, plus 0.16 s for the rest; an H100 in bf16 runs the convs 3-4 x faster but the memory-bound parts less, so 0.3-0.6 s per sample-step, and 1.3 x that under block checkpointing. Per seed, training on 2017-2021 (1826 samples per epoch) with the full 1-2-4-8-16 recipe:

| Stage | Epochs | H100-hours at 0.3 s | at 0.6 s |
|---|---|---|---|
| 1-day | 30 | 4.6 | 9.1 |
| 2-day | 15 | 4.6 | 9.1 |
| 4-day | 10 | 6.1 | 12.2 |
| 8-day, pushforward 4 | 8 | 7.9 | 15.8 |
| 16-day, checkpointed | 6 | 15.6 | 31.2 |
| Total from scratch, 2017-2021 | | about 40 | about 80 |
| Total from scratch, 2001-2021 (x 4.2) | | about 170 | about 330 |
| Warm start from the 0.04 deg `rcan_s1` winner, stages 8 and 16 only | | about 25 | about 50 |

Add a factor of 1.5 for loader stalls and DDP inefficiency: 60-120 H100-hours per seed on 2017-2021, 250-500 on the whole record. The artifact's 846-892 H100-h per seed (2026-10-13) was for a full-field model from scratch without the coarse warm start; this plan's warm start is why the number falls. One warm-started seed on 2017-2024 is about 35-80 H100-hours; three are 100-250.

**Instance and storage.** One `p5.48xlarge` (8 x H100 80 GB, 2 TB RAM, 8 x 3.84 TB NVMe) holds the 1.65 TB int16 pack in the page cache after one epoch and runs the three seeds as three DDP jobs or one seed per 2-3 GPUs; a seed is 1-2 days wall. The whole-record pack (5.1 TB) streams from NVMe. Prefer `p5en.48xlarge` (H200 141 GB) if lazy targets are not built. Third-party trackers list `p5.48xlarge` at about $55/h on-demand in us-east-1 and `p5en.48xlarge` at about $63/h (unofficial, 2026 figures; confirm in the console); three warm-started seeds are then about 110 instance-hours, $6-7 K on-demand, less on Capacity Blocks. S3 for the pack and checkpoints: 2-6 TB, about $50-140 per month. Checkpoint every stage's `last.ckpt` to S3 and resume from it, since spot or capacity-block interruptions otherwise restart a stage (the 2026-10-22 lesson).

**DDP note.** Gradients are 40 MB; the all-reduce is nothing. Per-GPU step time is unchanged, so 8 GPUs give 8 x the samples per update, not 8 x the updates. The recipe's epoch counts were set for global batch 8; at global batch 32 (4 per GPU x 8) scale the learning rate by sqrt(4) with a warm-up, and run one A/B of validation loss against samples seen at the 1-day stage before trusting it (`docs/rea-compute.md`). Or run one seed per 2 GPUs and skip the question.

**What the 1 km emulator is for.** The 1/100 deg reanalysis is a different HYCOM configuration from GOMb0.04, and GrASE is 0.04 deg. The 0.04 deg levels and the mesh carry over as initialisation, not as a claim of transfer. Its test is its own 2024 (or a held-out year) and the same scorecard; scoring it against GrASE needs the regrid the alignment probes never built.

## 6. Implementation plan

Code goes to hycom-emulator on `feat/rcan` from `origin/main`. Tests run on RCC (CPU, synthetic packs as `tests/test_rea_unet.py` builds them); probes and runs on skynet GPUs 2-3 under the card. Every step ends with a test or a measured number.

1. **`src/hycom_emulator/rcan.py`: the model.** `RcanGraphLAM(GraphLAM)` reading `datastore.config["rcan"]`: `{width, levels: [{cell_deg, enc: [groups, blocks], dec: [groups, blocks]}], channel_attention, reduction, norm, checkpoint_blocks, tile: {size, halo} | null}`. Reuse `convnet.ResBlock`, `_image`, `_nodes`, `_run` and `_fill_new_modules`'s pattern. New modules: `ResidualGroup`, `RIR`, `SpaceToDepth` (reshape + 1x1 conv initialised to the mean), `PixelShuffleUp` (conv3x3 F -> s^2 F initialised to nearest replication, `nn.PixelShuffle`), `Head`. Level strides come from `cell_deg / grid spacing`, rounded, with a check against the pack's `x` spacing; padding by replication to a multiple of the stride, as `_pad` does. Register `rcan_graph_lam` in `MODELS`; add `REA_ARCHS` entries `rcan_s2`, `rcan_s2_light`, `rcan_s1`, `rcan_1km` and make `apply_arch` write the `rcan` section. Tests (`tests/test_rcan.py`): identity start reproduces `graph_lam`'s step up to the cell average on a synthetic pack; S2D then pixel shuffle at init is the identity on a constant field; a checkpoint with fewer levels loads with the new level at identity (state-dict pre-hook keyed by `cell_deg`); every graph_lam weight carries over; the tiled forward equals the untiled one to 1e-5 with halo >= receptive field, and differs with a smaller halo; `checkpoint_blocks` changes no output.
2. **Graph for a pyramid.** `nlam.build_graph` already builds `<graph>_s<k>` through `coarse_graph`; add a `mesh_from` option so the 1 km pack takes the mesh files of the 0.04 deg `multiscale` graph instead of running `create_graph` on its own 3.2 M nodes. Test: g2m sender counts per mesh node on the stride-2 pack at stride 2 match PR 42's (median 4, none without a sender).
3. **int16 pack.** `rea_pack build --dtype int16` writes standardised fixed point (scale 2000) with the scale in `meta.zarr`; `datastore._open` casts on read (the memmap stays int16, the cast happens per sample in the worker, so host shared memory halves too). Tests: round trip within 0.0005 std; a value beyond +-16 std is clipped and counted. Then a stride-1 int16 pack of the whole record is 335 GB and fits in 500 GB with one trainer, which is the stride-1 pretraining option if the fine-tune card wants it.
4. **Probe before any card (smoke-before-full-runs).** `rea_train probe` on skynet with `--model rcan_graph_lam --graph multiscale_s2` for `rcan_s2_light` and `rcan_s2`: 50 steps at 1-day batch 8 and at 16-day batch 8 with `--checkpoint_steps`, 16-mixed, 4 workers; record s/step, peak GPU, RssAnon/RssShmem. A CPU smoke of one epoch on a 2-month window through `train_rea.sh ARCH=rcan_s2` with the read-out. Replace section 3.3's estimates with the numbers and set the card's caps from them (cap = graph_lam's stage caps x the measured ratio, rounded up).
5. **Card emu-rea-rcan (workspace `runs/`, through `guard pr open`).** Arms: `rcan_s2` from scratch, seed 1; control: emu-rea-forcing base arm (3 seeds, its read-outs exist). Recipe, data, splits, read-out and hypotheses as in section 4. Budget: cap + 1.5 h read-out, with the rule that a run starts only while spent + cap stays inside it. Kill criteria as emu-rea-unet2's (NaN or loss > 100 after epoch 1; time cap; host memory admission at 620 GB and the youngest-step stop at 650 GB; 3 attempts for non-design fixes). Launcher: `runs/emu-rea-forcing-r2/track1.sh` after its known gaps are fixed (unknown MODE, `/dev/shm` cleanup, spend summing). `checks:` from the step 1-3 tests at the frozen commit plus the identity-start tolerance (one-step val loss at init within 1.2 x revert-fp16's on 4 val samples, as unet2 did). Skynet GPUs 2-3; GPUs 0-1 are busy as of 2026-11-04 (79 GB used, 100% utilisation each).
6. **Sharp stage on the winner** (if H1 passes or nearly passes): `finetune_rea.sh` with `--amse 1` on 32-day rollouts, 3 epochs, lr 2.5e-5, seed 1; emu-rea-sharp-s12's hypothesis. A successor card, since cards are immutable.
7. **Card emu-rea-rcan-s1** (gated on 5): `rcan_s1` grown from the winner on `rea_s1_2017.pack`, batch 4, stages 4/8/16 with the caps of section 5.1, seed 1; control: the winner itself scored on the same test starts at stride 2 (its scores regridded is not needed: compare Gulf column RMSE and the scorecard ratios, both resolution-independent in definition; the 10-50 km band is where stride 1 can differ). Hypothesis: H1 band 10-50 km variance ratio and front p95 improve at RMSE within 2%.
8. **1 km line** (gated on 7 and on the user's AWS decision; not part of the first build): (a) `rea_pack` on the GOMb0.01 layout, header check first (depths, 2d variables, topo), build 2017-2024 int16 on skynet, upload to S3; (b) lazy targets and the tiled 0.01 deg level in `rea_train`/`rcan.py`, tested on CPU with a synthetic 1 km-shaped pack cropped small; (c) an AMI or container with the skynet venv (`uv` lock from `pyproject.toml`, torch cu130), `train_rea.sh` with `--devices` for DDP, checkpoints synced to S3 each stage; (d) a 2-GPU smoke on a 2-month window, then the 1-day A/B of global batch 8 vs 32; (e) three warm-started seeds; (f) the same scorecard on the 1 km test year.

## 7. Decision rules and risks

- The card's `negative_result_means`: if `rcan_s2` fails H1 from scratch (one seed now, three before a verdict), a conv path between the grid and the mesh is not the blur lever at 0.08 deg, and the remaining levers are the target (adjusted MSE, isotach surrogate) and the probabilistic head; the stride-1 and 1 km lines do not run. If H1 passes and H2 fails, the added variance is misplaced (as emu-b00-053-conv); the sharp stage does not run and the next arm is `rcan_s2_light` or the stability term raised.
- Single-seed margins under about 5% at 16-21 days are noise (emu-rea-revert). The one-seed runs of this revision answer "is it promising"; a verdict needs seeds 2 and 3, and the hypotheses keep the control's seed range so the one seed is read against it.
- Host memory on skynet is shared and not ours: admit by measured footprint at every start, 4 loader workers, two trainers at most while a foreign process holds over 100 GB.
- Compile: `torch.compile` of a model with tiling or Python-level level loops may fall back; the probe says. `--checkpoint_steps` steps already run eagerly.
- The 1 km source files are unverified beyond their listing; step 8a starts with one day's header on skynet and stops if the 2d fields or depths differ from the 0.04 deg set.
- AWS prices above are from third-party trackers, not the AWS pricing page.
