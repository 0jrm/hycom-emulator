# RCAN around the mesh, redesigned from first principles (draft, 2026-11-04)

Proposal, not a card. It follows the first rcan_s2 run (emu-rea-rcan, stopped 2026-11-04 after a free-run blow-up,
incident 2), a reading of the RCAN paper (Zhang et al., ECCV 2018) against `src/hycom_emulator/rcan.py`, and the five
points raised on it: input range, depth, channel attention, normalization and learning rate, loss. The design in
`docs/plan-2026-11-04-rcan-graph-emulator.md` stays as written for the running card; this note says what the next
build should look like if the idea is started again from the task rather than from the paper.

## 1. What the task is, and what RCAN is for

Super-resolution maps a low-resolution image to the one high-resolution image that produced it. The mapping is local
and deterministic, so a very deep, narrow, un-normalized conv stack with long skips learns it, and L1 keeps the
edges sharp. Three properties made the paper's choices work: bounded inputs ([0, 1] pixels), a deterministic target,
and no state carried between calls.

The emulator shares one of them at short leads. One day of sub-mesh evolution is close to deterministic: the mesoscale
field advects fronts by a few cells, and the front's shape is a local function of yesterday's field. That is the part
an RCAN-style decoder can learn. It does not share the other two. Inputs are standardized anomalies with no bound, and
the model is called on its own output for 16 to 60 days, so any path that is linear in the input becomes a feedback
loop. Beyond a few days the target is not deterministic, and no architecture removes the blur a mean-squared loss
then asks for.

So the correct analogy is narrower than "RCAN with a graph in the middle": the mesh message at the coupled level is the
low-resolution image, the fine decoder is the super-resolution network, and the encoder is only the degradation
operator (space-to-depth) plus whatever the mesh needs to see. Everything below follows from that.

## 2. The redesign

### 2.1 Inputs bounded by construction (point 1)

The blow-up came from a 1250 m salinity cell at 35σ. The cause is upstream of the model: `rea_pack` standardizes each
channel with one Gulf-wide std and floors the change std at 5% of the state std (`DIFF_STD_FLOOR`), but puts no floor
on the state std itself (`rea_pack.py:527`). At depth the std is tiny, so ordinary bottom-layer variability becomes an
extreme feature, and the same std sets the loss weight of that channel.

- **Floor the state std in the pack.** For each level variable, `state_std = max(raw_std, floor)`, with the floor the
  larger of a physical minimum (0.02 psu, 0.05 °C, 0.01 m/s) and 10% of that variable's median std over the column.
  Record the floored channels in `meta.zarr` attrs, as the int16 clip counts are recorded now. Every model and every
  loss inherit the fix; the 10σ head clip and the 16σ int16 clip stop being load-bearing.
- **One entry into the residual stream, normalized.** Remove `head`. The grid embedder (MLP ending in a LayerNorm)
  already gives each point a bounded feature vector and the first block's 3x3 conv reads the neighbourhood of it. A
  zero-initialised head feeding the same stream duplicates that path with no bound. Dropping it keeps GraphLAM
  checkpoints loadable, since the head starts at zero anyway.
- **Bound the exit as well.** The plan recorded that the RMS of `output_map`'s input grew from 1.8 to 7.5 in 30
  steps under rcan while graph_lam's stayed at 1.7. The decoder writes `nodes(d)` into the output sum with no norm.
  Read it through a per-point LayerNorm followed by a zero-initialised 1x1 conv, the same discipline every other
  write into the stream already has. At init the model is still GraphLAM's step exactly.

### 2.2 Depth where the sharpness is made (point 2)

RCAN is 200 blocks deep at 64 channels; rcan_s2 is 6 blocks per module at 128. The long and short skips pay for depth
and are close to decorative at 6 blocks. Width is pinned to `hidden_dim` (128) by the mesh, so the only lever is depth,
and the place to spend it is the fine decoder, which is the super-resolution network in this analogy.

| level | encoder (groups x blocks) | decoder (groups x blocks) | role |
|---|---|---|---|
| 0.08° (grid) | 1 x 3 | 3 x 4 | encoder: what the mesh needs to see; decoder: the SR network |
| 0.16° (coupled) | 1 x 3 | 1 x 3 | unchanged |

Fine-level blocks go from 12 to 15, so the step cost rises by roughly a quarter of the fine level's share, about
1.1-1.2x rcan_s2. Receptive radius of the fine decoder: 3 groups x (2 x 4 + 1) + 1 = 28 cells, about 2.2°, which
covers several days of advection. If the first result is positive but short of the margin, the next arm is 4 x 5 in
the decoder, not a wider stream.

For the 1 km build the same rule holds: the 0.01° level gets a light encoder (1 x 2) and a deeper decoder (2 x 4),
tiled, and the 0.04° and coarser levels keep the shape above.

### 2.3 Channel attention off at the fine level (point 3)

In the paper CA is worth 0.03 dB over the skips. In the emulator each block's squeeze pools over the whole Gulf, so
every block receives a per-sample global gate, which is one more feedback path in the rollout and breaks the exact
tiling the 1 km path depends on. Default `channel_attention: false` at the finest level, keep it available at the
coupled level where the field is 132 x 97 cells and tiling is not needed, and run CA-on at 0.08° as an ablation arm,
not a fixture. The parameter saving is nominal; the gain is a model whose fine level is a pure local operator.

### 2.4 Pre-norm stays; the schedule gets a warm-up and a lower peak (point 4)

EDSR and RCAN dropped BatchNorm because it normalizes away the image's range. Per-point pre-norm LayerNorm is not
that: the residual stream itself is never normalized, only each block's read of it, and the zero-initialised second
conv is a learned residual scale (EDSR's fixed 0.1, learned from zero). Keep both.

What the paper does say is lr 1e-4 at batch 16 of 48 x 48 patches. The recipe starts at 1e-3 on full 263 x 193 fields
at batch 8, which is roughly 150x more pixels per step, so a higher rate is defensible, but the first rcan build
diverged there and was rescued by norms. Add a linear warm-up of 500 steps to the first stage and cap the from-scratch
peak at 5e-4; the later stages already start at or below 1e-4. This is an execution change under the frozen design and
goes in `execution.tsv`.

### 2.5 Loss: wmse for the verdict, a sharpness fine-tune after (point 5)

RCAN's edges come partly from L1. The failed `grad` arm was L1 on error gradients, not L1 on the state; plain L1 or
Charbonnier on standardized states with the same channel weights has not been run. It still cannot sharpen what is
unpredictable at 16 days, so it belongs where adjusted MSE already sits in the plan: a fine-tune stage on the winner,
after the architecture verdict, with Charbonnier (eps 1e-3) as a second arm beside `--amse 1`. The card keeps
`rea_wmse` with the stability term so the comparison with the control is clean.

### 2.6 What stays as built

Space-to-depth down with a mean start, pixel shuffle up with a nearest start, the zero-initialised long-skip and
level-skip convs, per-block checkpointing, tiling with a halo, level growth from a coarser checkpoint, the mask as an
extra input channel to each block's first conv. These match the paper where the paper applies and are sound where it
does not.

## 2.7 Beyond RCAN: what the field suggests, ranked by expected effect on the blur

The failure is blur and a low kinetic-energy ratio from day 1. The items below are ordered by how much each should
move that, with the evidence they rest on.

1. **Probabilistic training for the long-lead blur (first after the architecture verdict).** Past the predictability
   limit the mean field is the optimum of every squared error, so no deterministic model keeps the day-21 KE ratio.
   Every current emulator that stays sharp is probabilistic: GenCast (diffusion), AIFS-CRPS and NeuralGCM (noise
   injection, CRPS on a small ensemble). The CRPS route is cheap here: a latent noise vector concatenated to the mesh
   embedding, two members per batch, the fair CRPS of the Gulf functionals (the `rea_wmse` quantities) as the loss on
   the winner's checkpoint. About twice a deterministic step. The KE ratio and the 50-100 km variance read-outs are
   exactly what it restores. Adjusted MSE (section 2.5) is the deterministic half-step toward it.
2. **Physics-informed changes that fit the pyramid, cheap to add in the next build.**
   - *Partial convolutions.* Land and below-bottom cells hold one constant and every 3x3 conv reads them, so coastal
     and bottom cells see a fake neighbour. A partial conv (Liu et al. 2018) renormalises each tap over the valid
     cells in its window; one change in `ResBlock`, with the mask the block already receives.
   - *Column mixing.* The 22 depths are independent channels. A per-point MLP or a small attention over the depth
     axis with depth as the position gives the model the column's ordering and makes the stability penalty easier to
     satisfy; Aurora and Pangu treat levels as an axis for the same reason. Pointwise, so it is cheap and tiles exactly.
   - *An advection prior.* Fronts blur because a conv regresses them. A warp layer at the fine level that predicts a
     displacement field and resamples the previous state (`grid_sample`) before the decoder's conv correction keeps
     gradients by construction, as semi-Lagrangian schemes do. The one architectural idea that targets day-1 sharpness
     directly; medium effort; it composes with the pyramid as the first operation of the fine decoder.
   - *Missing forcing.* The pack carries wind, day of year and insolation, no heat or freshwater flux. If the GOFS
     forcing fields exist on skynet, adding them is more physics than any loss term. One look on disk settles it.
3. **Attention and transformers, where they would go.**
   - *Fine level: Swin-style window attention blocks* in place of conv blocks in the decoder (SwinIR, HAT), the same
     residual-in-residual structure with a few tenths of a dB over RCAN on super-resolution. Shifted windows keep a
     bounded receptive field, so tiling with a halo stays exact, which the 1 km path needs. A later arm.
   - *Mesh level: an attention processor* over the 6,561 mesh nodes instead of the 4-layer GNN (GenCast's graph
     transformer). Full attention at that size is cheap with fused kernels. The Loop Current is a long-range feature,
     so this is the arm that could help beyond the mesoscale; gain for the ocean unknown.
   - *Not recommended:* a plain ViT over the grid (blurs at the patch scale), Fourier neural operator layers in place
     of the mesh (no published win over graphs on a bounded domain with land), PDE-residual losses (primitive
     equations on a reanalysis with unresolved physics have not helped any emulator so far).
4. **Use RCAN where it is RCAN.** The 1 km plan trains the emulator at 0.01°. The alternative is to emulate at 0.04°
   or 0.08° and learn a true super-resolution downscaler from coarsened 1/100° fields to the 1/100° truth, conditioned
   on bathymetry. A deterministic, local, paired mapping is the case the paper's design was made for, at a fraction of
   a 1 km emulator's cost.

Order to run them: the CRPS ensemble on the winner; partial convolutions and column mixing folded into the next
build; the warp prior as its own arm; the transformer arms once there is a stable baseline to compare against.

## 3. Order of work

1. State-std floor in `rea_pack` (stats only; the int16 payload is unchanged if the floor is applied at read time
   through `_FixedPoint`, otherwise repack), with the floored channels listed in meta attrs and a test that no
   standardized state in the train split exceeds a stated bound.
2. `rcan.py`: remove `head`, add the exit LayerNorm and zero 1x1, make `channel_attention` per level, keep the
   outlier regression test from `fix/rcan-head` (it should pass by construction, with no clip constant).
3. `REA_ARCHS["rcan_s2_v2"]` with the 2.2 table; probe 50 steps against graph_lam and rcan_s2 for s/step, peak
   memory and the output_map input RMS.
4. Warm-up and peak-rate options in `train_rea.sh`, logged as execution changes.
5. Smoke (1-2-4 days, two months), then one seed on the full recipe; the same read-outs and hypotheses as the
   running card, against the control's three-seed mean and range.
6. Fine-tune arms on the winner: `--amse 1` and Charbonnier.

## 4. Risks

- The floored std changes the input scale of the deep channels, so nothing trained on the old stats transfers. That
  is the price of fixing it in the data; the clip was the only path that kept emu-rea-rcan's checkpoint, and it
  costs a tuned constant in the model forever.
- Removing `head` removes the one path that carried per-point magnitude into the conv stream. The embedder's LayerNorm
  keeps direction, not size. If the decoder turns out to need magnitude, the fix is a std floor plus a bounded
  (tanh) head, not an unbounded one.
- A deeper decoder with CA off at the fine level is further from "RCAN" by name. The name was never the point; the
  mechanism (deep local residual stack after the coarse representation, long skips, sub-pixel upscale) is intact.

## 5. As built and measured (2026-11-05, feature/rcan-v2)

Steps 1-5 of section 3, except the full-recipe run, which waits for the card emu-rea-rcan-v2 and the user.

**State-std floor (2.1).** `rea_pack` floors each level variable's state std at max(0.05 degC, 0.02 psu, 0.01 m/s;
10% of the column's median std) and keeps the raw std as `state_std_raw`; meta.zarr attrs list the floored channels and
the floor per variable. The change std keeps its floor on the raw state std. neural-lam's wmse weights a standardized
error by state_std / state_diff_std, so each channel's loss weight in physical units is 1 / state_diff_std and does not
move with the floor: the loss stays the control's. emu-rea-rcan read the float32 `rea_s2.pack`, so no int16 rescaling
is needed for it; `rea_pack refloor` writes `rea_s2_floor.pack`, which links rea_s2's arrays and holds the floored
meta.zarr (rea_s2's own statistics stay, since every other run reads them). For an int16 pack refloor keeps the std
the values were quantized with as `state_int16_std`, and the datastore decodes with it.

On rea_s2 the floors are 0.254 degC, 0.026 psu, 0.023 and 0.028 m/s, and only T and S at 1250, 1500 and 2000 m are
floored. `rea_pack zscan` (every 10th train row, 766 rows, real points of each level):

| | all points | Gulf |
|---|---|---|
| largest standardized \|state\|, raw std | 50.5 sigma (temp_2000m) | 27.0 (salin_4m) |
| same, floored std | 40.2 (salin_70m, 81.7W 22.5N) | 27.0 (salin_4m, Atchafalaya plume) |
| channels above 12 sigma after the floor | 31 of 91 | 10 |
| deep T/S (1250-2000 m), largest after / before | 35.1 / 47.0 (salin_1250m) | 12.7 / 23.6 |

So the floor as written does not give the 12-sigma bound section 3 step 1 asks for. What stays above it is mostly real
heavy tails that no per-channel std can bound without crushing the upper ocean's scaling: river plumes in surface
salinity, the Florida Current and Gulf Stream velocities near the east edge, a cell off Cuba with salinity 6 psu off
the mean at 40-100 m for months in 2010, and bottom-layer cells such as the blow-up cell (85.44W 23.11N, bottom at
1250 m, whose 2024 truth reaches 26 sigma in temp and 42 sigma in salin after the floor). The bound test
(`tests/test_rea_pack_bound.py`, REA_PACK=<pack>) is a strict xfail that records these numbers. The model, not the
pack, bounds them: see the free run below.

**Model (2.1-2.3).** `head` is gone; the grid embedder is the stream's one entry. The decoder writes into the output
sum through `exit_norm` (LayerNorm) and `exit` (1x1, zero-initialised), beside the mesh message replicated to the grid,
so at init the model is what it was (GraphLAM exactly with one level) and a GraphLAM checkpoint loads; old rcan_s2
checkpoints with a `head` no longer load strictly. A level's `channel_attention` overrides the section's.
`REA_ARCHS["rcan_s2_v2"]`: 0.08 deg 1 x 3 / 3 x 4, attention off; 0.16 deg 1 x 3 / 1 x 3, attention on. The outlier
test (one channel at 1000 sigma) passes with no clip, on the fine stream and on the output map's input.

**Schedule (2.4).** `rea_train --warmup N` (linear, multiplicative with Plateau, count kept in the checkpoint);
`train_rea.sh` defaults for an rcan ARCH: LR 5e-4 (peak and every stage's cap), WARMUP 500 in the first stage when
it starts from scratch; other arches unchanged.

**Probe** (skynet GPU 3, 2026-11-05, code ded3199, rea_s2_floor.pack, 50 steps, batch 8, 4 workers, 16-mixed,
--compile --nondeterministic, rea_wmse --stability 0.137; 1-day from scratch at lr 1e-3, seed 1; 16-day pushforward 4,
--train_first_step, --checkpoint_steps from revert-fp16; output_map input RMS by a forward hook, one graph break):

| | 1-day s/step | peak GPU | output_map input RMS, step 1 / 10 / 30 / 50 | 16-day s/step | peak GPU | host anon + shmem (16-day) |
|---|---|---|---|---|---|---|
| graph_lam | 0.114 | 12.1 GiB | 1.83 / 1.81 / 1.78 / 1.78 | 3.02 | 22.2 GiB | 7 + 43 GB |
| rcan_s2 (with the new exit) | 0.371 (3.3x) | 11.1 GiB | 1.83 / 1.90 / 2.31 / 2.39 | 7.93 (2.6x) | 19.4 GiB | 5 + 40 GB |
| rcan_s2_v2 | 0.342 (3.0x) | 11.1 GiB | 1.83 / 1.95 / 2.25 / 2.42 | 8.23 (2.7x) | 19.4 GiB | 5 + 40 GB |

The RMS does not stay flat under rcan from scratch at 1e-3: it grows 1.3x in 50 steps, where the first build's grew
4x in 30 (1.8 to 7.5); graph_lam's falls slightly. This probe did not split the sum into its terms (the encoder
stream `fine + encoding_grid_mlp(fine)`, which enters unnormalized as in GraphLAM, the replicated mesh message, and the
exit, which is bounded by its weights but not fixed); over 60 free-run days below the sum stays flat. Loaded from revert-fp16 the 16-day RMS is flat (0.80 to 0.77).

**Smoke** (train_rea.sh ARCH=rcan_s2_v2, 2021-01-01..03-01, stages 1-2-4 one epoch each, validation losses):

| | 1-day | 2-day | 4-day |
|---|---|---|---|
| rcan_s2_v2, LR 1e-3, no warm-up (the plan's smoke schedule) | 2.20 | 2.145 | 2.151 |
| rcan_s2_v2, the script's defaults (LR 5e-4, WARMUP 500) | 3.11 | 2.198 | 2.14 |
| rcan_s2, plan section 3.3 | 2.27 | 2.19 | 2.20 |
| graph_lam, plan section 3.3 | 3.19 | 2.47 | 2.33 |

The validation loss is wmse in physical units over the change std, which the floor leaves unchanged, so the rows
compare. With the defaults the 1-day stage's 7 steps never leave the warm-up (lr at most 7e-6) and the later stages
catch up. Read-out in smoke size: 4 of 4 files.

**Free run** (freerun.py, 60 days from 2024-07-01, smoke A's 4-day best): largest normalised Gulf SSH error 0.37
sigma at day 1, 2.2 at 10, 4.0 at 30, 5.9 at 50 (the maximum), 5.4 at 60, near 88.5W 26.5N, not at the deep cells (the
first build: 8940 sigma by day 15, NaN by 55). Stream norms at the blow-up cell (85.44W 23.11N), days 1 / 30 / 60: fine
encoder 19.3 / 19.0 / 19.0, fine decoder 216.5 / 216.8 / 217.0, exit 2.3 / 2.3 / 2.3, output map input 29.0 / 29.4 /
29.5; at 84.64W 25.51N (bottom 1250 m): 16.8 / 18.6 / 19.5, 215.9 / 217.7 / 217.6, 2.3 / 2.3 / 2.3, 26.2 / 28.3 / 29.5.
The field-wide maximum of the output map's input stays 32.5-32.7. The bottom T and S errors at the blow-up cell are
large (10-26 sigma of the floored std) and follow the truth's swings there without growing: a two-month smoke does
not learn those cells, and nothing feeds them back.
