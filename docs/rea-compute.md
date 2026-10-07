# Compute plan for the reanalysis emulator: GPU memory, batch size, data serving, 1-3 GPUs

Companion to `docs/rea-pipeline.md`. Numbers marked **measured** come from track E runs on skynet (A100 80 GB, `graph_lam` multiscale, hidden 128, 4 processor layers, fp32; memory notes of 2026-10-01..06). The rest are **estimates** from those numbers. The first smoke run on GPU must replace them.

## Hardware (measured 2026-10-06)

- **GPUs.** 4 × A100 80 GB. Every pair is NVLinked (NV4). GPUs 0-1 sit on NUMA node 0 (CPUs 0-63, 128-191), GPUs 2-3 on node 1 (64-127, 192-255). We use GPU 3. GPUs 1-2 belong to another user today, and GPU 0 is never ours.
- **CPU and memory.** 2 × EPYC 7763 (256 threads), 1007 GB RAM, `/dev/shm` 504 GB, `/scratch` 678 GB free. About 90 GB is other users' baseline, and the live track E runs add 150-200 GB per arm.
- **Software.** neural-lam takes `--devices 1 2 3` and runs DistributedDataParallel through Lightning (`strategy="auto"`, one process per GPU, each with its own loader workers).

## GPU memory model

Fitting the measured B00 points (full 525 × 385 grid, 202 125 nodes):

| B00 point | Measured | Model `M0 + c·b·s` (M0 9 GB, c 7.5 GB per sample-step) |
|---|---|---|
| bs1, 1 step | 16 GB | 16.5 |
| bs4, 1 step | 36 GB | 39 |
| bs8, 1 step | 70 GB | 69 |
| bs4, 2 steps | 69 GB | 69 |
| bs4, 3 steps | OOM | 99 |
| hidden 256, bs1 | 31 GB | c doubles with width |

Activations on the grid nodes dominate, and they scale with hidden width × nodes × rollout steps × batch. The reanalysis state has 91 channels against B00's 209, which shrinks only the encoder input. The estimate is c ≈ 6-7.5 GB per sample-step at stride 1, and about 4× less at stride 2 (50 759 nodes): c ≈ 1.5-1.9 GB, M0 ≈ 4 GB.

Largest batch per GPU that stays under about 70 GB (10% headroom):

| Grid | Rollout | Hidden 128 | Hidden 256 |
|---|---|---|---|
| stride 1 (0.04°) | 1 day | 8 | 4 |
| stride 1 | 2 days | 4 | 2 |
| stride 1 | 4 days | 2 (tight; 1 safe) | 1 (tight; checkpointing for headroom) |
| stride 2 (0.08°) | 4 days | 8 | 4 |
| stride 2 | 8 days | 4 | 2 |

**Extra GPUs do not change this table.** DDP keeps a full model and its own micro-batch on every GPU, so the per-GPU memory, and with it the longest rollout and the widest model, stay the same. Longer rollouts at full resolution come from activation checkpointing (it took ConvGraphLAM from OOM to 69 GB at 1.43 s/step against 0.75, measured 2026-10-05) or from stride 2, never from more GPUs. Processor depth is cheap (measured), so add layers before widening.

## Throughput

- **Step time.** Measured B00: 0.75 s/step at bs4 × 2 steps, i.e. about 0.094 s per sample-step on the full grid when the GPU is the limit. Estimates: stride 1 ≈ 0.08-0.09 s per sample-step, stride 2 ≈ 0.025-0.035.
- **Loader.** Measured B00: about 4 GB/s of copies between processes with 8 forked workers (1.2 batches/s at bs4, 0.9 GB samples); more workers or pinned memory did not help. One reanalysis sample (2 initial rows + 4 targets, 91 channels, plus forcing windows) is 0.49 GB at stride 1 and 0.12 GB at stride 2.

| Stage | Per-GPU batch | GB per step | Est. GPU s/step | GB/s needed per GPU |
|---|---|---|---|---|
| stride 2, 4-day rollout | 8 | 0.96 | 0.8-1.1 | 0.9-1.2 |
| stride 1, 4-day rollout | 2 | 0.98 | 0.65-0.75 | 1.3-1.5 |

One GPU sits well under the 4 GB/s ceiling, so the GPU sets the pace. Three GPUs need about 3-4.5 GB/s in total. One question decides whether that holds: is the 4 GB/s ceiling per process or for the whole host?

- **Per process.** Each DDP rank has its own main process and workers, so three ranks should scale.
- **Host-wide.** The cause would then be memory bandwidth or page-cache contention, and three ranks would end up loader-bound near 4 GB/s.

Measure it in the first multi-GPU smoke: steps/s at 1, 2 and 3 ranks.

Two placement rules help either way:
- Pin each rank's workers to the CPU node of its GPU (GPU 1 to node 0; GPUs 2, 3 to node 1).
- Stage `/dev/shm` with `numactl --interleave=all`, so no single NUMA node serves every page.

## Host RAM

- **The pack is shared, not copied.** It is memory-mapped from `/dev/shm`, so all ranks and workers share one page-cache copy. Stride 2 over 2001-2024 is 168 GB. Stride 1 over 2017-2024 is 217 GB. Both together (385 GB) fit in `/dev/shm`'s 504 GB, but not next to live runs.
- **Per-rank memory.** It is in-flight batches (workers × prefetch 2 × batch × sample ≈ 15-16 GB for either stage above) plus per-worker Python/torch overhead.
  - Track E measured 150-200 GB per arm with 0.9 GB samples, against an in-flight estimate of about 60 GB. So the per-worker overhead is large and not understood.
  - Until a smoke run measures it, budget 60-100 GB per rank.

| Setup | Pack in `/dev/shm` | Ranks × per-rank | Above the 90 GB baseline | Fits next to live E runs (about 320 GB)? |
|---|---|---|---|---|
| stride 2, 1 GPU | 168 | 1 × 60-100 | 230-270 GB | no (640-680 GB in use, over the 600 GB watchdog) |
| stride 2, 3 GPUs | 168 | 3 × 60-100 | 350-470 GB | no |
| stride 1 fine-tune, 3 GPUs | 217 | 3 × 60-100 | 400-520 GB | no |

So the reanalysis runs start only after track E's queue drains (E2 is the last card, then the freeze rule decides).

## What 2 or 3 GPUs buy

**Epochs: yes, nearly linearly.** Gradients are about 5 M parameters (20 MB; a 58 MB unet_norm checkpoint holds weights plus two Adam moments), and an all-reduce over NVLink costs under a millisecond. Per-GPU step time is unchanged, so an epoch takes about 1/n the time if the loader keeps up.

| Stage (train rows) | 1 GPU | 2 GPUs | 3 GPUs |
|---|---|---|---|
| stride 2 pretrain, 7655 days, bs8 per GPU, 4-day rollout | 956 steps × ≈1 s ≈ 16 min/epoch | ≈ 8 min | ≈ 5.5 min |
| stride 1 fine-tune, 1826 days (2017-2021), bs2 per GPU, 4-day rollout | 913 × ≈0.7 s ≈ 11 min/epoch | ≈ 5.5 min | ≈ 3.8 min |

**Updates: no.** The global batch is n × the per-GPU batch, so each epoch has n times fewer optimizer steps. Track E compared runs at equal update counts (E2 matched E1's updates). At a fixed number of updates DDP gives no speed-up, only a bigger batch per update. So the gain is real only if a larger global batch learns as much per sample. That holds for these models up to batches of tens (GraphCast trained at 32), usually with the learning rate scaled by sqrt(n) and a warm-up. It has to be shown here with one short A/B on validation loss against samples seen: global batch 8 on 1 GPU against 24 on 3 GPUs.

**For ablations, run one arm per GPU instead.** There is no synchronisation, no learning-rate question, and the arms share one `/dev/shm` pack. Track E ran two arms that way, and RAM capped it at two: a third pushed it past 600 GB. That was with 39 GB packs and 150-200 GB per arm. The shared pack makes the reanalysis cheaper per extra arm, but the per-arm overhead still caps it at 2-3.

**Recommendation.**
1. Pretrain at stride 2, hidden 128, 4-6 processor layers, per-GPU batch 8.
   - 1 GPU: about 16 min/epoch.
   - If GPUs 1-2 free up: DDP on 3 with lr × sqrt(3), after the A/B above.
2. Fine-tune at stride 1, per-GPU batch 2, 1 → 2 → 4-day curriculum, activation checkpointing for the 4-day stage if batch 2 does not fit.
3. Run ablations one arm per GPU, never more than the RAM table allows.

**Measure in the first GPU smoke:**
- peak GPU memory at stride 2, bs8, 4 steps;
- per-rank host RSS;
- loader steps/s at 1, 2 and 3 ranks.

Then replace every estimate in this file with the measured value.

## Options to cut data movement (not needed for 1 GPU)

- **Whole pack on the GPU.** The stride-4 pack (42 GB) fits on one GPU, and a stride-2 subset of about 10 years (about 70 GB) nearly fits. That removes the loader for quick experiments.
- **float16 storage of standardized values.** It halves bytes. The resolution near ±4 std is about 0.004 std. Check it against the daily-change std per channel before use, because changes are small next to the state std.
