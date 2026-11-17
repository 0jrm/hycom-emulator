# Plan: CRAM principles on the reanalysis emulator (2026-11-17)

Source: Hoover, Blayney and Whittington, "Classical Energies Learn to Reason 1,000x Faster than Looped Models",
arXiv 2611.17309v1. CRAM is a trainable Hopfield-Tank energy for puzzles. Its lessons, as this project reads them:

1. Learn a local energy with weights shared across positions, and train it with a local objective (pseudolikelihood:
   each cell's conditional given the true values of the others). Do not differentiate through the solver.
2. Put the search into inference: sample with noise, clamp what is known, run several chains.
3. A trivial state can have lower energy than the solution (Maze's empty board). The blurred squared-error forecast is
   our empty board, so an energy over ocean states must be used as a two-sided typical-set test, not minimised.
4. Small structured models first (Maze: 18 parameters). Non-local rules need a hidden layer.
5. Honesty: warm versus cold timings, test-set exposure, and easy benchmarks (our day 1) do not show skill.

**Decision of 2026-11-17 (the user):** RCAN and the deterministic sharpness line are parked. This answers the H1 branch
of emu-rea-blur-budget-r3 (day-7 slope at 50-100 km about 0.72, below 1: the forecast keeps more small-scale variance
than its skill supports, so there is no predictable signal left to sharpen).

## The four pieces

Code on hycom-emulator branch `feat/energy-cram`.

| Piece | Module | What it is | Card |
|---|---|---|---|
| C | `stencil.py` | dx(i) = W phi(x)(i) [+ b(i)]: a shared 5 x 5 linear stencil over all 91 channels, with the forcing, a per-cell bias arm (CRAM's B_i), and an arm with advection products u dq/dx, v dq/dy (pairwise terms). Least squares on one-day pairs, ridge chosen on held-out 2021. Runs as a `.pt` checkpoint through `rollout_rea`. | emu-rea-stencil |
| B | `realism.py` | Each channel predicted from its 5 x 5 window minus the cell's own column (Gaussian pseudolikelihood). E_f = mean squared standardized residual per field. A state is realistic when log E_f lies in [q01, q99] of 2021 truth. Blur pushes E below the band and noise pushes it above. | emu-rea-realism |
| A | `corrector.py` | Per lead bin: a ridge mean correction A_b phi(x_hat) fitted on base-s1 free runs from train years, then members drawn by block Gibbs on the residual. Conditionals and column covariance come from train residuals; the boundary band and seabed are clamped (CRAM's givens); temperature T_b gives spread/error = 1 on 2021 runs. | emu-rea-corrector |
| D | `column_energy.py` | Thesis link: a discrete CRAM energy over T and S at 22 levels plus SSH in a column, Q quantile bins, pairwise tables, PL training. Clamp SSH and SST and Gibbs-sample the profile. Compared with a Gaussian regression and climatology. | nsp-column-energy |

## Order and cost

All four fit in under 6 GPU-hours on skynet GPUs 2-3, because no rollout is ever differentiated. Probes use train years
only (fit 2001-2020, held-out 2021). Cards fix their hypotheses before any 2022-2024 number is computed. Val
(2022-2023) and test (2024, out of regime) are scored once, on emu-rea-blur-budget-r3's starts (every 10 days), against
the base seeds and persistence that r3 already reports.

## What is not in scope

- Training any neural network.
- The neural denoiser version of A. It is the follow-up only if the linear-Gaussian corrector shows the shape works but
  is too weak.
- In-loop noise (members fed back into the emulator). It is a follow-up after the post-hoc corrector is scored.
- RCAN, `rcan_s2_v2` and the 1 km emulator.
- NeSPReSO's own Argo test set. D runs on reanalysis columns; an Argo comparison belongs in the thesis plan.
