# hycom-emulator

Learned, differentiable surrogate of one HYCOM + TSIS cycle on GOMb0.04 (41 layers). It has two modules:

- A, analysis: background + observations → increment.
- B, 24 h step: state + increment → state 24 h later, with the increment applied by 24 h IAU.

The target is the GrASE period, 2025-04-01 to 2025-09-30.

The repo is a nested clone of the gom-da workspace (`hycom-emulator/` next to `src/gom_da`). Its rules apply here: read the workspace `AGENTS.md` first. Code moves by git only. You write and test on RCC, then `git pull` on skynet to train. Data moves separately.
