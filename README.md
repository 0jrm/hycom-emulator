# hycom-emulator

Learned, differentiable surrogate of one HYCOM + TSIS cycle on GOMb0.04 (41 layers). It has two modules:

- A, analysis: background + observations → increment.
- B, 24 h step: state + increment → state 24 h later, with the increment applied by 24 h IAU.

The target is the GrASE period, 2025-04-01 to 2025-09-30.

The repo is a nested clone of the gom-da workspace (`hycom-emulator/` next to `src/gom_da`). Its rules apply here: read the workspace `AGENTS.md` first. Code moves by git only. You write and test on RCC, then `git pull` on skynet to train. Data moves separately.

## Setup (RCC)

```bash
module load python-uv/0.9.7
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[build,test]"
.venv/bin/python -m pytest
```

`.[build]` installs the workspace's `gom-da` from `..`. It is only needed where HYCOM/TSIS files are read.

## Systems

Each data source is one `configs/systems/<name>.toml`. Print its configuration fingerprint with:

```bash
.venv/bin/python -m hycom_emulator.system configs/systems/abozec_054.toml
```

The fingerprint has two digests:

- `structural` hashes the configuration: every blkdat entry except step sizes, output frequencies and labels, plus the TSIS namelist and build, forcing, and the relaxation mask.
- `strict` hashes the structural fields plus the step sizes, the executables and the initial restart.

A checkpoint is deployable only if its last fine-tune ran on a system whose `structural` digest equals the canonical GrASE system's.

## Catalog

```bash
.venv/bin/python -m hycom_emulator.catalog configs/systems/abozec_054.toml
```

This lists every product a system kept and, for each cycle, which roles are present. A cycle is named by its analysis time t_a at 18Z. Roles are files at fixed offsets from t_a: the 24 h-mean background parts, the 00Z snapshot, the increment, and the obs/inov files. They are defined in `catalog.ROLES`.

## Setup (skynet, GPU training)

```bash
git clone git@github.com:0jrm/hycom-emulator.git /unity/g2/jmiranda/hycom-emulator
cd /unity/g2/jmiranda/hycom-emulator
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[ml,test]" --torch-backend cu130
```

Training data is copied from RCC, never code. Use only GPUs 0–2 (workspace rule).
