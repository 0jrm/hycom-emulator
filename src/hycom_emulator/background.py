"""The 24 h-mean background TSIS analysed, rebuilt from the cycle's two partial-mean archives.

do_archm1d.csh feeds archm(t_a-21h) (6 h mean) and archm(t_a-9h) (18 h mean) to hycom_mnsq, then
archv2restart turns the result into the restart xprep reads. hycom_mnsq (meanstd/src/mod_mean.F)
weights each archive by its record count n and layer mass oneta*dp', forms T and S as mass-weighted
means, and copies T, S from the layer above where the mean thickness is below 1e-6 Pa.

This reproduces TSIS H(xb) to float32 in every layer at or above that threshold
(tests/test_background.py). In massless layers archv2restart then resets T, which is not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gom_da.eval.hycom_archive import Grid, load_2d, load_3d, parse_archv_index
from gom_da.eval.timeline import parse_blkdat

from hycom_emulator.catalog import Cycle
from hycom_emulator.system import SystemConfig

MASSLESS_PA = 1e-6  # mod_mean.F: dw_m < 0.000001 projects T, S from the layer above
LAYER_FIELDS = ("temp", "salin")
SURFACE_FIELDS = ("srfhgt", "montg1")


@dataclass(frozen=True)
class MeanArchive:
    n: int
    oneta: np.ndarray  # (j, i)
    thknss: np.ndarray  # (k, j, i), dp' in Pa
    layer: dict[str, np.ndarray]
    surface: dict[str, np.ndarray]


@dataclass(frozen=True)
class Background:
    thknss: np.ndarray  # (k, j, i), dp' in Pa, as archived
    oneta: np.ndarray
    layer: dict[str, np.ndarray]
    surface: dict[str, np.ndarray]


def grid_of(cfg: SystemConfig) -> Grid:
    blk = parse_blkdat(cfg.blkdat.read_text())
    return Grid(int(blk["idm"]), int(blk["jdm"]), int(blk["kdm"]), np.empty(0), np.empty(0))


def record_count(bpath: Path) -> int:
    """The `no. recs` column of a mean archive, read from its first thknss line."""
    for line in Path(bpath).read_text().splitlines():
        if line.startswith("thknss"):
            return int(line.split("=", 1)[1].split()[0])
    raise ValueError(f"no thknss record in {bpath}")


def read_mean_archive(apath: Path, grid: Grid) -> MeanArchive:
    bpath = apath.with_suffix(".b")
    idx = parse_archv_index(bpath)
    return MeanArchive(
        n=record_count(bpath),
        oneta=load_2d(apath, idx, "oneta", grid),
        thknss=load_3d(apath, idx, "thknss", grid),
        layer={f: load_3d(apath, idx, f, grid) for f in LAYER_FIELDS},
        surface={f: load_2d(apath, idx, f, grid) for f in SURFACE_FIELDS},
    )


def mean_background(parts: list[MeanArchive]) -> Background:
    n_total = sum(p.n for p in parts)
    weights = [p.n * p.oneta * p.thknss for p in parts]
    mass = sum(weights)
    dw = mass / n_total
    layer = {}
    for f in LAYER_FIELDS:
        x = sum(w * p.layer[f] for w, p in zip(weights, parts)) / np.where(mass > 0, mass, 1.0)
        for k in range(1, x.shape[0]):
            thin = dw[k] < MASSLESS_PA
            x[k][thin] = x[k - 1][thin]
        layer[f] = x
    oneta = sum(p.n * p.oneta for p in parts) / n_total
    return Background(
        thknss=dw / oneta,
        oneta=oneta,
        layer=layer,
        surface={f: sum(p.n * p.surface[f] for p in parts) / n_total for f in SURFACE_FIELDS},
    )


def cycle_background(cycle: Cycle, grid: Grid) -> Background:
    parts = [cycle.files[role] for role in ("archm_21", "archm_09")]
    if any(p is None for p in parts):
        raise FileNotFoundError(f"cycle {cycle.analysis}: missing {cycle.missing(('archm_21', 'archm_09'))}")
    return mean_background([read_mean_archive(p, grid) for p in parts])
