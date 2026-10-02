"""Stack the pack_b00 folders of runs over the same dates into one ensemble pack.

    out/state.npy    (ensemble_member, time, grid_index, state_feature)   float32
    out/forcing.npy  (ensemble_member, time, grid_index, forcing_feature) float32
    out/meta.zarr    the members' shared coordinates, static fields and boundary mask, an
                     ensemble_member coordinate, and statistics pooled over all members

E2 trains B00 on the free run and its cycled twins as neural-lam ensemble members: one time axis,
one atmosphere, and no sample window that straddles two runs. The members must share times, grid,
features, static fields and boundary mask. The statistics are recomputed from the data the way
prepare_b00 computes them for one run (ocean points of train rows, changes within a run), so a
one-member stack reproduces that member's statistics. Averaging the members' stored statistics
would be wrong: a free run's increments are all zero, and their stored std is the placeholder 1.

Run `python -m hycom_emulator.stack_b00 <out_dir> --train-end YYYY-MM-DD <name>=<pack> ...`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from hycom_emulator.datastore import PACKED
from hycom_emulator.prepare_b00 import TrainStats

SHARED = ("time", "x", "y", "state_feature", "forcing_feature", "static_feature", "static", "boundary_mask")


def stack(members: dict[str, Path], out: Path, train_end: np.datetime64) -> None:
    metas = {n: xr.open_zarr(p / "meta.zarr", consolidated=True, chunks=None).load() for n, p in members.items()}
    first_name, first = next(iter(metas.items()))
    for name, meta in metas.items():
        for v in SHARED:
            if not first[v].equals(meta[v]):
                raise ValueError(f"{name}: {v} differs from {first_name}")

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = {n: {v: np.load(p / f"{v}.npy", mmap_mode="r") for v in PACKED} for n, p in members.items()}
    dst = {
        v: np.lib.format.open_memmap(tmp / f"{v}.npy", mode="w+", dtype=np.float32, shape=(len(members), *src[first_name][v].shape))
        for v in PACKED
    }
    ocean = first["static"].sel(static_feature="ocean").values.astype(bool)
    train = first["time"].values <= train_end
    stats = TrainStats()
    for i, arrays in enumerate(src.values()):
        stats.new_run()
        for n in range(train.size):
            for v in PACKED:
                dst[v][i, n] = arrays[v][n]
            if train[n]:
                stats.add(arrays["state"][n][ocean], arrays["forcing"][n][ocean])
            else:
                stats.new_run()
    for arr in dst.values():
        arr.flush()
    del dst

    pooled = stats.dataset(first["static"].values[ocean])
    meta = first.drop_vars(list(pooled.data_vars)).merge(pooled).assign_coords(ensemble_member=list(members))
    meta.attrs["train_end"] = str(train_end)
    for v in meta.variables.values():
        v.encoding.clear()
    meta.to_zarr(tmp / "meta.zarr", mode="w", consolidated=True)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--train-end", required=True, type=np.datetime64)
    p.add_argument("members", nargs="+", help="name=path of a pack_b00 folder, in member order")
    a = p.parse_args()
    members = {n: Path(path) for n, path in (m.split("=", 1) for m in a.members)}
    stack(members, a.out, a.train_end)


if __name__ == "__main__":
    main()
