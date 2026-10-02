"""Pack a prepare_b00 zarr into the folder training reads from RAM.

    out/state.npy    (time, grid_index, state_feature)   float32
    out/forcing.npy  (time, grid_index, forcing_feature) float32
    out/meta.zarr    everything else: coordinates, statistics, static fields, boundary mask

train_b00.sh copies the folder to /dev/shm and the datastore memory-maps the two arrays.

Run `python -m hycom_emulator.pack_b00 <b00.zarr> <out_dir>`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from hycom_emulator.datastore import PACKED


def pack(src: Path, out: Path) -> None:
    ds = xr.open_zarr(src, consolidated=True)
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in PACKED:
        da = ds[name].transpose("time", "grid_index", f"{name}_feature")
        arr = np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=np.float32, shape=da.shape)
        for i in range(da.sizes["time"]):
            arr[i] = da.isel(time=i).values
        arr.flush()
        del arr
    meta = ds.drop_vars(list(PACKED))
    for v in meta.variables.values():
        v.encoding.clear()
    meta.to_zarr(tmp / "meta.zarr", mode="w", consolidated=True)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    pack(a.src, a.out)


if __name__ == "__main__":
    main()
