"""neural-lam datastore over a zarr written by prepare_b00.

The config is a small YAML next to nothing in particular:

    zarr: /path/to/b00.zarr     # a prepare_b00 zarr, or a pack_b00 folder
    splits:
      train: [2025-03-04, 2025-07-31]
      val:   [2025-08-06, 2025-08-15]
      test:  [2025-08-21, 2025-09-01]

Importing this module registers the datastore kind `hycom` with neural-lam.
Training reads a pack_b00 folder staged in /dev/shm: state and forcing are memory-mapped .npy,
so the DataLoader workers (forked, see nlam.py) share one copy of the pages and nothing is
decoded per sample. A zarr opens lazily (no dask, so forking stays safe); fine for tests and scoring.
A stack_b00 folder has a leading ensemble_member axis: neural-lam then draws each sample from one
member, so every member is a separate run over the same dates.
"""

from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import yaml
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore, CartesianGridShape

CATEGORIES = ("state", "forcing", "static")


class HycomDatastore(BaseRegularGridDatastore):
    SHORT_NAME = "hycom"

    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._config = yaml.safe_load(self._config_path.read_text())
        self._ds = _open(Path(self._config["zarr"]))
        self.is_ensemble = self.has_ensemble_forcing = "ensemble_member" in self._ds["state"].dims
        if "physics" in self._config:  # hycom_wmse reads its settings from here
            from hycom_emulator.physics import CONTEXT

            CONTEXT.configure(self)
        for split in ("train", "val", "test"):
            if split not in self._config["splits"]:
                raise ValueError(f"{config_path}: missing split {split}")

    @property
    def root_path(self) -> Path:
        return self._config_path.parent

    @property
    def config(self) -> dict:
        return self._config

    @property
    def step_length(self) -> timedelta:
        dt = np.unique(np.diff(self._ds["time"].values))
        if dt.size != 1:
            raise ValueError(f"time steps are not uniform: {dt}")
        return timedelta(seconds=int(dt[0] / np.timedelta64(1, "s")))

    def get_vars_units(self, category: str) -> list[str]:
        return self._ds[f"{category}_feature_units"].values.tolist()

    def get_vars_names(self, category: str) -> list[str]:
        return self._ds[f"{category}_feature"].values.tolist()

    def get_vars_long_names(self, category: str) -> list[str]:
        return self.get_vars_names(category)

    def get_num_data_vars(self, category: str) -> int:
        return self._ds.sizes[f"{category}_feature"]

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        ds = xr.Dataset({f"{category}_{op}": self._ds[f"{category}_{op}"] for op in ("mean", "std")})
        if category == "state":
            for op in ("mean", "std"):
                ds[f"state_diff_{op}_standardized"] = self._ds[f"state_diff_{op}"] / self._ds["state_std"]
        return ds

    def get_dataarray(self, category: str, split: str | None, standardize: bool = False) -> xr.DataArray | None:
        if category not in CATEGORIES:
            raise ValueError(category)
        da = self._ds[category].set_index(grid_index=self.spatial_coordinates)
        if "time" in da.dims:
            start, end = self._config["splits"][split]
            da = da.sel(time=slice(str(start), str(end)))  # whole days, end inclusive
        da = da.transpose(*self.expected_dim_order(category=category))
        return self._standardize_datarray(da, category=category) if standardize else da

    @cached_property
    def boundary_mask(self) -> xr.DataArray:
        return self._ds["boundary_mask"].set_index(grid_index=self.spatial_coordinates)

    @property
    def coords_projection(self) -> ccrs.Projection:
        return ccrs.PlateCarree()

    @cached_property
    def grid_shape_state(self) -> CartesianGridShape:
        return CartesianGridShape(x=np.unique(self._ds["x"].values).size, y=np.unique(self._ds["y"].values).size)

    def get_xy(self, category: str, stacked: bool) -> np.ndarray:
        xy = np.stack([self._ds["x"].values, self._ds["y"].values], axis=1)
        if stacked:
            return xy
        shape = self.grid_shape_state
        return xy.reshape(shape.x, shape.y, 2)

    def state_feature_weights_values(self) -> list[float]:
        return [1.0] * self.get_num_data_vars("state")


def _open(path: Path) -> xr.Dataset:
    if not (path / "meta.zarr").is_dir():
        return xr.open_zarr(path, consolidated=True, chunks=None)  # lazy without dask: safe to fork
    ds = xr.open_zarr(path / "meta.zarr", consolidated=True, chunks=None).load()
    for name in PACKED:
        arr = np.load(path / f"{name}.npy", mmap_mode="r")
        dims = ("time", "grid_index", f"{name}_feature")
        ds[name] = (("ensemble_member", *dims) if arr.ndim == 4 else dims, arr)
    return ds


PACKED = ("state", "forcing")

DATASTORES[HycomDatastore.SHORT_NAME] = HycomDatastore
