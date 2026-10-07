"""neural-lam datastore over a zarr written by prepare_b00.

The config is a small YAML next to nothing in particular:

    zarr: /path/to/b00.zarr     # a prepare_b00 zarr, or a pack_b00 folder
    splits:
      train: [2025-03-04, 2025-07-31]
      val:   [2025-08-06, 2025-08-15]
      test:  [2025-08-21, 2025-09-01]
    exclude_source_changes: true   # optional, default false

Importing this module registers the datastore kind `hycom` with neural-lam.
Training reads a pack_b00 folder staged in /dev/shm: state and forcing are memory-mapped .npy,
so the DataLoader workers (forked, see nlam.py) share one copy of the pages and nothing is
decoded per sample. A zarr opens lazily (no dask, so forking stays safe); fine for tests and scoring.
A stack_b00 folder has a leading ensemble_member axis: neural-lam then draws each sample from one
member, so every member is a separate run over the same dates.

`exclude_source_changes` drops the train samples whose state rows straddle a change of source experiment
(SOURCE_CHANGES): a jump between two experiments is not dynamics. nlam.py applies it to the train split only.
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
# Days whose reanalysis row comes from another source than the day before (docs/rea-pipeline.md).
SOURCE_CHANGES = np.array(
    ["2017-06-01", "2017-06-02", "2021-01-01", "2024-01-01", "2024-01-02", "2024-01-06", "2024-02-01", "2024-02-02", "2024-04-02"],
    dtype="datetime64[D]",
)


def kept_windows(times: np.ndarray, n_rows: int, n_samples: int, changes: np.ndarray = SOURCE_CHANGES) -> np.ndarray:
    """Indices i < n_samples of the windows times[i]..times[i + n_rows - 1] that hold no change day c with
    first < c <= last, so no two rows of a window come from different sources. Compared as whole days."""
    days = np.asarray(times).astype("datetime64[D]")
    first, last = days[:n_samples, None], days[n_rows - 1 : n_rows - 1 + n_samples, None]
    c = np.asarray(changes, dtype="datetime64[D]")[None]
    return np.flatnonzero(~((first < c) & (c <= last)).any(1))


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
    def exclude_source_changes(self) -> bool:
        return bool(self._config.get("exclude_source_changes", False))

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
