from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from hycom_emulator.background import cycle_background, grid_of
from hycom_emulator.build_store import build
from hycom_emulator.catalog import cycles
from hycom_emulator.system import SystemConfig
from hycom_emulator.tsis import read_layer_obs

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"

pytestmark = pytest.mark.skipif(
    not Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4").is_dir(),
    reason="needs RCC /gpfs/research/coaps/abozec",
)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    cfg = SystemConfig.from_toml(CONFIG)
    out = tmp_path_factory.mktemp("store") / "s.zarr"
    build(cfg, out, date(2025, 6, 10), date(2025, 6, 11))
    return cfg, out


def test_two_cycles_with_statics(store):
    import xarray as xr

    cfg, out = store
    with xr.open_zarr(out, consolidated=False) as ds:
        assert list(ds["cycle"].values) == [np.datetime64("2025-06-10T18"), np.datetime64("2025-06-11T18")]
        assert ds["xb_temp"].shape == (2, 41, 385, 525) and ds["xb_temp"].dtype == np.float32
        ocean = ds["ocean"].values
        assert 0.3 < ocean.mean() < 0.9
        assert np.isnan(ds["xb_temp"][0, 0].values[~ocean]).all()
        assert np.isfinite(ds["xb_temp"][0, 0].values[ocean]).all()
        assert ds.attrs["system"] == cfg.name and len(ds.attrs["structural"]) == 64


def test_store_round_trips_background_and_obs(store):
    import xarray as xr

    cfg, out = store
    cycle = next(c for c in cycles(cfg) if c.analysis == datetime(2025, 6, 11, 18))
    xb = cycle_background(cycle, grid_of(cfg))
    with xr.open_zarr(out, consolidated=False) as ds, xr.open_zarr(out, group="obs/2025061118", consolidated=False) as obs:
        got = ds["xb_salin"].sel(cycle="2025-06-11T18").values
        want = np.where(np.isfinite(got), xb.layer["salin"], np.nan).astype(np.float32)
        np.testing.assert_array_equal(got, want)
        assert obs.sizes["row"] == read_layer_obs(cycle.files["tsis_obs"]).val.size
    with xr.open_zarr(out, group="obs/2025061018", consolidated=False) as first:
        assert first.sizes["row"] > 0


def test_rerun_writes_nothing(store):
    cfg, out = store
    log = build(cfg, out, date(2025, 6, 10), date(2025, 6, 11))
    assert all(line.endswith("already written") for line in log)
