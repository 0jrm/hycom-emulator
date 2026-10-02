"""Atmospheric forcing blocks and their use in B00. Runs on a compute node."""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from hycom_emulator.background import grid_of
from hycom_emulator.build_store import build as build_store
from hycom_emulator.forcing import HourlyField, build
from hycom_emulator.system import SystemConfig

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"
RMU = Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4/data/relax.rmu.a")

pytestmark = pytest.mark.skipif(not RMU.is_file(), reason="needs RCC /gpfs/research/coaps/abozec")


@pytest.fixture(scope="module")
def cfg():
    return SystemConfig.from_toml(CONFIG)


@pytest.fixture(scope="module")
def atm(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("atm") / "atm.zarr"
    build(cfg, out, datetime(2025, 6, 2, 0), datetime(2025, 6, 4, 18))
    return out


def test_every_field_is_hourly_through_the_period(cfg):
    g = grid_of(cfg)
    for name, pattern in cfg.forcing_files.items():
        f = HourlyField(str(cfg.expt_dir / pattern), g.idm, g.jdm)
        t = datetime(2025, 3, 2)
        missing = [t + timedelta(hours=h) for h in range(24 * 183) if t + timedelta(hours=h) not in f.where]
        assert missing == [], f"{name}: {len(missing)} hours missing, first {missing[:3]}"


def test_block_is_the_mean_of_six_hours(cfg, atm):
    import xarray as xr

    g = grid_of(cfg)
    f = HourlyField(str(cfg.expt_dir / cfg.forcing_files["airtmp"]), g.idm, g.jdm)
    start = datetime(2025, 6, 3, 12)
    want = np.mean([f.at(start + timedelta(hours=h)) for h in range(6)], axis=0)
    with xr.open_zarr(atm, consolidated=False) as ds:
        assert ds.sizes["time"] == 12
        got = ds["atm_airtmp"].sel(time=np.datetime64(start)).values
        np.testing.assert_allclose(got, want, rtol=1e-6)
        assert 5 < float(np.nanmin(got)) and float(np.nanmax(got)) < 40
        spd = ds["atm_wndspd"].values
        mean_wind = np.hypot(ds["atm_wndewd"].values, ds["atm_wndnwd"].values)
        assert (spd >= mean_wind - 1e-4).all()


def test_b00_forcing_carries_the_24h_atmosphere(cfg, atm, tmp_path):
    import xarray as xr

    from hycom_emulator.prepare_b00 import prepare

    store = tmp_path / "s.zarr"
    build_store(cfg, store, date(2025, 6, 1), date(2025, 6, 5))
    prepare(store, tmp_path / "b00.zarr", RMU, np.datetime64("2025-06-04"), stride=16, atm=atm)
    with xr.open_zarr(tmp_path / "b00.zarr", consolidated=True) as b, xr.open_zarr(atm, consolidated=False) as a:
        names = list(b["forcing_feature"].values)
        assert len(names) == 41 * 3 + 11 and "atm_wndspd" in names
        t = np.datetime64("2025-06-04T00")
        blocks = [t - np.timedelta64(h, "h") for h in (24, 18, 12, 6)]
        want = a["atm_airtmp"].sel(time=blocks).values.mean(axis=0)[::16, ::16].T.reshape(-1)
        got = b["forcing"].sel(time=t, forcing_feature="atm_airtmp").values
        np.testing.assert_allclose(got, want, rtol=1e-6)
