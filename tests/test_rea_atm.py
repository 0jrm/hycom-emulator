import numpy as np
import pytest
import xarray as xr

from hycom_emulator.rea_atm import WINDOWS, Bilinear, daily, data_vars, hourly_path


def test_bilinear_is_exact_on_a_linear_field_and_wraps_longitude():
    lat, lon = np.linspace(18, 32, 8), np.linspace(260, 284, 13)
    f = 2 * lat[:, None] + 0.5 * lon[None]
    plat, plon = np.array([18.0, 25.3, 31.9]), np.array([-100.0 + 0.1, -85.2, -76.5])
    got = Bilinear.make(lat, lon, plat, plon)(f[None])[0]
    assert np.allclose(got, 2 * plat + 0.5 * (plon + 360))


def test_bilinear_refuses_points_outside_the_grid():
    with pytest.raises(ValueError, match="outside"):
        Bilinear.make(np.linspace(20, 30, 5), np.linspace(-90, -80, 5), np.array([19.0]), np.array([-85.0]))


def _hourly(tmp_path, hours: np.ndarray, values: np.ndarray):
    t = np.datetime64("2011-01-01T00:00") + hours.astype("timedelta64[h]")
    mt = (t - np.datetime64("1900-12-31T00:00")) / np.timedelta64(1, "D")
    ds = xr.Dataset({"airtmp": (("MT", "Latitude", "Longitude"), values[:, None, None] * np.ones((1, 2, 3), np.float32))},
                    coords={"MT": ("MT", mt, {"units": "days since 1900-12-31 00:00:00"}), "Latitude": [20.0, 21.0],
                            "Longitude": [-90.0, -89.0, -88.0]})
    p = tmp_path / "x.nc"
    ds.to_netcdf(p)
    return p


def test_daily_windows_and_incomplete_days(tmp_path):
    hours = np.concatenate([np.arange(24), np.arange(24, 47)])  # day 1 complete, day 2 missing 23Z
    p = _hourly(tmp_path, hours, (hours % 24).astype(np.float32))
    days = np.array(["2011-01-01", "2011-01-02"], dtype="datetime64[D]")
    a, _ = daily(p, "airtmp", days, "a")
    b, _ = daily(p, "airtmp", days, "b")
    assert a[0, 0, 0] == pytest.approx(11.5)
    assert b[0, 0, 0] == pytest.approx((WINDOWS["b"] * np.arange(24)).sum() / WINDOWS["b"].sum())
    assert np.isnan(a[1]).all() and np.isnan(b[1]).all()


def test_data_vars_and_paths(tmp_path):
    p = _hourly(tmp_path, np.arange(24), np.zeros(24, np.float32))
    assert data_vars(xr.open_dataset(p)) == ["airtmp"]
    assert hourly_path(tmp_path, 2010, "Taqa").name == "cfsr_gom-sea_2010_01hr_Taqa.nc"
    assert hourly_path(tmp_path, 2011, "uv-10m").parts[-2:] == ("CFSv2_GOM", "cfsv2_gom-sec2_2011_01hr_uv-10m.nc")


def test_years_arg_takes_lists_and_ranges():
    from hycom_emulator.rea_atm import years_arg
    assert years_arg("2001,2010-2012") == [2001, 2010, 2011, 2012]
    assert years_arg("2000-2002") == [2000, 2001, 2002]


def test_pack_appends_the_atm_channels_and_their_statistics(tmp_path):
    import json
    import os

    from hycom_emulator import rea_atm
    from hycom_emulator.datastore import _open

    T, G = 4, 6
    base, atm, out = tmp_path / "base", tmp_path / "atm", tmp_path / "overlay"
    base.mkdir(), atm.mkdir()
    rng = np.random.default_rng(0)
    f0 = rng.normal(size=(T, G, 2)).astype(np.float32)
    a = rng.normal(size=(T, G, len(rea_atm.CHANNELS))).astype(np.float32)
    np.save(base / "forcing.npy", f0)
    np.save(base / "state.npy", rng.normal(size=(T, G, 3)).astype(np.float32))
    np.save(atm / "atm.npy", a)
    for name in ("written.npy",):
        np.save(base / name, np.ones(T, bool))
    np.savez(base / "grid.npz", x=np.arange(G))
    (base / "plan.json").write_text("{}")
    xr.Dataset({"forcing_mean": ("forcing_feature", [0.1, 0.2]), "forcing_std": ("forcing_feature", [1.0, 2.0]),
                "state_mean": ("state_feature", np.zeros(3))},
               coords={"forcing_feature": ["wnd_ewd", "wnd_nwd"], "forcing_feature_units": ("forcing_feature", ["m/s", "m/s"]),
                       "state_feature": ["a", "b", "c"], "time": np.arange(T).astype("datetime64[D]")}).to_zarr(base / "meta.zarr", consolidated=True)
    stats = {c: float(i) for i, c in enumerate(rea_atm.CHANNELS)}
    (atm / "meta.json").write_text(json.dumps({"channels": list(rea_atm.CHANNELS), "units": {c: "u" for c in rea_atm.CHANNELS},
                                               "train_mean": stats, "train_std": {c: v + 1 for c, v in stats.items()}}))
    rea_atm.pack(base, atm, out)
    os.symlink(base / "state.npy", out / "state.npy")
    ds = _open(out)
    assert ds.sizes["forcing_feature"] == 2 + len(rea_atm.CHANNELS)
    assert list(ds.forcing_feature.values) == ["wnd_ewd", "wnd_nwd", *rea_atm.CHANNELS]
    assert np.array_equal(ds["forcing"].values[..., :2], f0) and np.array_equal(ds["forcing"].values[..., 2:], a)
    assert np.allclose(ds.forcing_mean.values, [0.1, 0.2, *stats.values()])
    assert np.allclose(ds.forcing_std.values, [1.0, 2.0, *(v + 1 for v in stats.values())])
    assert ds.sizes["state_feature"] == 3
