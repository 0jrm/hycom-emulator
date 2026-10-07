"""rea_pack: GOMb0.04 reanalysis daily means -> pack folder. Tiny synthetic netCDF archive, no RCC data."""

import json
import shutil
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from hycom_emulator import rea_pack
from hycom_emulator.prepare_b00 import DIFF_STD_FLOOR
from hycom_emulator.rea_pack import Plan, build, gulf_mask

NC_FILL = np.float32(1.267651e30)
AXIS = np.array([0.0, 10.0, 50.0, 100.0, 200.0])
DEPTHS = (0.0, 10.0, 100.0, 200.0)  # 50 m left out: level_index must skip it
NY, NX = 12, 14
LON = -92 + 0.5 * np.arange(NX)
LAT = 20 + 0.5 * np.arange(NY) + 0.01 * np.arange(NY) ** 2  # uneven, like Mercator
DAYS = np.arange("2010-12-29", "2011-01-05", dtype="datetime64[D]")  # crosses a year directory
MISSING = np.datetime64("2011-01-01")
TRAIN_END = date(2011, 1, 2)
SHELF = (slice(6, 10), slice(10, 14))  # 60 m: 0, 10, 50 m real
SLOPE = (slice(2, 6), slice(10, 14))  # 150 m: down to 100 m real


def _bottom():
    d = np.full((NY, NX), 3000.0, np.float32)
    d[SHELF], d[SLOPE] = 60.0, 150.0
    d[10:, :] = 0.0
    d[:, :2] = 0.0
    return d


def _source(t):
    """Native fields of day t: {nc name: array}, fills applied."""
    rng = np.random.default_rng(t)
    bottom = _bottom()
    ocean = bottom > 0
    real = AXIS[:, None, None] < bottom[None]
    shape3, shape2 = (AXIS.size, NY, NX), (NY, NX)
    f = {
        "water_temp": 25 - 0.01 * AXIS[:, None, None] + rng.normal(0, 0.5, shape3),
        "salinity": 36 + rng.normal(0, 0.1, shape3),
        "u": rng.normal(0, 0.2, shape3),
        "v": rng.normal(0, 0.2, shape3),
        "w_velocity": rng.normal(0, 1e-4, shape3),
    }
    f = {k: np.where(real, v, NC_FILL).astype(np.float32) for k, v in f.items()}
    for k in ("ssh", "mixed_layer_thickness", "u_barotropic_velocity", "v_barotropic_velocity"):
        f[k] = np.where(ocean, rng.normal(0, 0.3, shape2), NC_FILL).astype(np.float32)
    for k in ("wnd_ewd", "wnd_nwd"):
        f[k] = rng.normal(0, 5, shape2).astype(np.float32)
    f["wnd_ewd"][4, 6] = np.nan
    return f


def _write_nc(path, fields, names, three_d):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("MT", 1)
        nc.createDimension("Latitude", NY)
        nc.createDimension("Longitude", NX)
        nc.createVariable("Latitude", "f8", ("Latitude",))[:] = LAT
        nc.createVariable("Longitude", "f8", ("Longitude",))[:] = LON
        dims = ("MT", "Latitude", "Longitude")
        if three_d:
            nc.createDimension("Depth", AXIS.size)
            nc.createVariable("Depth", "f8", ("Depth",))[:] = AXIS
            dims = ("MT", "Depth", "Latitude", "Longitude")
        for n in names:
            v = nc.createVariable(n, "f4", dims, fill_value=NC_FILL)
            v.set_auto_maskandscale(False)
            ok = fields[n][np.isfinite(fields[n]) & (np.abs(fields[n]) < 1e29)]
            v.valid_range = np.array([ok.min(), ok.max()], np.float32)
            v[:] = fields[n][None]


def make_archive(root: Path, missing=(MISSING,)):
    source = {}
    for t, day in enumerate(DAYS):
        if day in missing:
            continue
        source[day] = f = _source(t)
        p3, p2 = rea_pack.day_paths(root, day)
        _write_nc(p3, f, ["u", "v", "w_velocity", "water_temp", "salinity"], True)
        _write_nc(p2, f, ["ssh", "mixed_layer_thickness", "u_barotropic_velocity", "v_barotropic_velocity", "wnd_ewd", "wnd_nwd"], False)
    topo = np.where(_bottom() > 0, _bottom(), np.float32(2.0**100)).reshape(-1)
    record = np.concatenate([topo, np.zeros(-topo.size % 4096, np.float32)])
    (root / "regional.depth.a").write_bytes(record.astype(">f4").tobytes())
    return source


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("rea")
    return root, make_archive(root)


def plan_for(root, **kw):
    args = dict(root=str(root), start=date(2010, 12, 29), end=date(2011, 1, 4), train_end=TRAIN_END,
                depths=DEPTHS, band=2, topo=str(root / "regional.depth.a"))
    return Plan(**{**args, **kw})


@pytest.fixture(scope="module")
def pack(archive, tmp_path_factory):
    root, source = archive
    out = tmp_path_factory.mktemp("pack") / "p"
    build(plan_for(root), out, workers=1)
    return out, source


def _arrays(out):
    return np.load(out / "state.npy"), np.load(out / "forcing.npy")


def _meta(out):
    return xr.open_zarr(out / "meta.zarr", consolidated=True).load()


def test_layout(pack):
    out, source = pack
    state, forcing = _arrays(out)
    meta = _meta(out)
    names = meta.state_feature.values.tolist()
    assert names[:5] == ["temp_0m", "temp_10m", "temp_100m", "temp_200m", "salin_0m"]
    assert names[-3:] == ["ssh", "ubaro", "vbaro"] and len(names) == 4 * len(DEPTHS) + 3
    assert meta.forcing_feature.values.tolist() == ["wnd_ewd", "wnd_nwd", "sin_doy", "cos_doy", "insolation"]
    assert meta.static_feature.values.tolist() == ["depth", "lon", "lat", "coriolis", "ocean", "gulf"]
    assert state.shape == (DAYS.size, NX * NY, len(names)) and state.dtype == np.float32
    assert forcing.shape == (DAYS.size, NX * NY, 5)
    units = dict(zip(names, meta.state_feature_units.values.tolist()))
    assert (units["temp_200m"], units["salin_0m"], units["v_10m"], units["ssh"]) == ("degC", "psu", "m/s", "m")
    assert meta.forcing_feature_units.values.tolist() == ["m/s", "m/s", "1", "1", "W/m2"]
    assert meta.static_feature_units.values[-1] == "1"

    j, i = 3, 7
    gi = i * NY + j  # x outer, y inner
    assert meta.x.values[gi] == pytest.approx(LON[i]) and meta.y.values[gi] == pytest.approx(LAT[j])
    assert state[0, gi, names.index("temp_100m")] == source[DAYS[0]]["water_temp"][3, j, i]
    assert state[1, gi, names.index("vbaro")] == source[DAYS[1]]["v_barotropic_velocity"][j, i]
    assert forcing[2, gi, 1] == source[DAYS[2]]["wnd_nwd"][j, i]
    assert meta.time.values[0] == np.datetime64("2010-12-29T12:00")
    static = meta.static.sel(static_feature="depth").values.reshape(NX, NY).T
    np.testing.assert_array_equal(static, _bottom())


def test_points_that_are_not_real_hold_the_start_day_mean(pack):
    out, source = pack
    state, forcing = _arrays(out)
    meta = _meta(out)
    names = meta.state_feature.values.tolist()
    assert np.isfinite(state).all() and np.isfinite(forcing).all()
    assert np.abs(state).max() < 1e3 and np.abs(forcing).max() < 1e4
    grid = lambda a: a.reshape(a.shape[0], NX, NY, -1).transpose(0, 2, 1, 3)  # (time, y, x, feature)
    s = grid(state)
    src = source[DAYS[0]]
    bottom = _bottom()
    for v, nc in (("temp", "water_temp"), ("u", "u"), ("ssh", "ssh")):
        for depth in (DEPTHS if v != "ssh" else (None,)):
            name, real = (v, bottom > 0) if depth is None else (f"{v}_{depth:g}m", bottom > depth)
            k = list(AXIS).index(depth) if depth is not None else None
            field = src[nc] if k is None else src[nc][k]
            fill = np.float32(field[real].mean(dtype=np.float64))
            c = names.index(name)
            assert (s[:, ~real, c] == pytest.approx(fill, rel=1e-6)), name
            assert (s[:, real, c] != fill).all(), name
    shelf_j, shelf_i = 7, 11
    assert s[0, shelf_j, shelf_i, names.index("temp_10m")] == src["water_temp"][1, shelf_j, shelf_i]
    slope_j, slope_i = 3, 12
    assert s[0, slope_j, slope_i, names.index("temp_100m")] == src["water_temp"][3, slope_j, slope_i]
    assert grid(forcing)[1, 4, 6, 0] == 0  # the NaN wind


def _pack_snapshot(out):
    state, forcing = _arrays(out)
    return state, forcing, _meta(out)


def test_resume_reads_only_unmarked_rows(archive, tmp_path, monkeypatch):
    root, _ = archive
    out = tmp_path / "p"
    plan = plan_for(root)
    build(plan, out, workers=1)
    state0, forcing0, meta0 = _pack_snapshot(out)

    written = np.load(out / "written.npy")
    written[4] = False
    np.save(out / "written.npy", written)
    st = np.load(out / "state.npy", mmap_mode="r+")
    st[4] = 1e30
    st.flush()
    del st
    shutil.rmtree(out / "meta.zarr")

    calls = []
    real = rea_pack.read_day
    monkeypatch.setattr(rea_pack, "read_day", lambda root, day, *a: calls.append(day) or real(root, day, *a))
    build(plan, out, workers=1)
    assert calls == [DAYS[4]]
    state1, forcing1, meta1 = _pack_snapshot(out)
    np.testing.assert_array_equal(state1, state0)
    np.testing.assert_array_equal(forcing1, forcing0)
    xr.testing.assert_identical(meta1, meta0)

    calls.clear()
    build(plan, out, workers=1)
    assert calls == []


def test_another_plan_in_the_folder_raises(archive, tmp_path):
    root, _ = archive
    out = tmp_path / "p"
    build(plan_for(root), out, workers=1)
    with pytest.raises(ValueError, match="another plan.*stride"):
        build(plan_for(root, stride=2), out, workers=1)
    assert json.loads((out / "plan.json").read_text())["stride"] == 1


def test_missing_day_is_interpolated_and_flagged(pack, archive):
    out, _ = pack
    state, forcing = _arrays(out)
    meta = _meta(out)
    n = int(np.flatnonzero(DAYS == MISSING)[0])
    assert meta.time_filled.values.tolist() == [d == MISSING for d in DAYS]
    np.testing.assert_allclose(state[n], 0.5 * (state[n - 1] + state[n + 1]), rtol=1e-6)
    np.testing.assert_allclose(forcing[n, :, :2], 0.5 * (forcing[n - 1, :, :2] + forcing[n + 1, :, :2]), rtol=1e-6)
    cal = rea_pack.calendar(MISSING, LAT, NX)
    np.testing.assert_allclose(forcing[n, :, 2:], cal.transpose(2, 1, 0).reshape(-1, 3), rtol=1e-6)


def test_a_long_gap_raises_before_writing(tmp_path):
    root = tmp_path / "rea"
    make_archive(root, missing=(np.datetime64("2010-12-31"), MISSING))
    out = tmp_path / "p"
    with pytest.raises(ValueError, match="gaps longer than 1 days: 2010-12-31..2011-01-01"):
        build(plan_for(root, max_gap=1), out, workers=1)
    assert not out.exists()


def test_statistics_match_numpy(pack):
    out, _ = pack
    state, forcing = _arrays(out)
    meta = _meta(out)
    names = meta.state_feature.values.tolist()
    level_ocean = meta.level_ocean.values.astype(bool)
    ocean = meta.static.sel(static_feature="ocean").values.astype(bool)
    assert meta.level.values.tolist() == list(DEPTHS)
    assert level_ocean[:, 0].sum() == ocean.sum() > level_ocean[:, 2].sum() > level_ocean[:, 3].sum()
    use = [0, 1, 2, 4]  # train rows, the missing day excluded
    pairs = [(0, 1), (1, 2)]  # 2 -> 4 straddles the filled day
    for c, name in enumerate(names):
        var, _, depth = name.rpartition("_")
        mask = level_ocean[:, DEPTHS.index(float(depth[:-1]))] if var else ocean
        x = state[use][:, mask, c].astype(np.float64)
        d = np.concatenate([state[b][mask, c] - state[a][mask, c] for a, b in pairs]).astype(np.float64)
        assert float(meta.state_mean[c]) == pytest.approx(x.mean(), rel=1e-5, abs=1e-6), name
        assert float(meta.state_std[c]) == pytest.approx(x.std(), rel=1e-5), name
        assert float(meta.state_diff_mean[c]) == pytest.approx(d.mean(), rel=1e-4, abs=1e-6), name
        assert float(meta.state_diff_std[c]) == pytest.approx(max(d.std(), DIFF_STD_FLOOR * x.std()), rel=1e-5), name
    f = forcing[use][:, ocean].astype(np.float64)
    np.testing.assert_allclose(meta.forcing_mean, f.mean((0, 1)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(meta.forcing_std, np.where(f.std((0, 1)) > 0, f.std((0, 1)), 1), rtol=1e-5)
    static = meta.static.values[ocean].astype(np.float64)
    np.testing.assert_allclose(meta.static_mean, static.mean(0), rtol=1e-5)
    assert meta.attrs["train_end"] == "2011-01-02" and meta.attrs["stride"] == 1


def test_boundary_and_gulf_statics(pack):
    out, _ = pack
    meta = _meta(out)
    bm = meta.boundary_mask.values.reshape(NX, NY).T.astype(bool)
    ocean = _bottom() > 0
    assert bm[~ocean].all()
    assert bm[0, 5] and bm[3, 5] and not bm[4, 5]  # open south edge rows 0-1, then band 2
    gulf = meta.static.sel(static_feature="gulf").values.reshape(NX, NY).T.astype(bool)
    assert not gulf[~ocean].any() and gulf[9, 4]


def test_gulf_mask_cuts_at_sections():
    n = 20
    lon2d, lat2d = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    ocean = np.zeros((n, n), bool)
    ocean[2:18, 2:10] = True  # west basin
    ocean[2:18, 12:18] = True  # east basin
    ocean[9:11, 10:12] = True  # the channel
    section = (((10.5, 7.0), (10.5, 12.0)),)
    gulf = gulf_mask(ocean, lon2d, lat2d, section, (5.0, 5.0))
    assert gulf[5, 5] and gulf[15, 3] and not gulf[5, 15] and not gulf[9, 10]
    assert gulf_mask(ocean, lon2d, lat2d, (), (5.0, 5.0))[5, 15]
    with pytest.raises(ValueError, match="seed"):
        gulf_mask(ocean, lon2d, lat2d, section, (0.0, 0.0))


def test_two_workers_and_stride(archive, tmp_path):
    root, source = archive
    plan = plan_for(root, stride=2)
    build(plan, tmp_path / "w2", workers=2)
    build(plan, tmp_path / "w1", workers=1)
    a, b = _arrays(tmp_path / "w2"), _arrays(tmp_path / "w1")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    ny = NY // 2
    j, i = 4, 8
    gi = (i // 2) * ny + j // 2
    meta = _meta(tmp_path / "w2")
    assert a[0].shape[1] == ny * (NX // 2)
    assert meta.x.values[gi] == pytest.approx(LON[i]) and meta.y.values[gi] == pytest.approx(LAT[j])
    assert a[0][4, gi, 1] == source[DAYS[4]]["water_temp"][1, j, i]


def test_datastore_opens_the_pack(pack):
    pytest.importorskip("neural_lam")
    from hycom_emulator.datastore import HycomDatastore

    out, _ = pack
    cfg = out.parent / "rea.yaml"
    cfg.write_text(
        f"zarr: {out}\nsplits:\n  train: [2010-12-29, 2011-01-02]\n  val: [2011-01-03, 2011-01-04]\n"
        "  test: [2011-01-03, 2011-01-04]\n"
    )
    ds = HycomDatastore(cfg)
    st = ds.get_dataarray("state", "train")
    assert st.shape == (5, NX * NY, 4 * len(DEPTHS) + 3)
    assert ds.step_length.days == 1 and ds.step_length.seconds == 0
    assert ds.grid_shape_state.x == NX and ds.grid_shape_state.y == NY
    assert not np.isnan(ds.get_dataarray("forcing", "val", standardize=True).values).any()
