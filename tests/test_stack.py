"""stack_b00: runs over the same dates become neural-lam ensemble members. Synthetic packs, no RCC data."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hycom_emulator.prepare_b00 import DIFF_STD_FLOOR, TrainStats
from hycom_emulator.stack_b00 import stack

NX, NY, T = 4, 3, 10
STATE = ["temp_k01", "salin_k01", "thknss_k01", "u_k01", "v_k01", "srfhgt", "montg1", "ubaro", "vbaro"]
FORCING = ["inc_temp_k01", "inc_salin_k01", "inc_thknss_k01", "atm_airtmp"]
TIMES = np.arange("2025-06-01", "2025-06-11", dtype="datetime64[D]").astype("datetime64[ns]")
TRAIN_END = np.datetime64("2025-06-06")
SPLITS = "splits:\n  train: [2025-06-01, 2025-06-06]\n  val: [2025-06-05, 2025-06-09]\n  test: [2025-06-05, 2025-06-10]\n"


def _static():
    x, y = np.meshgrid(-90 + 0.04 * np.arange(NX), 25 + 0.04 * np.arange(NY), indexing="ij")
    ocean = np.ones(NX * NY, np.float32)
    ocean[0] = 0
    lat = y.reshape(-1)
    static = np.stack([1000 * ocean, x.reshape(-1), lat, 1e-4 * np.sin(np.deg2rad(lat)), ocean], axis=1).astype(np.float32)
    boundary = (1 - ocean).astype(np.int8)
    boundary[-1] = 1
    return x.reshape(-1), lat, static, boundary


def _run(rng, free: bool, atm: np.ndarray):
    state = rng.normal(size=(T, NX * NY, len(STATE))).astype(np.float32)
    state[:, :, 2] = 50.0 + np.arange(NX * NY)  # a fixed layer thickness: its change std needs the floor
    forcing = np.concatenate([np.zeros((T, NX * NY, 3)) if free else rng.normal(size=(T, NX * NY, 3)), atm], axis=2)
    return state, forcing.astype(np.float32)


def write_pack(path: Path, state: np.ndarray, forcing: np.ndarray, boundary=None) -> Path:
    x, y, static, default_boundary = _static()
    ocean = static[:, 4].astype(bool)
    stats = TrainStats()
    for n in np.flatnonzero(TIMES <= TRAIN_END):
        stats.add(state[n][ocean], forcing[n][ocean])
    path.mkdir()
    np.save(path / "state.npy", state)
    np.save(path / "forcing.npy", forcing)
    meta = stats.dataset(static[ocean]).assign(
        static=(("grid_index", "static_feature"), static),
        boundary_mask=(("grid_index",), default_boundary if boundary is None else boundary),
    )
    meta = meta.assign_coords(
        time=TIMES,
        x=(("grid_index",), x),
        y=(("grid_index",), y),
        state_feature=STATE,
        forcing_feature=FORCING,
        static_feature=["depth", "lon", "lat", "coriolis", "ocean"],
        state_feature_units=(("state_feature",), ["1"] * len(STATE)),
        forcing_feature_units=(("forcing_feature",), ["1"] * len(FORCING)),
        static_feature_units=(("static_feature",), ["1"] * 5),
    )
    meta.to_zarr(path / "meta.zarr", mode="w", consolidated=True)
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("stack")
    rng = np.random.default_rng(0)
    atm = rng.normal(size=(T, NX * NY, 1))
    arrays = {"free": _run(rng, True, atm), "twin": _run(rng, False, atm)}
    packs = {n: write_pack(root / n, *a) for n, a in arrays.items()}
    stack(packs, root / "ens", TRAIN_END)
    (root / "b00.yaml").write_text(f"zarr: {root / 'ens'}\n{SPLITS}")
    (root / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    return root, arrays


def test_statistics_pool_all_runs(runs):
    root, arrays = runs
    meta = xr.open_zarr(root / "ens" / "meta.zarr")
    ocean = _static()[2][:, 4].astype(bool)
    train = TIMES <= TRAIN_END
    st = np.concatenate([s[train][:, ocean] for s, _ in arrays.values()]).astype(np.float64)
    fo = np.concatenate([f[train][:, ocean] for _, f in arrays.values()]).astype(np.float64)
    diff = np.concatenate([np.diff(s[train][:, ocean].astype(np.float64), axis=0) for s, _ in arrays.values()])
    st, fo, diff = (a.reshape(-1, a.shape[-1]) for a in (st, fo, diff))
    np.testing.assert_allclose(meta.state_mean, st.mean(0), rtol=1e-5)
    np.testing.assert_allclose(meta.state_std, st.std(0), rtol=1e-5)
    np.testing.assert_allclose(meta.forcing_std, fo.std(0), rtol=1e-5)  # not the free run's placeholder 1
    np.testing.assert_allclose(meta.state_diff_std, np.maximum(diff.std(0), DIFF_STD_FLOOR * st.std(0)), rtol=1e-5, atol=1e-7)
    assert float(meta.state_diff_std[2]) == pytest.approx(DIFF_STD_FLOOR * float(meta.state_std[2]))
    assert meta.ensemble_member.values.tolist() == ["free", "twin"]


def test_samples_come_from_one_member(runs):
    from neural_lam.weather_dataset import WeatherDataset

    from hycom_emulator.datastore import HycomDatastore

    root, arrays = runs
    ds = HycomDatastore(root / "b00.yaml")
    assert ds.is_ensemble and ds.has_ensemble_forcing
    assert ds.get_dataarray("state", "train").dims == ("time", "ensemble_member", "grid_index", "state_feature")
    data = WeatherDataset(ds, split="train", ar_steps=1, num_past_forcing_steps=1, num_future_forcing_steps=1)
    assert len(data) == 2 * (6 - 4 + 1)
    for idx in range(len(data)):
        sample, member = divmod(idx, 2)
        state, forcing = arrays[["free", "twin"][member]]
        init, target, forc, _ = data[idx]
        np.testing.assert_array_equal(init.numpy(), state[sample : sample + 2])
        np.testing.assert_array_equal(target.numpy(), state[sample + 2 : sample + 3])
        np.testing.assert_array_equal(forc.numpy()[0, :, 0::3], forcing[sample + 1])  # (feature, window) stacked, window inner


def test_scores_each_member(runs):
    from hycom_emulator.evaluate_b00 import evaluate

    root, _ = runs
    res = evaluate(root / "nlam.yaml", None, "test", ar_steps=2)
    free, twin = res["members"]["free"], res["members"]["twin"]
    assert free["samples"] == twin["samples"] == 6 - 5 + 1
    t = free["scores"]["temp"]["+24h"]
    assert t["rmse_persistence_inc"] == t["rmse_persistence"]  # zero increments
    assert twin["scores"]["temp"]["+24h"]["rmse_persistence_inc"] != twin["scores"]["temp"]["+24h"]["rmse_persistence"]


def test_members_must_share_the_grid(tmp_path):
    rng = np.random.default_rng(1)
    atm = rng.normal(size=(T, NX * NY, 1))
    a = write_pack(tmp_path / "a", *_run(rng, True, atm))
    other = _static()[3].copy()
    other[5] = 1
    b = write_pack(tmp_path / "b", *_run(rng, False, atm), boundary=other)
    with pytest.raises(ValueError, match="boundary_mask"):
        stack({"a": a, "b": b}, tmp_path / "ens", TRAIN_END)
