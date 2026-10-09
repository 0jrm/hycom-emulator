import numpy as np
import xarray as xr

from hycom_emulator.evaluate_rea import (
    accumulate,
    front_distances,
    front_grid,
    in_period,
    level_weights,
    point_weights,
    summarize,
    summarize_fronts,
)
from hycom_emulator.isotach import EARTH_RADIUS_KM
from hycom_emulator.prepare_b00 import _to_grid_index

LEVELS = np.array([0.0, 10.0, 100.0])
NAMES = [f"{v}_{d:g}m" for v in ("temp", "salin", "u", "v") for d in LEVELS] + ["ssh", "ubaro", "vbaro"]


def _setup(rng, n=50):
    level_ocean = np.ones((n, 3), bool)
    level_ocean[:10, 2] = False
    region = np.ones(n, bool)
    region[-5:] = False
    area = rng.uniform(0.5, 1.0, n)
    return point_weights(NAMES, area, region, level_ocean, LEVELS), area, region, level_ocean


def test_level_weights_cover_the_column():
    dz = level_weights(LEVELS, bottom=200.0)
    assert np.allclose(dz, [5.0, 50.0, 145.0]) and np.isclose(dz.sum(), 200.0)


def test_point_weights_drop_unreal_levels_and_points_outside_the_region():
    w, area, region, level_ocean = _setup(np.random.default_rng(0))
    j = NAMES.index("temp_100m")
    assert (w[:10, j] == 0).all() and (w[-5:, j] == 0).all() and np.allclose(w[10:-5, j], area[10:-5])
    assert np.allclose(w[:-5, NAMES.index("ssh")], area[:-5])


def test_a_perfect_forecast_scores_zero_and_correlates_one():
    rng = np.random.default_rng(1)
    w, *_ = _setup(rng)
    x0 = rng.normal(size=(50, len(NAMES)))
    truth = x0[None] + rng.normal(size=(2, 50, len(NAMES)))
    acc: dict = {}
    accumulate(acc, x0, truth, truth.copy(), w)
    s = summarize(acc, NAMES, level_weights(LEVELS))
    for name in ("temp_0m", "temp", "ssh"):
        assert s[name]["1"]["rmse_model"] < 1e-12 and np.isclose(s[name]["2"]["corr_change"], 1.0)
        assert s[name]["1"]["rmse_persistence"] > 0.5


def test_scores_match_a_direct_weighted_computation():
    rng = np.random.default_rng(2)
    w, *_ = _setup(rng)
    acc: dict = {}
    samples = [(rng.normal(size=(50, len(NAMES))), rng.normal(size=(2, 50, len(NAMES))), rng.normal(size=(2, 50, len(NAMES)))) for _ in range(3)]
    for x0, truth, pred in samples:
        accumulate(acc, x0, truth, pred, w)
    s = summarize(acc, NAMES, level_weights(LEVELS))
    j = NAMES.index("salin_10m")
    e = np.concatenate([p[1, :, j] - t[1, :, j] for _, t, p in samples])
    ww = np.concatenate([w[:, j]] * 3)
    assert np.isclose(s["salin_10m"]["2"]["rmse_model"], np.sqrt(np.sum(ww * e**2) / ww.sum()))
    assert np.isclose(s["salin_10m"]["2"]["bias_model"], np.sum(ww * e) / ww.sum())
    dz = level_weights(LEVELS)
    js = [NAMES.index(f"u_{d:g}m") for d in LEVELS]
    e_col = np.concatenate([np.concatenate([p[0, :, k] - t[0, :, k] for _, t, p in samples]) for k in js])
    w_col = np.concatenate([np.concatenate([w[:, k] * dz[i]] * 3) for i, k in enumerate(js)])
    assert np.isclose(s["u"]["1"]["rmse_model"], np.sqrt(np.sum(w_col * e_col**2) / w_col.sum()))


def test_in_period_needs_the_initial_and_every_target_day_inside():
    days = lambda first, n: (np.datetime64(first) + np.arange(n) * np.timedelta64(1, "D")).astype("datetime64[ns]").astype(np.int64)  # noqa: E731
    p = ("2024-04-02", "2024-08-31")
    assert in_period(days("2024-04-04", 4), p)
    assert not in_period(days("2024-04-03", 4), p), "initial day 2024-04-01 is outside"
    assert in_period(days("2024-08-28", 4), p) and not in_period(days("2024-08-29", 4), p)
    assert in_period(days("2020-01-01", 4), None)


STRIDE2_DLON = 0.08


def _stride2_meta(ny=30, nx=40):
    """A pack-shaped meta.zarr on a stride-2 Mercator grid: square 0.08 deg cells from 21.0N, a land frame, deep inside."""
    lon = -84.0 + STRIDE2_DLON * np.arange(nx)
    lat = np.empty(ny)
    lat[0] = 21.0
    for j in range(1, ny):
        lat[j] = lat[j - 1] + STRIDE2_DLON * np.cos(np.deg2rad(lat[j - 1]))
    lon2d, lat2d = np.meshgrid(lon, lat)
    depth = np.full((ny, nx), 1000.0)
    depth[[0, -1]], depth[:, [0, -1]] = 0.0, 0.0
    static = np.stack([depth, lon2d, lat2d, np.zeros_like(depth), depth > 0, depth > 0])
    gx, gy = np.meshgrid(lon, lat, indexing="ij")
    meta = xr.Dataset(
        {"static": (("grid_index", "static_feature"), _to_grid_index(static))},
        coords={"x": ("grid_index", gx.reshape(-1)), "y": ("grid_index", gy.reshape(-1)),
                "static_feature": ["depth", "lon", "lat", "coriolis", "ocean", "gulf"]},
    )
    return meta, lon2d, lat2d, depth


def _state(fast_cols, ny, nx):
    """(grid, feature) with surface speed 0.6*sqrt(2) = 0.85 m/s west of fast_cols and 0 east: neither component alone reaches 1.5 kt."""
    uv = np.zeros((ny, nx))
    uv[:, :fast_cols] = 0.6
    x = np.zeros((len(NAMES), ny, nx))
    x[NAMES.index("u_0m")] = x[NAMES.index("v_0m")] = uv
    return _to_grid_index(x)


def test_a_front_shifted_three_cells_on_the_stride2_grid_is_three_stride2_cells_away():
    ny, nx, shift = 30, 40, 3
    meta, lon2d, lat2d, depth = _stride2_meta(ny, nx)
    fg = front_grid(meta, NAMES)
    assert fg.shape == (ny, nx)
    assert np.array_equal(fg.region, (depth > 500) & (lon2d <= -81.0) & (lat2d >= 21.5))
    truth = _state(16, ny, nx)
    (d,) = front_distances(fg, truth[None], _state(16 + shift, ny, nx)[None])
    rows = fg.region[:, 15]
    km = shift * EARTH_RADIUS_KM * np.deg2rad(STRIDE2_DLON) * np.cos(np.deg2rad(lat2d[rows, 0]))
    assert d.status == "ok"
    assert np.isclose(d.hausdorff_km, km.max(), rtol=1e-4), "the southernmost region row has the widest cells"
    assert np.isclose(d.mean_km, km.mean(), rtol=1e-4)
    assert 24.0 < d.mean_km < 25.0, "3 stride-2 cells of 8.1-8.3 km, not 3 native cells of ~4.1 km"


def test_persistence_scores_one_state_at_every_lead_and_counts_empty_fronts():
    ny, nx = 30, 40
    meta, *_ = _stride2_meta(ny, nx)
    fg = front_grid(meta, NAMES)
    truth = np.stack([_state(16, ny, nx), _state(18, ny, nx)])
    persist = front_distances(fg, truth, _state(16, ny, nx))
    assert persist[0].mean_km == 0.0 and persist[1].mean_km > 0.0
    slow = front_distances(fg, truth, np.zeros_like(truth))
    s = summarize_fronts([persist, slow])
    assert s["1"]["mean_km"] == 0.0 and s["1"]["status"] == {"ok": 1, "true_empty": 0, "pred_empty": 1, "both_empty": 0}
    assert np.isclose(s["2"]["hausdorff_km"], persist[1].hausdorff_km)
