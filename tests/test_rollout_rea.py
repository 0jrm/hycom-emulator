from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import ndimage

from hycom_emulator.evaluate_rea import level_weights, point_weights
from hycom_emulator.rollout_rea import (
    BANDS, STATS, Start, Step, band_variance, done, finalize, geometry, loop_current, max_horizon, plane, rollout_stats,
    start_set, to_dataset, unroll, window_index, write_part,
)

LEVELS = np.array([0.0, 10.0, 100.0])
NAMES = [f"{v}_{d:g}m" for v in ("temp", "salin", "u", "v") for d in LEVELS] + ["ssh", "ubaro", "vbaro"]
G, F = 20, len(NAMES)


def _geo(rng):
    level_ocean = np.ones((G, 3), bool)
    level_ocean[:4, 2] = False
    region = np.ones(G, bool)
    region[-3:] = False
    area = rng.uniform(0.5, 1.0, G)
    pl = plane((5, 4), np.linspace(-90, -85, G), np.linspace(22, 26, G), np.full(G, 1000.0), np.ones(G, bool),
               {"ssh": region, "temp_100m": region & level_ocean[:, 2]}, 2)
    return replace(geometry(NAMES, area, region, level_ocean, LEVELS), plane=pl), area, region, level_ocean


def _step(rng, **over):
    t = {k: torch.tensor(rng.normal(size=(2, G, F)), dtype=torch.float32) for k in ("pred", "truth", "x0", "pred_prev", "truth_prev")}
    return Step(**(t | over))


def test_start_set_dates_horizons_flags_stride_limit():
    dates = np.arange(np.datetime64("2001-01-16"), np.datetime64("2024-09-01"), np.timedelta64(1, "D"))
    s = start_set(dates, 365)
    oos = [x for x in s if x.date >= np.datetime64("2022-01-01")]
    ins = [x for x in s if x.date < np.datetime64("2022-01-01")]
    assert ins[0].date == np.datetime64("2001-02-01") and len(ins) == len(range(0, 7609, 30)) and all(x.horizon == 365 for x in ins)
    assert min(x.date for x in oos) == np.datetime64("2022-01-01") and max(x.date for x in oos) == np.datetime64("2024-08-29")
    last = next(x for x in s if x.date == np.datetime64("2024-08-29"))
    assert last.horizon == 1 and last.index == len(dates) - 3 and s[-1] == last
    assert [x.horizon for x in s] == sorted((x.horizon for x in s), reverse=True)
    assert all(dates[x.index] == x.date for x in s)
    strided = start_set(dates, 365, stride=7)
    assert len([x for x in strided if x.date >= np.datetime64("2022-01-01")]) == len(range(0, 973, 7))
    assert len([x for x in strided if x.date < np.datetime64("2022-01-01")]) == len(ins)
    assert start_set(dates, 365, limit=3) == s[:3]
    explicit = start_set(dates, 10, ["2024-08-25", "2022-03-01", "2024-08-30"])
    assert [(str(x.date), x.horizon) for x in explicit] == [("2022-03-01", 10), ("2024-08-25", 5)]
    with pytest.raises(ValueError):
        start_set(dates, 10, ["2000-01-01"])


def test_window_index_math():
    T = 40
    assert max_horizon(5, T) == 33 and max_horizon(0, T) == 0
    s, chunk, n = 5, 10, max_horizon(5, T)
    assert window_index(s, 0, chunk) == 4 and window_index(s, 3, chunk) == 34
    # WeatherDataset(ar_steps=n): idx needs state rows idx..idx+1+n and forcing up to row idx+n+2
    assert window_index(s, 0, chunk) + n + 2 == T - 1


class FakeWindow:
    """WeatherDataset slicing (num_past = num_future = 1) over a synthetic record."""

    def __init__(self, state, forcing, times, n):
        self.state, self.forcing, self.times, self.n = state, forcing, times, n

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self.state) - self.n - 3
        rows = self.state[idx:idx + 2 + self.n]
        forcing = torch.stack([self.forcing[idx + 1 + k:idx + 4 + k].permute(1, 2, 0).reshape(G, -1) for k in range(self.n)])
        return rows[:2], rows[2:], forcing, torch.tensor(self.times[idx + 2:idx + 2 + self.n].astype(np.int64))


class FakeModule:
    """Standardizes like ForecasterModule and unrolls like ARForecaster: boundary from the target each step."""

    device = torch.device("cpu")

    def __init__(self, rng):
        self.state_mean = torch.tensor(rng.normal(size=F), dtype=torch.float32)
        self.state_std = torch.tensor(rng.uniform(0.5, 2.0, F), dtype=torch.float32)
        self.boundary = torch.zeros(G, 1)
        self.boundary[:2] = 1

    def on_after_batch_transfer(self, batch, _):
        init, target, forcing, times = batch
        return (init - self.state_mean) / self.state_std, (target - self.state_mean) / self.state_std, forcing, times

    def common_step(self, batch):
        init, target, forcing, times = batch
        pp, p, out = init[:, 0], init[:, 1], []
        for i in range(forcing.shape[1]):
            new = 0.6 * p + 0.3 * pp + 0.1 * forcing[:, i, :, :1]
            new = self.boundary * target[:, i] + (1 - self.boundary) * new
            out.append(new)
            pp, p = p, new
        return torch.stack(out, 1), target, None, times


@pytest.fixture
def record():
    rng = np.random.default_rng(0)
    T = 30
    times = np.arange(np.datetime64("2022-01-01T12"), np.datetime64("2022-01-31T12"), np.timedelta64(1, "D")).astype("datetime64[ns]")
    state = torch.tensor(rng.normal(size=(T, G, F)), dtype=torch.float32)
    forcing = torch.tensor(rng.normal(size=(T, G, 2)), dtype=torch.float32)
    return FakeModule(rng), lambda n: FakeWindow(state, forcing, times, n), times, state


def _collect(module, window, times, starts, chunk):
    out = {}
    for lead, pos, step in unroll(module, window, times, starts, chunk):
        for i, b in enumerate(pos):
            out[b, lead] = (step.pred[i].clone(), step.truth[i].clone(), step.pred_prev[i].clone(), step.x0[i].clone())
    return out


def test_chunked_unroll_matches_one_window_and_padding_changes_nothing(record):
    module, window, times, state = record
    starts = start_set(times.astype("datetime64[D]"), 20, ["2022-01-03", "2022-01-20", "2022-01-11"])
    assert [(str(s.date), s.horizon) for s in starts] == [("2022-01-03", 20), ("2022-01-11", 18), ("2022-01-20", 9)]
    single = _collect(module, window, times, starts, 20)
    chunked = _collect(module, window, times, starts, 4)
    alone = {(b, lead): v for b, s in enumerate(starts) for (_, lead), v in _collect(module, window, times, [s], 4).items()}
    assert set(single) == set(chunked) == set(alone) == {(b, lead) for b, s in enumerate(starts) for lead in range(1, s.horizon + 1)}
    for k in single:
        for a, b, c in zip(single[k], chunked[k], alone[k]):
            assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)
    for b, s in enumerate(starts):
        pred, truth, prev, x0 = single[b, 1]
        assert torch.equal(truth, state[s.index + 1]) and torch.equal(x0, state[s.index]) and torch.allclose(prev, x0)
        assert torch.allclose(single[b, 2][2], pred)
        assert torch.allclose(pred[:2], truth[:2], atol=1e-5)
    s = starts[0]
    sp = (state[s.index - 1] - module.state_mean) / module.state_std
    p = (state[s.index] - module.state_mean) / module.state_std
    new = (0.6 * p + 0.3 * sp + 0.1 * window(1)[s.index - 1][2][0, :, :1]) * module.state_std + module.state_mean
    assert torch.allclose(single[0, 1][0][2:], new[2:], atol=1e-5)


def test_rollout_stats_are_nan_beyond_the_horizon(record):
    module, window, times, _ = record
    starts = start_set(times.astype("datetime64[D]"), 20, ["2022-01-03", "2022-01-20"])
    geo, *_ = _geo(np.random.default_rng(1))
    v = rollout_stats(module, window, times, starts, 4, geo)
    assert v.shape == (2, 20, len(STATS))
    always = [i for i, k in enumerate(STATS) if not (k.startswith("lc_north") or k.endswith("km_ratio"))]  # toy grid: no LC, bands empty
    assert np.isfinite(v[0][:, always]).all() and np.isfinite(v[1, :starts[1].horizon][:, always]).all()
    assert np.isnan(v[1, starts[1].horizon:]).all()


def _image_plane(lon1, lat1, stride=2, masks=None):
    """Plane over a lon x lat grid, every point Gulf and 1000 m deep; returns it and (grid,) <- (ny, nx) flattening."""
    lat2, lon2 = np.meshgrid(lat1, lon1, indexing="ij")
    flat = lambda img: img.T.reshape(-1)  # noqa: E731
    n = lat2.size
    masks = masks or {"ssh": np.ones(n, bool)}
    return plane((len(lon1), len(lat1)), flat(lon2), flat(lat2), np.full(n, 1000.0), np.ones(n, bool), masks, stride), flat, lat2, lon2


def test_band_variance_puts_a_plane_wave_in_its_band_and_sums_to_the_tapered_variance():
    pl, flat, lat2, lon2 = _image_plane(np.linspace(-90, -80, 128), np.zeros(96) + np.linspace(0, 1e-6, 96))
    dx = 0.04 * 2 * 111.32
    x = np.arange(128) * dx
    for wavelength, band in ((75.0, "50to100km"), (30.0, "lt50km"), (400.0, "gt200km")):
        img = np.cos(2 * np.pi * x / wavelength)[None].repeat(96, 0)
        v = band_variance(torch.tensor(flat(img))[None], pl, "ssh")[0]
        w = pl.taper["ssh"].numpy()
        a = (img - (img * w).sum() / w.sum()) * w
        assert np.isclose(v.sum().item(), (a**2).sum() / (w**2).sum(), rtol=1e-6)
        assert v[list(BANDS).index(band)] > 0.9 * v.sum(), (wavelength, v)


def test_band_variance_ratio_falls_at_small_scales_for_a_blurred_field():
    pl, flat, *_ = _image_plane(np.linspace(-90, -80, 128), np.linspace(20, 20.001, 128))
    truth = np.random.default_rng(5).normal(size=(128, 128))
    blur = ndimage.gaussian_filter(truth, 3)
    r = (band_variance(torch.tensor(flat(blur))[None], pl, "ssh") / band_variance(torch.tensor(flat(truth))[None], pl, "ssh"))[0]
    assert r[0] < 0.1 and r[0] < r[1] < r[2] < r[3]


def test_loop_current_extent_eddies_and_a_shifted_disk():
    lon1, lat1 = np.arange(-98, -79.95, 0.1), np.arange(18, 32.01, 0.1)
    pl, flat, lat2, lon2 = _image_plane(lon1, lat1)
    disk = lambda la, lo, r: np.hypot(lat2 - la, lon2 - lo) <= r  # noqa: E731
    ssh = 0.6 * disk(23.0, -86.0, 2.0) + 0.6 * disk(26.0, -93.0, 1.0) + 0.6 * disk(29.0, -88.0, 0.12)
    north, n_lce, area = loop_current(ssh, pl)
    assert np.isclose(north, lat2[disk(23.0, -86.0, 2.0)].max()) and n_lce == 1
    assert np.isclose(area, pl.cell_km2[disk(23.0, -86.0, 2.0)].sum())
    shifted = 0.6 * disk(24.0, -86.0, 2.0) + 0.6 * disk(26.0, -93.0, 1.0)
    assert np.isclose(loop_current(shifted, pl)[0] - north, 1.0, atol=0.051)
    no_lc = 0.6 * disk(26.0, -93.0, 1.0)
    assert np.isnan(loop_current(no_lc, pl)[0]) and loop_current(no_lc, pl)[1] == 1
    s = torch.tensor(np.stack([flat(ssh), flat(shifted)]), dtype=torch.float64)
    x = torch.zeros(2, s.shape[1], 1, dtype=torch.float64)
    names = ["ssh"]
    geo = type("G", (), {"names": names, "plane": pl})()
    step = Step(s[[1, 1]][..., None], s[..., None], x, x, x)
    err = STATS["lc_north_error"].fn(step, geo)
    assert np.isclose(err[0].item(), loop_current(shifted, pl)[0] - north) and err[1].item() == 0.0


def test_rmse_bias_match_a_masked_column_computation():
    rng = np.random.default_rng(2)
    geo, area, region, level_ocean = _geo(rng)
    s = _step(rng)
    pw = point_weights(NAMES, area, region, level_ocean, LEVELS)
    dz = level_weights(LEVELS)
    for b in range(2):
        cols = [NAMES.index(f"salin_{d:g}m") for d in LEVELS]
        w = pw[:, cols] * dz
        e = s.pred[b, :, cols].double().numpy() - s.truth[b, :, cols].double().numpy()
        ep = s.x0[b, :, cols].double().numpy() - s.truth[b, :, cols].double().numpy()
        assert np.isclose(STATS["rmse_salin"].fn(s, geo)[b], np.sqrt((w * e**2).sum() / w.sum()))
        assert np.isclose(STATS["bias_salin"].fn(s, geo)[b], (w * e).sum() / w.sum())
        assert np.isclose(STATS["rmse_pers_salin"].fn(s, geo)[b], np.sqrt((w * ep**2).sum() / w.sum()))
        assert np.isclose(STATS["bias_pers_salin"].fn(s, geo)[b], (w * ep).sum() / w.sum())
        j = NAMES.index("ssh")
        e = (s.pred[b, :, j] - s.truth[b, :, j]).double().numpy()
        assert np.isclose(STATS["rmse_ssh"].fn(s, geo)[b], np.sqrt((area * region * e**2).sum() / (area * region).sum()))
    off = s.truth.clone()
    off[:, :4, NAMES.index("temp_100m")] = 1e6
    off[:, -3:] = 1e6
    assert torch.equal(STATS["rmse_temp"].fn(Step(off, s.truth, s.x0, s.pred_prev, s.truth_prev), geo), torch.zeros(2, dtype=torch.float64))


def test_corr_change_is_one_for_identical_and_minus_one_for_negated_changes():
    rng = np.random.default_rng(3)
    geo, *_ = _geo(rng)
    s = _step(rng)
    change = s.truth - s.truth_prev
    same = Step(s.pred_prev + change, s.truth, s.x0, s.pred_prev, s.truth_prev)
    flip = Step(s.pred_prev - change, s.truth, s.x0, s.pred_prev, s.truth_prev)
    for f in ("temp", "u", "ssh"):
        assert torch.allclose(STATS[f"corr_change_{f}"].fn(same, geo), torch.ones(2, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(STATS[f"corr_change_{f}"].fn(flip, geo), -torch.ones(2, dtype=torch.float64), atol=1e-6)
    still = Step(s.pred_prev.clone(), s.truth, s.x0, s.pred_prev, s.truth_prev)
    assert torch.isnan(STATS["corr_change_v"].fn(still, geo)).all()


def test_ke_maxspeed_and_gulf_means():
    rng = np.random.default_rng(4)
    geo, area, region, level_ocean = _geo(rng)
    s = _step(rng)
    pw = point_weights(NAMES, area, region, level_ocean, LEVELS)
    dz = level_weights(LEVELS)
    x = s.pred[0].double().numpy()
    ke = sum(pw[:, NAMES.index(f"u_{d:g}m")] * dz[k] * 0.5 * (x[:, NAMES.index(f"u_{d:g}m")] ** 2 + x[:, NAMES.index(f"v_{d:g}m")] ** 2)
             for k, d in enumerate(LEVELS)).sum() / (area * region).sum()
    assert np.isclose(STATS["ke_model"].fn(s, geo)[0], ke)
    s.truth[1, -1, NAMES.index("u_0m")] = 50.0
    speed = np.hypot(s.truth[1, :, NAMES.index("u_0m")].numpy(), s.truth[1, :, NAMES.index("v_0m")].numpy())
    assert np.isclose(STATS["maxspeed_truth"].fn(s, geo)[1], speed[region].max()) and speed.max() > speed[region].max()
    j = NAMES.index("temp_100m")
    w = area * region * level_ocean[:, 2]
    assert np.isclose(STATS["mean_temp_100m_truth"].fn(s, geo)[1], (w * s.truth[1, :, j].double().numpy()).sum() / w.sum())
    assert np.isclose(STATS["mean_ssh_model"].fn(s, geo)[0], (area * region * x[:, NAMES.index("ssh")]).sum() / (area * region).sum())


def _part(dates, horizons, H, md5="abc"):
    starts = [Start(np.datetime64(d), 10, h) for d, h in zip(dates, horizons)]
    v = np.full((len(starts), H, len(STATS)), np.nan)
    for i, h in enumerate(horizons):
        v[i, :h] = i + 1.0
    return to_dataset(starts, v, {"checkpoint_md5": md5, "label": "x"})


def test_parts_are_skipped_on_rerun_and_concatenated_in_order(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(parts, _part(["2022-03-01", "2001-02-01"], [5, 3], 5))
    write_part(parts, _part(["2022-01-01"], [7], 7))
    assert sorted(p.name for p in parts.iterdir()) == ["2022-01-01_1.nc", "2022-03-01_2.nc"]
    have = done(parts)
    assert have == {np.datetime64("2022-03-01"): (5, "abc"), np.datetime64("2001-02-01"): (3, "abc"), np.datetime64("2022-01-01"): (7, "abc")}
    ds = finalize(parts, tmp_path / "stats.nc")
    assert [str(t)[:10] for t in ds.start.values] == ["2001-02-01", "2022-01-01", "2022-03-01"]
    assert ds.sizes["lead"] == 7 and list(ds.in_sample.values) == [True, False, False] and list(ds.horizon.values) == [3, 7, 5]
    r = ds.rmse_temp.values
    assert np.isfinite(r[0, :3]).all() and np.isnan(r[0, 3:]).all() and np.isnan(r[2, 5:]).all() and np.isfinite(r[1]).all()
    assert ds.valid.values[2, 6] == np.datetime64("2022-03-08")
    assert ds.rmse_temp.attrs["units"] == "degC" and ds.attrs["label"] == "x"


def test_loads_ensemble_checkpoints():
    from neural_lam.models import MODELS

    assert "crps_graph_lam" in MODELS
