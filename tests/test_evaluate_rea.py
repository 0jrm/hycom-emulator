import numpy as np

from hycom_emulator.evaluate_rea import accumulate, in_period, level_weights, point_weights, summarize

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
