import numpy as np

from hycom_emulator.conservation import CP, RHO0, apply, columns, functionals, offset_pattern, per_point, summarize
from hycom_emulator.evaluate_rea import level_weights

LEVELS = np.array([0.0, 10.0, 100.0, 2000.0])
NAMES = [f"{v}_{d:g}m" for v in ("temp", "salin", "u", "v") for d in LEVELS] + ["ssh", "ubaro", "vbaro"]
N = 60


def _setup(rng):
    level_ocean = np.ones((N, LEVELS.size), bool)
    level_ocean[:20, 3] = False
    level_ocean[:5, 2] = False
    region = np.ones(N, bool)
    region[-8:] = False
    area = rng.uniform(0.5, 1.0, N)
    fs = functionals(NAMES, area, region, level_ocean, LEVELS)
    return {f.name: f for f in fs}, fs, area, region, level_ocean


def _col(var):
    return [NAMES.index(f"{var}_{d:g}m") for d in LEVELS]


def test_a_uniform_offset_shifts_each_mean_by_itself_and_the_heat_content_by_rho_cp_times_mean_depth():
    rng = np.random.default_rng(0)
    by_name, fs, area, region, level_ocean = _setup(rng)
    x = rng.normal(size=(N, len(NAMES)))
    y = x.copy()
    y[:, _col("temp")] += 0.3
    y[:, NAMES.index("ssh")] += 0.02
    d = apply(fs, y) - apply(fs, x)
    q = [f.name for f in fs]
    assert np.isclose(d[q.index("temp_mean")], 0.3) and np.isclose(d[q.index("ssh_mean")], 0.02)
    assert np.allclose(d[[q.index(n) for n in ("salin_mean", "u_mean", "vbaro_mean", "salt_content")]], 0.0)
    a = area * region
    depth = (level_ocean * level_weights(LEVELS)).sum(1)
    assert np.isclose(d[q.index("heat_content")], RHO0 * CP * 0.3 * np.sum(a * depth) / a.sum())


def test_points_outside_the_region_and_below_the_floor_do_not_count():
    rng = np.random.default_rng(1)
    _, fs, _, region, level_ocean = _setup(rng)
    x = rng.normal(size=(N, len(NAMES)))
    y = x.copy()
    y[~region] += 1e6
    for k, j in enumerate(_col("salin")):
        y[~level_ocean[:, k], j] += 1e6
    assert np.allclose(apply(fs, y), apply(fs, x))


def test_quantities_match_a_direct_computation_and_broadcast_over_leading_axes():
    rng = np.random.default_rng(2)
    by_name, fs, area, region, level_ocean = _setup(rng)
    x = rng.normal(size=(3, 2, N, len(NAMES)))
    v = apply(fs, x)
    assert v.shape == (3, 2, len(fs))
    s = x[1, 0][:, _col("salin")]
    vol = (area * region)[:, None] * level_weights(LEVELS) * level_ocean
    q = [f.name for f in fs]
    assert np.isclose(v[1, 0, q.index("salin_mean")], np.sum(vol * s) / vol.sum())
    assert np.isclose(v[1, 0, q.index("salt_content")], RHO0 * 1e-3 * np.sum(vol * s) / np.sum(area * region))
    assert np.isclose(v[1, 0, q.index("ubaro_mean")], np.sum(area * region * x[1, 0, :, NAMES.index("ubaro")]) / np.sum(area * region))


def test_offset_and_pattern_add_in_quadrature_to_the_rmse():
    rng = np.random.default_rng(3)
    w = rng.uniform(0.0, 1.0, N)
    e = rng.normal(size=(4, N)) + 0.5
    offset, pattern = offset_pattern(e, w)
    rmse = np.sqrt((w * e**2).sum(-1) / w.sum())
    assert np.allclose(offset**2 + pattern**2, rmse**2) and np.allclose(offset, (w * e).sum(-1) / w.sum())
    offset, pattern = offset_pattern(np.full((2, N), 0.7), w)
    assert np.allclose(offset, 0.7) and np.allclose(pattern, 0.0)


def test_summary_reports_error_and_one_step_changes_per_lead():
    true = np.array([[[0.0], [1.0], [3.0]], [[0.0], [2.0], [2.0]]])
    pred = true + np.array([[[0.0], [0.5], [1.0]], [[0.0], [0.5], [2.0]]])
    s = summarize(true, pred, [type("F", (), {"name": "q", "units": "m"})()])["q"]["leads"]
    assert s["1"]["error_mean"] == 0.5 and s["2"]["error_mean"] == 1.5 and s["2"]["error_std"] == 0.5
    assert s["1"]["truth_step"] == 1.5 and s["2"]["truth_step"] == 1.0
    assert s["1"]["model_step"] == 2.0 and s["2"]["model_step"] == 2.0 and s["2"]["positive_fraction"] == 1.0


def test_column_contents_integrate_real_levels_and_their_area_mean_is_the_content_quantity():
    rng = np.random.default_rng(4)
    by_name, fs, area, region, level_ocean = _setup(rng)
    x = rng.normal(size=(N, len(NAMES)))
    col = per_point(columns(NAMES, level_ocean, LEVELS), x)
    dz = level_weights(LEVELS) * level_ocean
    assert col.shape == (N, 2)
    assert np.allclose(col[:, 0], RHO0 * CP * (x[:, _col("temp")] * dz).sum(1))
    a = area * region
    q = [f.name for f in fs]
    assert np.isclose((a * col[:, 1]).sum() / a.sum(), apply(fs, x)[q.index("salt_content")])
