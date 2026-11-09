"""hycom_emulator.topology (vendored persistence-ocean engine; its tests ported) and hycom_emulator.structure."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import ndimage

from hycom_emulator import structure as st
from hycom_emulator import topology as tp


# ------------------------------------------------------------------- synthetic fields (persistence-ocean's)

def gaussian_bump(shape, center, amplitude, width):
    rows, cols = np.indices(shape)
    return amplitude * np.exp(-((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2.0 * width**2))


def ridge(shape, row, first, last, value):
    r = np.zeros(shape)
    r[row, first:last + 1] = value
    return r


def two_bumps(shape=(41, 81), ridge_value=0.4):
    row = shape[0] // 2
    values = np.maximum.reduce([gaussian_bump(shape, (row, 20), 1.0, 2.5), gaussian_bump(shape, (row, 60), 0.8, 2.5),
                                ridge(shape, row, 20, 60, ridge_value)])
    return values, {"tall": (row, 20), "short": (row, 60)}


def nested_bumps(shape=(41, 121), outer=0.3, inner=0.5):
    row = shape[0] // 2
    values = np.maximum.reduce([gaussian_bump(shape, (row, 20), 1.0, 2.5), gaussian_bump(shape, (row, 60), 0.8, 2.5),
                                gaussian_bump(shape, (row, 100), 0.65, 2.5), ridge(shape, row, 20, 60, outer),
                                ridge(shape, row, 60, 100, inner)])
    return values, {"tall": (row, 20), "middle": (row, 60), "small": (row, 100)}


def random_smooth_field(shape, seed, passes=2):
    v = np.random.default_rng(seed).random(shape)
    for _ in range(passes):
        v = (v + np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)) / 5.0
    return v


def field_with_land(shape, seed):
    v = random_smooth_field(shape, seed)
    v[np.random.default_rng(seed + 1000).random(shape) < 0.12] = np.nan
    return v


# ------------------------------------------------------------------------------------- engine (ported tests)

def test_two_bumps_short_dies_at_the_ridge_value():
    values, _ = two_bumps()
    pairs, _ = tp.compute_superlevel_persistence(values)
    finite = np.isfinite(pairs.death)
    assert pairs.birth[finite] == pytest.approx([0.8]) and pairs.death[finite] == pytest.approx([0.4])
    assert pairs.birth[~finite] == pytest.approx([1.0])


def test_nested_bumps_middle_absorbs_small_then_dies_into_tall():
    values, _ = nested_bumps()
    pairs, _ = tp.compute_superlevel_persistence(values)
    got = sorted(zip(pairs.birth.round(6), pairs.death))
    assert got == [(0.65, 0.5), (0.8, 0.3), (1.0, -np.inf)]


def test_open_boundary_kills_a_bump_at_its_highest_edge_value():
    values = gaussian_bump((41, 41), (20, 2), 1.0, 3.0)
    pairs, _ = tp.compute_superlevel_persistence(values, open_boundary=True)
    assert len(pairs.birth) == 1 and pairs.death[0] == pytest.approx(values[20, 0])


def connected(values, a, b, level, boundary):
    labels, _ = ndimage.label(values >= level)
    la, lb = labels[a], labels[b]
    if la == 0 or lb == 0:
        return False
    edge = set(np.unique(labels[boundary & (labels > 0)]))
    return la == lb or (la in edge and lb in edge)


def brute_level(values, a, b, boundary):
    for level in np.unique(values[~np.isnan(values)])[::-1]:
        if connected(values, a, b, level, boundary):
            return level
    return -np.inf


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("open_boundary", [False, True])
def test_cell_connection_level_equals_label_oracle(seed, open_boundary):
    values = field_with_land((14, 17), seed)
    ocean = ~np.isnan(values)
    boundary = tp.open_boundary_cells(ocean) if open_boundary else np.zeros_like(ocean)
    _, forest = tp.compute_superlevel_persistence(values, open_boundary=open_boundary)
    ranked = tp.cells_by_rank(forest)
    cells = np.argwhere(ocean)
    rng = np.random.default_rng(seed)
    for _ in range(25):
        a, b = (tuple(cells[k]) for k in rng.choice(len(cells), 2))
        rank = max(int(forest.insertion_rank[b]), tp.cell_connection_rank(forest, a, forest.owner[b]))
        assert tp.level_at_rank(values, ranked, rank) == brute_level(values, a, b, boundary)


def test_nested_bumps_connection_levels_are_the_ridges():
    values, maxima = nested_bumps()
    _, forest = tp.compute_superlevel_persistence(values)
    ranked = tp.cells_by_rank(forest)
    ids = {k: int(forest.owner[ij]) for k, ij in maxima.items()}
    level = lambda a, b: tp.level_at_rank(values, ranked, tp.connection_rank(forest, ids[a], ids[b]))  # noqa: E731
    assert level("small", "middle") == 0.5 and level("small", "tall") == 0.3 and level("middle", "tall") == 0.3
    assert level("tall", "tall") == np.inf


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_betti0_matches_thresholded_components(seed, connectivity):
    values = field_with_land((30, 45), seed)
    pairs, _ = tp.compute_superlevel_persistence(values, connectivity=connectivity)
    structure = np.ones((3, 3)) if connectivity == 8 else None
    finite = values[~np.isnan(values)]
    for t in np.quantile(finite, [0.05, 0.3, 0.5, 0.7, 0.9, 0.99]):
        n = ndimage.label(np.nan_to_num(values, nan=-np.inf) >= t, structure=structure)[1]
        assert n == ((pairs.birth >= t) & (pairs.death < t)).sum()


def test_cyclones_are_minima_in_h_units():
    shape = (20, 40)
    h = -np.maximum(gaussian_bump(shape, (10, 10), 0.3, 3), gaussian_bump(shape, (10, 30), 0.2, 3))
    p = tp.compute_eddy_persistence(h, connectivity=8, open_boundary=True)
    assert np.all(p["cyclone"].birth < p["cyclone"].death)
    assert sorted(p["cyclone"].persistence.round(2)) == [0.2, 0.3]
    d = tp.diagram(p["cyclone"], "cyclone", 0.05)
    assert (d[:, 0] > d[:, 1]).all() and len(d) == 2


# --------------------------------------------------------------------------------- Loop Current (ported tests)

def synthetic_gg(shape, seed_cells):
    seed = np.zeros(shape, bool)
    for ij in seed_cells:
        seed[ij] = True
    return tp.GulfGrid(np.linspace(25.0, 25.0 + 0.25 * (shape[0] - 1), shape[0]),
                       np.linspace(-90.0, -90.0 + 0.25 * (shape[1] - 1), shape[1]), np.ones(shape, bool), seed)


def test_gulf_region_cuts_caribbean_batabano_and_atlantic():
    region = tp.gulf_region(np.array([20.0, 22.125, 25.0, 27.0]), np.array([-90.0, -86.0, -83.0, -80.5]))
    assert region[2, 0] and region[2, 1] and region[3, 2] and region[0, 0]
    assert not region[0, 1] and not region[1, 2] and not region[:, 3].any()


def test_neck_saddle_of_the_short_bump_is_the_ridge_when_seeded_on_the_tall_bump():
    values, maxima = two_bumps()
    s = tp.neck_saddles(values, synthetic_gg(values.shape, [maxima["tall"]]))
    short = s[(s.max_i == maxima["short"][0]) & (s.max_j == maxima["short"][1])].iloc[0]
    assert short.h_saddle == 0.4 and short.neck_depth == pytest.approx(0.4)
    tall = s[s.max_j == maxima["tall"][1]].iloc[0]
    assert tall.saddle_at_seed and tall.neck_depth == 0.0


def test_neck_when_the_seed_side_is_lower():
    values, maxima = two_bumps()
    s = tp.neck_saddles(values, synthetic_gg(values.shape, [maxima["short"]]))
    tall = s[(s.max_i == maxima["tall"][0]) & (s.max_j == maxima["tall"][1])].iloc[0]
    assert tall.neck_depth == pytest.approx(0.8 - 0.4)


RULE = tp.TouchRule(min_depth_m=0.05, longer_than_days=3)
DEEP, SHALLOW = 0.3, 0.18


def pinching_blob(days_joined, days_apart, days_rejoined=0, join=DEEP, days_apart_after=0):
    shape = (5, 12)
    frames = []
    for n, neck in [(days_joined, join), (days_apart, 0.1), (days_rejoined, join), (days_apart_after, 0.1)]:
        for _ in range(n):
            f = np.zeros(shape)
            f[2, 0:4], f[2, 6:10], f[2, 4:6] = 0.5, 0.4, neck
            frames.append(f)
    dates = pd.date_range("2020-01-01", periods=len(frames))
    return tp.GulfAnomaly(dates, np.stack(frames), np.zeros(len(frames))), synthetic_gg(shape, [(2, 0)])


def test_piece_that_never_rejoins_separates_on_the_day_it_breaks_off():
    an, gg = pinching_blob(3, 40)
    d = tp.lc17_detachments(an, gg, RULE)
    assert len(d) == 1
    r = d.iloc[0]
    assert r.date == an.dates[3] and (r.piece_max_i, r.piece_max_j) == (2, 6)
    assert r.outcome == "separated" and r.fate == "record_end" and r.final_separation


def test_piece_that_rejoins_within_the_window_is_a_reattachment():
    an, gg = pinching_blob(3, 10, 5)
    r = tp.lc17_detachments(an, gg, RULE).iloc[0]
    assert r.outcome == "reattached" and r.fate_date == an.dates[13] and not r.final_separation


def test_piece_that_rejoins_after_the_window_is_a_separation_that_was_not_final():
    r = tp.lc17_detachments(*pinching_blob(3, 40, 5), RULE).iloc[0]
    assert r.outcome == "separated" and r.fate == "reattached" and not r.final_separation


@pytest.mark.parametrize(("join", "days_rejoined"), [(SHALLOW, 20), (DEEP, RULE.longer_than_days)])
def test_insignificant_touch_is_ignored_and_its_break_off_is_not_a_new_detachment(join, days_rejoined):
    an, gg = pinching_blob(3, 10, days_rejoined, join, 40)
    d = tp.lc17_detachments(an, gg, RULE)
    assert len(d) == 1
    r = d.iloc[0]
    assert r.date == an.dates[3] and r.outcome == "separated" and r.final_separation
    assert r.n_ignored_touch_days == days_rejoined
    assert r.max_ignored_touch_depth_m == pytest.approx(join - tp.LC_CONTOUR)


def test_deep_touch_longer_than_the_rule_reattaches_on_its_first_day():
    an, gg = pinching_blob(3, 10, RULE.longer_than_days + 1, DEEP, 40)
    d = tp.lc17_detachments(an, gg, RULE)
    assert d.iloc[0].outcome == "reattached" and d.iloc[0].fate_date == an.dates[13]
    assert len(d) == 2 and d.iloc[1].date == an.dates[13 + RULE.longer_than_days + 1]


@pytest.mark.parametrize(("fate", "days", "expected"), [
    ("reattached", 30, "reattached"), ("reattached", 31, "separated"), ("dissipated", 5, "dissipated"),
    ("dissipated", 31, "separated"), ("record_end", 10, "censored"), ("record_end", 31, "separated")])
def test_classify_detachment(fate, days, expected):
    assert tp.classify_detachment(fate, days) == expected


# ---------------------------------------------------------------------------------------------- distances

def test_wasserstein_and_bottleneck():
    a = np.array([[1.0, 0.0], [0.5, 0.3]])
    assert st.wasserstein(a, a) == 0.0 and st.bottleneck(a, a) == 0.0
    assert st.wasserstein(a[:1], np.empty((0, 2))) == pytest.approx(0.5)
    shifted = a + [[0.05, 0.0], [0.0, 0.02]]
    assert st.wasserstein(a, shifted) == pytest.approx(0.07) and st.bottleneck(a, shifted) == pytest.approx(0.05)
    extra = np.vstack([a, [[0.2, 0.1]]])
    assert st.wasserstein(a, extra) == pytest.approx(0.05) and st.bottleneck(a, extra) == pytest.approx(0.05)


# ------------------------------------------------------------------------------------------------- grids

def test_quarter_regrid_recovers_cell_means_and_masks_by_half():
    lon, lat = np.meshgrid(np.arange(-90.0, -88.0, 0.08) + 0.04, np.arange(24.0, 26.0, 0.08) + 0.04)
    lon, lat = lon.ravel(), lat.ravel()
    region = ~((lon > -89.0) & (lat > 25.0))
    g = st.quarter(lon, lat, region)
    value = np.floor(lon / 0.25) * 100 + np.floor(lat / 0.25)
    img = g(value)
    expect = np.floor(g.longitude / 0.25)[None] * 100 + np.floor(g.latitude / 0.25)[:, None]
    assert np.allclose(img[g.mask], expect[g.mask]) and np.isnan(img[~g.mask]).all()
    assert g.mask.sum() == 64 - 16 and np.allclose(g.longitude % 0.25, 0.125)
    assert np.allclose(g(np.stack([value, 2 * value]))[1][g.mask], 2 * img[g.mask])


def test_noise_floor_scales_with_sigma():
    lon, lat = np.arange(-97.875, -81.0, 0.25), np.arange(21.625, 30.0, 0.25)
    g = st.Grid(lat, lon, None, np.ones((len(lat), len(lon)), bool))
    a, b = st.noise_floor(g, 10, sigma_m=0.035, seed=1), st.noise_floor(g, 10, sigma_m=0.07, seed=1)
    assert b["floor_m"] == pytest.approx(2 * a["floor_m"]) and 0.01 < a["floor_m"] < 0.15


# ----------------------------------------------------------------------------------------- end to end

GULF_LON, GULF_LAT = np.arange(-97.96, -81.0, 0.08), np.arange(18.04, 30.0, 0.08)


def fields_file(tmp_path, n_start=2, horizon=6, damp=0.5):
    """Synthetic fields.nc on a 0.08 deg Gulf box: an LC tongue from the Yucatan box and eight eddies; the model
    damps the eddies with lead."""
    nx, ny = len(GULF_LON), len(GULF_LAT)
    LON, LAT = np.meshgrid(GULF_LON, GULF_LAT, indexing="ij")
    lon, lat = LON.ravel(), LAT.ravel()
    centres = np.random.default_rng(0).uniform([-96, 23], [-88, 29], (8, 2))
    sign = np.where(np.arange(8) % 2, 1.0, -1.0)

    def ssh(t, amp):
        eddies = sum(s * amp * 0.25 * np.exp(-((lon - c[0] + 0.05 * t) ** 2 + (lat - c[1]) ** 2) / (2 * 0.5**2))
                     for c, s in zip(centres, sign))
        return 0.5 * np.exp(-((lon + 86.0) ** 2 + np.clip(lat - 25.0, 0, None) ** 2) / (2 * 0.8**2)) * (lat > 21.0) + eddies

    lead = np.arange(1, horizon + 1)
    truth = np.stack([np.stack([ssh(s + l, 1.0) for l in lead]) for s in range(n_start)])
    model = np.stack([np.stack([ssh(s + l, damp ** (l / horizon)) for l in lead]) for s in range(n_start)])
    region = tp.gulf_region(GULF_LAT, GULF_LON).T.ravel() & (lat > 18.5)
    ds = xr.Dataset({"ssh_model": (("start", "lead", "grid_index"), model.astype(np.float32)),
                     "ssh_truth": (("start", "lead", "grid_index"), truth.astype(np.float32)),
                     "ssh_start": (("start", "grid_index"), np.stack([ssh(s, 1.0) for s in range(n_start)]).astype(np.float32))},
                    {"start": np.datetime64("2024-01-01", "ns") + np.arange(n_start) * np.timedelta64(1, "D"),
                     "lead": lead, "horizon": ("start", np.full(n_start, horizon)), "lon": ("grid_index", lon),
                     "lat": ("grid_index", lat), "region": ("grid_index", region),
                     "depth": ("grid_index", np.where(lat > 29.5, 50.0, 3000.0))},
                    {"nx": nx, "ny": ny, "dlon": 0.08, "checkpoint": "synthetic"})
    path = tmp_path / "fields.nc"
    ds.to_netcdf(path)
    return path


def test_lc_grid_seed_is_ten_quarter_cells(tmp_path):
    ds = xr.open_dataset(fields_file(tmp_path, 1, 1))
    cg, lg = st.grids(ds)
    assert st.lc_grid(lg).seed.sum() == 10
    assert cg.latitude[cg.mask.any(1)].max() < 29.5  # shelf left out of the census
    assert lg.latitude[lg.mask.any(1)].max() > 29.5  # and kept for the LC


def test_score_end_to_end_sees_damped_eddies(tmp_path):
    res = st.score(fields_file(tmp_path), tmp_path / "out")
    assert res.attrs["caveat"] == st.CAVEAT
    assert (res.max_w1_model.isel(lead=0) < res.max_w1_model.isel(lead=-1)).all()
    assert (res.min_amp_sum_model.isel(lead=-1) < res.min_amp_sum_truth.isel(lead=-1)).all()
    ev = json.loads((tmp_path / "out" / "events.json").read_text())
    assert ev["caveat"] == st.CAVEAT and isinstance(ev["events"], list)
    md = st.report(tmp_path / "out", "synthetic")
    assert st.CAVEAT in md and (tmp_path / "out" / "structure.png").exists()


def test_native_grid_matches_image_order(tmp_path):
    ds = xr.open_dataset(fields_file(tmp_path, 1, 1))
    g = st.native(ds, np.ones(ds.sizes["grid_index"], bool))
    assert np.allclose(g(ds.lon.values), g.longitude[None, :] * np.ones(g.mask.shape))
    assert np.allclose(g(ds.lat.values), g.latitude[:, None] * np.ones(g.mask.shape))


def lc_series(days):
    """A 0.25 deg Gulf with an LC tongue and a ring at 26 N; days: list of (ring lon, bridge height)."""
    lat, lon = np.arange(21.625, 30.0, 0.25), np.arange(-97.875, -81.0, 0.25)
    gg = tp.gulf_grid(np.ones((len(lat), len(lon)), bool), lat, lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    tongue = 0.45 * np.exp(-((LON + 86.0) ** 2 + np.clip(LAT - 25.0, 0, None) ** 2) / (2 * 0.7**2))
    frames = []
    for ring_lon, bridge in days:
        ring = 0.35 * np.exp(-((LON - ring_lon) ** 2 + (LAT - 26.0) ** 2) / (2 * 0.6**2))
        rid = bridge * np.exp(-((LAT - 26.0) ** 2) / (2 * 0.3**2)) * (LON >= ring_lon) * (LON <= -86.0)
        frames.append(np.maximum.reduce([tongue, ring, rid]))
    return np.stack(frames), pd.date_range("2024-01-01", periods=len(frames)), gg


def test_ring_drifting_away_is_one_final_separation_with_a_neck_track():
    ssh, dates, gg = lc_series([(-86.0 - 0.2 * t, 0.0) for t in range(50)])
    _, ev = st.lc_events(ssh, dates, gg, np.ones(gg.gulf.shape))
    assert len(ev) == 1 and not ev[0]["censored"]
    assert 1 < ev[0]["day"] < 15 and ev[0]["piece_lat"] > 24.0
    assert ev[0]["neck"][0] > st.LC_FLOOR_M and min(ev[0]["neck"]) < 0


def test_ring_that_rejoins_for_long_is_not_separated_on_its_first_break():
    days = [(-86.0 - 0.2 * t, 0.0) for t in range(10)] + [(-87.0, 0.4)] * 15 + [(-88.0, 0.0)] * 30
    ssh, dates, gg = lc_series(days)
    det, ev = st.lc_events(ssh, dates, gg, np.ones(gg.gulf.shape))
    assert "reattached" in set(det.fate)
    assert all(e["day"] > 10 for e in ev)


def test_match_events_pairs_nearest_and_lists_extras():
    def e(who, lead, start="2024-01-01", censored=False):
        return {"start": start, "who": who, "lead": lead, "censored": censored, "neck": {"0": 0.3}, "piece_km2": 2e4}
    rows = st.match_events([e("truth", 20), e("model", 26), e("model", 50), e("truth", 40, censored=True), e("truth", 5, "2024-02-01")])
    got = [(r["start"], r["truth"] and r["truth"]["lead"], r["model"] and r["model"]["lead"]) for r in rows]
    assert got == [("2024-01-01", 20, 26), ("2024-01-01", None, 50), ("2024-02-01", 5, None)]
