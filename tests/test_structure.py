import json

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from hycom_emulator import structure as st


def bumps(shape, specs):
    """sum of amp * exp(-r^2 / (2 w^2)) on an index grid; specs: (amp, y, x, w)."""
    y, x = np.mgrid[: shape[0], : shape[1]].astype(float)
    return sum(a * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * w**2)) for a, cy, cx, w in specs)


def full(shape):
    m = np.ones(shape, bool)
    return m, st.graph(m)


def test_two_bumps_pairs_and_neck():
    mask, g = full((20, 40))
    h = bumps(mask.shape, [(1.0, 10, 10, 3), (0.6, 10, 30, 3)])
    m = st.superlevel(h[mask], g, seed=int(np.ravel_multi_index((10, 10), mask.shape)))
    finite = m.pairs[~m.essential]
    big = finite[st.persistence(finite) > 1e-6]
    saddle = h[10, 10:31].min()  # lowest point of the best path between the peaks is on the row joining them
    assert len(big) == 1
    assert big[0] == pytest.approx([h[10, 30], saddle])
    assert m.pairs[m.essential][0] == pytest.approx([h.max(), h.min()])
    neck = m.neck_depth(h[mask])
    assert neck[np.ravel_multi_index((10, 30), mask.shape)] == pytest.approx(h[10, 30] - saddle)
    assert neck[np.ravel_multi_index((10, 10), mask.shape)] == 0.0


def test_neck_when_the_seed_side_is_lower():
    mask, g = full((20, 40))
    h = bumps(mask.shape, [(0.5, 10, 10, 3), (0.9, 10, 30, 3)])
    m = st.superlevel(h[mask], g, seed=int(np.ravel_multi_index((10, 10), mask.shape)))
    saddle = h[10, 10:31].min()
    assert m.neck_depth(h[mask])[np.ravel_multi_index((10, 30), mask.shape)] == pytest.approx(h[10, 10] - saddle)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_betti0_matches_thresholded_components(seed):
    rng = np.random.default_rng(seed)
    mask = ndimage.gaussian_filter(rng.standard_normal((30, 45)), 3) > -0.15  # ragged coast and islands of land
    g = st.graph(mask)
    h = ndimage.gaussian_filter(rng.standard_normal(mask.shape), 1.5)[mask]
    pairs = st.superlevel(h, g).pairs
    for t in np.quantile(h, [0.05, 0.3, 0.5, 0.7, 0.9, 0.99]):
        img = np.zeros(mask.shape, bool)
        img[mask] = h >= t
        assert ndimage.label(img)[1] == ((pairs[:, 0] >= t) & (pairs[:, 1] < t)).sum()


def test_minima_are_maxima_of_the_negated_field():
    mask, g = full((20, 40))
    h = -bumps(mask.shape, [(0.3, 10, 10, 3), (0.2, 10, 30, 3)])
    d = st.diagrams(h[mask], g)
    assert st.census(d["min"], 0.05)["count"] == 2
    assert st.census(d["max"], 0.05)["count"] == 1  # only the essential pair: the field has no positive bump


def test_census_floor():
    pairs = np.array([[0.3, 0.0], [0.1, 0.06], [0.2, 0.1]])
    c = st.census(pairs, 0.05)
    assert c["count"] == 2 and c["amp_sum"] == pytest.approx(0.4)


def test_wasserstein_and_bottleneck():
    a = np.array([[1.0, 0.0], [0.5, 0.3]])
    assert st.wasserstein(a, a) == 0.0 and st.bottleneck(a, a) == 0.0
    assert st.wasserstein(a[:1], np.empty((0, 2))) == pytest.approx(0.5)  # to the diagonal: persistence / 2
    shifted = a + [[0.05, 0.0], [0.0, 0.02]]
    assert st.wasserstein(a, shifted) == pytest.approx(0.07)
    assert st.bottleneck(a, shifted) == pytest.approx(0.05)
    # a weak extra point costs its half persistence; the bottleneck is then the largest single cost
    extra = np.vstack([a, [[0.2, 0.1]]])
    assert st.wasserstein(a, extra) == pytest.approx(0.05)
    assert st.bottleneck(a, extra) == pytest.approx(0.05)
    assert st.bottleneck(a, shifted) <= st.wasserstein(a, shifted)


def test_quarter_regrid_recovers_cell_means_and_masks_by_half():
    lon, lat = np.meshgrid(np.arange(-90.0, -88.0, 0.08) + 0.04, np.arange(24.0, 26.0, 0.08) + 0.04)
    lon, lat = lon.ravel(), lat.ravel()
    region = ~((lon > -89.0) & (lat > 25.0))  # one quarter of the box is land
    region[(lon > -89.25) & (lon < -89.0) & (lat > 24.0) & (lat < 24.25) & (lon < -89.15)] = False  # under half of a cell
    r = st.quarter(lon, lat, region)
    cell_value = np.floor(lon / 0.25) * 100 + np.floor(lat / 0.25)
    out = r(cell_value)
    assert np.allclose(out, (np.floor(r.grid.lon / 0.25) * 100 + np.floor(r.grid.lat / 0.25))[r.grid.mask])
    assert not r.grid.mask[(r.grid.lon > -89.0) & (r.grid.lat > 25.0)].any()
    assert r.grid.mask.sum() == 64 - 16  # 8 x 8 cells, the land quarter out; the cell missing ~40% stays
    assert np.allclose(r(np.stack([cell_value, 2 * cell_value])), [out, 2 * out])
    assert np.allclose(r.grid.lon[0] % 0.25, 0.125)


def test_noise_floor_scales_with_sigma_and_falls_with_correlation():
    lon, lat = np.meshgrid(np.arange(-97.875, -81.0, 0.25), np.arange(21.625, 30.0, 0.25))
    grid = st.native(lon, lat, np.ones(lon.shape, bool), 0.25)
    a = st.noise_floor(grid, 20, sigma_m=0.035, seed=1)
    b = st.noise_floor(grid, 20, sigma_m=0.07, seed=1)
    assert b["floor_m"] == pytest.approx(2 * a["floor_m"])
    assert 0.02 < a["floor_m"] < 0.1
    assert st.noise_floor(grid, 20, corr_km=150, seed=1)["features_per_field"] < a["features_per_field"]


GULF = dict(lon=np.arange(-97.875, -81.0, 0.25), lat=np.arange(21.625, 30.0, 0.25))


def gulf_grid():
    lon, lat = np.meshgrid(GULF["lon"], GULF["lat"])
    return st.native(lon, lat, np.ones(lon.shape, bool), 0.25)


def lc_field(grid, ring_lon, ring_lat, ring_amp=0.35, bridge=0.0):
    """An LC tongue from the Yucatan box north to ~26 N and a ring; `bridge` adds a ridge between them."""
    lon, lat = grid.lon[grid.mask], grid.lat[grid.mask]
    tongue = 0.45 * np.exp(-((lon + 86.0) ** 2) / (2 * 0.7**2)) * (lat < 25.0) + 0.45 * np.exp(
        -((lon + 86.0) ** 2 + (lat - 25.0) ** 2) / (2 * 0.7**2)) * (lat >= 25.0)
    ring = ring_amp * np.exp(-((lon - ring_lon) ** 2 + (lat - ring_lat) ** 2) / (2 * 0.6**2))
    ridge = bridge * np.exp(-((lat - ring_lat) ** 2) / (2 * 0.3**2)) * (lon >= min(ring_lon, -86.0)) * (lon <= max(ring_lon, -86.0))
    return np.maximum(np.maximum(tongue, ring), ridge)


def run_lc(grid, rings):
    box = st.yucatan_cells(grid)
    days = [lc_day(grid, box, *r) for r in rings]
    return st.separations(days, grid)


def lc_day(grid, box, ring_lon, ring_lat, bridge=0.0):
    return st.lc_day(lc_field(grid, ring_lon, ring_lat, bridge=bridge), grid, box)


def test_yucatan_box_has_ten_quarter_cells():
    assert len(st.yucatan_cells(gulf_grid())) == 10


def test_ring_that_drifts_away_is_a_separation():
    grid = gulf_grid()
    path = [(-86.0 - 0.2 * t, 26.0) for t in range(50)]  # starts on the tongue's tip, drifts west 20 km a day
    evs = run_lc(grid, path)
    sep = [e for e in evs if e.status == "separation"]
    assert len(sep) == 1 and len(evs) == 1
    d = sep[0].detached
    assert 1 < d < 15
    assert sep[0].neck[d] > st.LC_FLOOR_M  # once detached the ring is its own peak, well above the floor
    assert sep[0].neck[d - 1] < sep[0].neck[d + 5]


def test_counted_rejoin_cancels_and_brief_rejoin_does_not():
    grid = gulf_grid()
    away = [(-86.0 - 0.2 * t, 26.0, 0.0) for t in range(10)]
    back_long = [(-87.0, 26.0, 0.4)] * 15 + [(-88.0, 26.0, 0.0)] * 30  # a 0.4 m ridge: margin 0.23 m for 15 days
    back_short = [(-87.0, 26.0, 0.4)] * 5 + [(-88.5 - 0.2 * t, 26.0, 0.0) for t in range(40)]
    long_ = run_lc(grid, away + back_long)
    short = run_lc(grid, away + back_short)
    assert long_[0].status == "rejoined" and long_[0].rejoined is not None
    assert short[0].status == "separation" and short[0].rejoined is None
    assert max(short[0].margin.values()) >= st.LC_FLOOR_M  # it did reconnect, only for too short a time


def test_series_too_short_is_unresolved():
    grid = gulf_grid()
    evs = run_lc(grid, [(-86.0 - 0.2 * t, 26.0) for t in range(20)])
    assert [e.status for e in evs] == ["unresolved"]


def fields_file(tmp_path, n_start=2, horizon=6, damp=0.5):
    """A synthetic fields.nc on a 0.08 deg grid: truth = eddies on an LC; model = the same with eddies damped."""
    lon1, lat1 = np.arange(-97.96, -81.0, 0.08), np.arange(21.54, 30.0, 0.08)
    nx, ny = len(lon1), len(lat1)
    LON, LAT = np.meshgrid(lon1, lat1, indexing="ij")  # (nx, ny): grid_index = ix * ny + iy
    lon, lat = LON.ravel(), LAT.ravel()
    rng = np.random.default_rng(0)
    centres = rng.uniform([-96, 23], [-88, 29], (8, 2))
    sign = np.where(np.arange(8) % 2, 1.0, -1.0)

    def ssh(t, amp):
        eddies = sum(s * amp * 0.25 * np.exp(-((lon - c[0] + 0.05 * t) ** 2 + (lat - c[1]) ** 2) / (2 * 0.5**2))
                     for c, s in zip(centres, sign))
        return 0.5 * np.exp(-((lon + 86.0) ** 2 + np.clip(lat - 25.0, 0, None) ** 2) / (2 * 0.8**2)) + eddies

    lead = np.arange(1, horizon + 1)
    truth = np.stack([np.stack([ssh(s + l, 1.0) for l in lead]) for s in range(n_start)])
    model = np.stack([np.stack([ssh(s + l, damp ** (l / horizon)) for l in lead]) for s in range(n_start)])
    ds = xr.Dataset({"ssh_model": (("start", "lead", "grid_index"), model.astype(np.float32)),
                     "ssh_truth": (("start", "lead", "grid_index"), truth.astype(np.float32)),
                     "ssh_start": (("start", "grid_index"), np.stack([ssh(s, 1.0) for s in range(n_start)]).astype(np.float32))},
                    {"start": np.datetime64("2024-01-01", "ns") + np.arange(n_start) * np.timedelta64(1, "D"),
                     "lead": lead, "horizon": ("start", np.full(n_start, horizon)), "lon": ("grid_index", lon),
                     "lat": ("grid_index", lat), "region": ("grid_index", np.ones(lon.size, bool))},
                    {"nx": nx, "ny": ny, "dlon": 0.08, "checkpoint": "synthetic"})
    path = tmp_path / "fields.nc"
    ds.to_netcdf(path)
    return path


def test_score_end_to_end_sees_damped_eddies(tmp_path):
    res = st.score(fields_file(tmp_path), tmp_path / "out")
    assert res.attrs["caveat"] == st.CAVEAT
    assert (res.max_w1_model.isel(lead=0) < res.max_w1_model.isel(lead=-1)).all()
    assert (res.max_amp_sum_model.isel(lead=-1) < res.max_amp_sum_truth.isel(lead=-1)).all()
    assert (res.min_amp_sum_model.isel(lead=-1) < res.min_amp_sum_truth.isel(lead=-1)).all()
    ev = json.loads((tmp_path / "out" / "events.json").read_text())
    assert ev["caveat"] == st.CAVEAT and isinstance(ev["events"], list)


def test_native_grid_matches_image_order(tmp_path):
    ds = xr.open_dataset(fields_file(tmp_path, n_start=1, horizon=1))
    grid, to = st.scoring_grid(ds, use_native=True)
    v = to(ds.lon.values)
    assert np.allclose(v, grid.lon[grid.mask]) and np.allclose(to(ds.lat.values), grid.lat[grid.mask])
