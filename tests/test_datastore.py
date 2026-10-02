"""B00 layout: our store -> neural-lam stacked zarr -> HycomDatastore. Runs on a compute node."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from hycom_emulator.build_store import build
from hycom_emulator.system import SystemConfig

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"
RMU = Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4/data/relax.rmu.a")

pytestmark = pytest.mark.skipif(not RMU.is_file(), reason="needs RCC /gpfs/research/coaps/abozec")


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    from hycom_emulator.prepare_b00 import prepare

    root = tmp_path_factory.mktemp("b00")
    store = root / "s.zarr"
    build(SystemConfig.from_toml(CONFIG), store, date(2025, 6, 1), date(2025, 6, 5))
    prepare(store, root / "b00.zarr", RMU, np.datetime64("2025-06-04"), stride=16)
    (root / "b00.yaml").write_text(
        f"zarr: {root / 'b00.zarr'}\nsplits:\n  train: [2025-06-03, 2025-06-04]\n"
        "  val: [2025-06-05, 2025-06-05]\n  test: [2025-06-05, 2025-06-05]\n"
    )
    return store, root / "b00.yaml"


def test_layout_and_masks(stores):
    from hycom_emulator.datastore import HycomDatastore

    _, cfg = stores
    ds = HycomDatastore(cfg)
    st = ds.get_dataarray("state", "train", standardize=True)
    assert st.dims == ("time", "grid_index", "state_feature")
    assert [str(t)[:13] for t in st["time"].values] == ["2025-06-03T00", "2025-06-04T00"]
    assert ds.get_num_data_vars("state") == 41 * 5 + 4 and ds.get_num_data_vars("forcing") == 41 * 3
    assert not np.isnan(st.values).any()
    bm = ds.boundary_mask.values.astype(bool)
    ocean = ds.get_dataarray("static", None).sel(static_feature="ocean").values.astype(bool)
    assert bm[~ocean].all() and 0.2 < (~bm).mean() < 0.8
    assert ds.step_length.days == 1 and ds.grid_shape_state.x * ds.grid_shape_state.y == bm.size


def test_forcing_is_the_iau_share_of_two_increments(stores):
    import xarray as xr

    from hycom_emulator.datastore import HycomDatastore

    store, cfg = stores
    ds = HycomDatastore(cfg)
    f = ds.get_dataarray("forcing", "val").isel(time=0)
    gi = int(np.flatnonzero(~ds.boundary_mask.values.astype(bool))[0])
    x, y = (float(v) for v in ds.get_xy("forcing", stacked=True)[gi])
    with xr.open_zarr(store, consolidated=False) as s:
        j = int(np.argmin(np.abs(s["plat"].values[:, 0] - y)))
        i = int(np.argmin(np.abs(s["plon"].values[0, :] - x)))
        inc = s["inc_temp"].sel(cycle=[np.datetime64(datetime(2025, 6, d, 18)) for d in (3, 4)]).values[:, 0, j, i]
    assert float(f.sel(forcing_feature="inc_temp_k01").isel(grid_index=gi)) == pytest.approx(0.75 * inc[0] + 0.25 * inc[1], abs=1e-6)


def test_no_channel_dominates_the_loss(stores):
    from hycom_emulator.datastore import HycomDatastore
    from hycom_emulator.prepare_b00 import DIFF_STD_FLOOR

    _, cfg = stores
    r = HycomDatastore(cfg).get_standardization_dataarray("state").state_diff_std_standardized.values
    assert np.isfinite(r).all() and r.min() >= DIFF_STD_FLOOR * (1 - 1e-6)
