"""Free run (expt_05.3): store rows line up with a cycled run's, increments are zero. Runs on a compute node."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hycom_emulator.build_store import build
from hycom_emulator.system import SystemConfig, fingerprint

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_053.toml"
RMU = Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.3/data/relax.rmu.a")

pytestmark = pytest.mark.skipif(not RMU.is_file(), reason="needs RCC /gpfs/research/coaps/abozec")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    out = tmp_path_factory.mktemp("free") / "s.zarr"
    build(SystemConfig.from_toml(CONFIG), out, date(2025, 6, 1), date(2025, 6, 4))
    return out


def test_free_run_store(store):
    ds = xr.open_zarr(store, consolidated=False)
    assert ds.sizes["cycle"] == 4 and ds.attrs["system"] == "abozec_053"
    assert not [n for n in ds.data_vars if n.startswith(("xb_", "l3"))]
    ocean = ds["ocean"].values
    for n in ("temp", "salin", "thknss"):
        inc = ds[f"inc_{n}"].isel(cycle=0).values
        assert np.all(inc[:, ocean] == 0) and np.all(np.isnan(inc[:, ~ocean]))
    assert np.isfinite(ds["s00_temp"].isel(cycle=0, layer=0).values[ocean]).all()


def test_fingerprint_marks_assimilation_off():
    fp = fingerprint(SystemConfig.from_toml(CONFIG))
    assert fp.structural_fields["assimilation"] == "off"
    assert not [k for k in fp.structural_fields if k.startswith(("nlist.", "tsis."))]


def test_prepare_b00_on_free_run(store, tmp_path):
    from hycom_emulator.prepare_b00 import prepare

    prepare(store, tmp_path / "b00.zarr", RMU, np.datetime64("2025-06-04"), stride=16)
    b = xr.open_zarr(tmp_path / "b00.zarr")
    assert b.sizes["time"] == 2 and b["state"].dtype == np.float32
    assert float(np.abs(b["forcing"].values).max()) == 0.0
