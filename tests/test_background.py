"""Gate for everything downstream: our background must be the one TSIS analysed."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hycom_emulator.background import MASSLESS_PA, cycle_background, grid_of
from hycom_emulator.catalog import cycles
from hycom_emulator.system import SystemConfig
from hycom_emulator.tsis import read_layer_obs

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"
FLOAT32_ATOL = 2e-5  # .a records are 32-bit; T near 30 C has a 2e-6 ulp

pytestmark = pytest.mark.skipif(
    not Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4").is_dir(),
    reason="needs RCC /gpfs/research/coaps/abozec",
)


@pytest.fixture(scope="module")
def cfg():
    return SystemConfig.from_toml(CONFIG)


@pytest.mark.parametrize("analysis", [datetime(2025, 4, 2, 18), datetime(2025, 6, 15, 18), datetime(2025, 8, 30, 18)])
def test_background_reproduces_tsis_hxb(cfg, analysis):
    cycle = next(c for c in cycles(cfg) if c.analysis == analysis)
    xb = cycle_background(cycle, grid_of(cfg))
    obs = read_layer_obs(cycle.files["tsis_obs"])
    dw = xb.thknss * xb.oneta
    for typ in ("temp", "salin"):
        o = obs.of(typ)
        diff = np.abs(o.hxb - xb.layer[typ][o.k, o.j, o.i])
        massive = dw[o.k, o.j, o.i] >= MASSLESS_PA
        diff, massive = diff[~o.clipped], massive[~o.clipped]
        assert o.k.size > 500
        assert diff[massive].max() < FLOAT32_ATOL, typ
        # archv2restart resets T in massless layers; keep that population small and visible.
        assert (diff[~massive] >= FLOAT32_ATOL).mean() < 0.1, typ
