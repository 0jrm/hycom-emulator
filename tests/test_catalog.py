from datetime import datetime
from pathlib import Path

import pytest

from hycom_emulator.catalog import ROLES, cycles, index, unreadable_dirs
from hycom_emulator.system import SystemConfig

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"

pytestmark = pytest.mark.skipif(
    not Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4").is_dir(),
    reason="needs RCC /gpfs/research/coaps/abozec",
)


@pytest.fixture(scope="module")
def cfg():
    return SystemConfig.from_toml(CONFIG)


@pytest.fixture(scope="module")
def idx(cfg):
    return index(cfg)


def test_product_counts_match_survey(idx):
    counts = {p: len(f) for p, f in idx.items()}
    assert counts == {
        "restart": 7,
        "incupd": 185,
        "inc_nc": 184,
        "archv": 184,
        "archm": 368,
        "tsis_obs": 184,
        "tsis_inov": 184,
    }


def test_restarts_are_monthly_at_18z(idx):
    assert sorted(idx["restart"]) == [datetime(2025, m, 1, 18) for m in range(3, 10)]


def test_every_cycle_has_module_a_and_b00_inputs(cfg, idx):
    cyc = cycles(cfg, idx)
    assert (cyc[0].analysis, cyc[-1].analysis, len(cyc)) == (datetime(2025, 3, 2, 18), datetime(2025, 9, 1, 18), 184)
    needed = [r for r in ROLES if not r.startswith("restart")]
    assert [c.analysis for c in cyc if c.missing(needed)] == []


def test_every_product_dir_is_readable(cfg):
    assert [d for p in cfg.products for d in unreadable_dirs(cfg.product_glob(p))] == []


def _bfields(bpath: Path) -> dict[str, list[tuple[float, float]]]:
    """min/max per record of an archive .b, from the table after the `field  time step` header."""
    lines = bpath.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("field")) + 1
    out: dict[str, list[tuple[float, float]]] = {}
    for line in lines[start:]:
        name, rest = line.split("=", 1)
        lo, hi = (float(v) for v in rest.split()[-2:])
        out.setdefault(name.strip(), []).append((lo, hi))
    return out


def test_incupd_is_a_difference_without_momentum(idx):
    """incflg -1: incupd holds xa - xb. A full state never has a negative layer thickness; u, v are
    never updated."""
    times = sorted(idx["incupd"])
    for when in (times[1], times[len(times) // 2], times[-1]):
        f = _bfields(idx["incupd"][when].with_suffix(".b"))
        assert len(f["thknss"]) == 41 and all(lo < 0 for lo, _ in f["thknss"])
        for name in ("u-vel.", "v-vel.", "u_btrop", "v_btrop"):
            assert all(lo == hi == 0.0 for lo, hi in f[name]), name
