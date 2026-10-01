import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from hycom_emulator.system import SystemConfig, fingerprint

CONFIG = Path(__file__).parents[1] / "configs/systems/abozec_054.toml"
EXPT = Path("/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/GOMb0.04/expt_05.4")

pytestmark = pytest.mark.skipif(not EXPT.is_dir(), reason="needs RCC /gpfs/research/coaps/abozec")


@pytest.fixture(scope="module")
def cfg():
    return SystemConfig.from_toml(CONFIG)


@pytest.fixture(scope="module")
def base(cfg):
    return fingerprint(cfg)


def test_runtime_blkdat_is_the_surveyed_one(cfg):
    assert hashlib.md5(cfg.blkdat.read_bytes()).hexdigest() == "c17d0019eb63a6274ddd3d3b313274af"


def test_runtime_echo_confirms_24h_iau(base):
    f = {**base.structural_fields, **base.strict_fields}
    assert (f["blkdat.incflg"], f["blkdat.incstp"], f["blkdat.baclin"]) == ("-1", "384", "225")


def test_june_time_step_is_numerical_only(cfg, base):
    june = fingerprint(replace(cfg, blkdat=EXPT / "blkdat.input_125f", runtime_log=None))
    assert june.structural == base.structural
    assert june.strict != base.strict


def test_interior_relax_of_05_1_is_structural(cfg, base):
    twin = fingerprint(replace(cfg, relax_rmu=EXPT.parent / "expt_05.1/data/relax.rmu.a"))
    assert twin.structural != base.structural
