"""The standardized state of a real rea_pack after the state-std floor, on a sample of its train rows.

Runs only with REA_PACK=<pack folder> (skynet: /scratch/jmiranda/hycom-emulator-data/rea_s2.pack, about 3 min for
every 10th row). The bound the redesign asked for (docs/redesign-2026-11-04-rcan-first-principles.md 2.1, 12 sigma)
does not hold on rea_s2 with the floor as specified, so that test is a strict xfail that says by how much: it starts
failing (XPASS) the day the statistics meet it, and then the mark goes.
"""

import os
from pathlib import Path

import pytest

from hycom_emulator import rea_pack

PACK = os.environ.get("REA_PACK")
BOUND = 12.0  # sigma
pytestmark = pytest.mark.skipif(not PACK, reason="set REA_PACK to a pack folder")


@pytest.fixture(scope="module")
def scan():
    return rea_pack.zscan(Path(PACK), every=int(os.environ.get("REA_PACK_EVERY", "10")))


def test_the_floor_only_lowers_the_extremes(scan):
    """Every channel's largest |z| is no larger after the floor; the floored ones are lower by raw / floored std."""
    for name, c in scan["channels"].items():
        assert c["max_sigma"] <= c["max_raw_sigma"] * (1 + 1e-6), name
        assert c["max_sigma"] == pytest.approx(c["max_raw_sigma"] * c["raw_std"] / c["std"], rel=1e-6), name


@pytest.mark.xfail(strict=True, reason="rea_s2, 2026-11-05, every 10th train row: after the floor 31 channels exceed "
                   "12 sigma, max 40.2 (salin_70m at -81.7 22.5, off Cuba); in the Gulf 10, max 27.0 (salin_4m in the "
                   "Atchafalaya plume). River plumes, the Florida Current and the Yucatan bottom layer are real tails "
                   "that a per-channel std floor does not bound; the model bounds them (rcan's LayerNorm entry and exit)")
def test_no_standardized_state_exceeds_the_bound(scan):
    over = {n: round(c["max_sigma"], 1) for n, c in scan["channels"].items() if c["max_sigma"] > BOUND}
    assert not over, f"{len(over)} channels above {BOUND} sigma: {over}"
