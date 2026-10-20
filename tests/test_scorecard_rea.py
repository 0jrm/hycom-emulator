import numpy as np

from hycom_emulator.rollout_rea import STATS, Start, to_dataset
from hycom_emulator.scorecard_rea import departure, scorecard


def test_departure_needs_three_leads_outside():
    c = np.array([1.0, 1.2, 1.0, 1.2, 1.3, 1.4, 1.0])
    assert departure(c, (0.9, 1.1)) == 4
    assert np.isnan(departure(np.ones(5), (0.9, 1.1)))


def test_scorecard_finds_the_lead_where_kinetic_energy_collapses():
    days = np.arange(np.datetime64("2020-01-01"), np.datetime64("2024-06-01"), 60).astype("datetime64[D]")
    starts = [Start(d, i, 30) for i, d in enumerate(days)]
    rng = np.random.default_rng(0)
    v = 1 + 0.01 * rng.standard_normal((len(starts), 30, len(STATS)))
    names = list(STATS)
    v[:, 11:, names.index("ke_model")] = 0.5
    for k in ("op_geo_s15_residual_truth", "op_geo_s15_residual_model"):
        v[..., names.index(k)] = 0.2
    ds = to_dataset(starts, v, {})
    rows = {r["diagnostic"]: r for r in scorecard(ds, "2024-01-01")}
    assert rows["ke"]["departs_test"] == 12 and rows["ke"]["departs_in_sample"] == 12
    assert np.isnan(rows["maxspeed"]["departs_test"])
    assert rows["geo_s15"]["usable_as_loss"] and not rows["geo_s50"]["usable_as_loss"]
    assert np.isnan(rows["geo_s15"]["departs_test"])
