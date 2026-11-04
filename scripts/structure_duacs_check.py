"""Check hycom_emulator.topology against persistence-ocean on its own data: the LC17 final separations of DUACS ADT
2017-2021 at the reference setting must be the seven of results/summary/lc_events_summary_L50_snp_central.csv
(persistence-ocean 205e3c3). Exit 1 on any difference.

    python scripts/structure_duacs_check.py <dir with duacs_gulf_wide_YYYY.nc>   (skynet: /unity/g2/jmiranda/persistence-ocean/duacs_subset)
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from hycom_emulator import structure as st
from hycom_emulator import topology as tp

SHA256 = {  # data_version of the reference run
    2017: "d741fa218fe35c9fc8e5e128d8fd601f17b2d6fd1a83f72b69550e20f20e7c68",
    2018: "19ae1a70e38854ba86404114ad24858a41a8e10190122e217a0584accd421251",
    2019: "731605d17ec502e198e9d613aaa486b9269382840f9c838a519518d359f5bf20",
    2020: "f24bba7397a7cbeede11af95f0d60e86f6421cd4398bfcd1705b32e3cd76a83f",
    2021: "a606e28199c1162ff420142ab47d6a955cbe84304620f06fb4c3930d10daa147",
}
EXPECTED = [  # separation_date, censored, piece_max_lat, piece_max_lon, piece_pixels
    ("2017-11-09", False, 27.125, -87.375, 27), ("2017-11-19", False, 25.125, -88.625, 31),
    ("2018-02-10", False, 27.125, -88.375, 70), ("2018-06-21", False, 25.125, -86.375, 172),
    ("2019-07-05", False, 26.375, -87.875, 114), ("2020-03-30", False, 25.375, -87.875, 84),
    ("2021-11-25", False, 26.625, -89.375, 65),
]


def main(root: Path) -> int:
    paths = [root / f"duacs_gulf_wide_{y}.nc" for y in SHA256]
    bad = [p.name for p, h in zip(paths, SHA256.values()) if hashlib.sha256(p.read_bytes()).hexdigest() != h]
    if bad:
        print(f"data differs from the reference run: {bad}")
        return 1
    ds = xr.concat([xr.open_dataset(p)[["adt", "sla"]].load() for p in paths], "time").sortby("latitude").sortby("longitude")
    ds = ds.sel(longitude=slice(-98.0, -80.0), latitude=slice(18.0, 31.0))
    ocean = (ds.adt.notnull() & ds.sla.notnull()).all("time").values
    gg = tp.gulf_grid(ocean, ds.latitude.values.astype(float), ds.longitude.values.astype(float))
    print(f"seed cells {gg.seed.sum()}, Gulf cells {gg.gulf.sum()}")
    an = tp.gulf_anomaly(ds.adt.values, ds.time.values, gg)
    det = tp.lc17_detachments(an, gg, tp.TouchRule(st.LC_FLOOR_M, st.TOUCH_DAYS))
    ev = tp.separation_events(det)
    got = [(str(r.separation_date.date()), bool(r.censored), float(r.piece_max_lat), float(r.piece_max_lon), int(r.piece_pixels))
           for r in ev.itertuples()]
    for g in got:
        print(*g, "" if g in EXPECTED else "<- not in the reference")
    missing = [e for e in EXPECTED if e not in got]
    for e in missing:
        print(*e, "<- reference event missing")
    print(f"{len(det)} detachments; {len(got)} events, {len(EXPECTED)} expected: {'MATCH' if not missing and len(got) == len(EXPECTED) else 'DIFFER'}")
    return 0 if got == EXPECTED else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
