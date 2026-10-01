"""Atmospheric forcing the HYCOM run read, averaged to 6 h blocks on the model grid.

05.4 builds hourly forcing per month at run time (MAKE_FORCE=1) into data/{wvel,flux,pcip,grad,lrad,mslp}.
Each .b lists one `name: day,span,range = <dtime> <span> <min> <max>` line per hourly record;
record n of the .a is the field valid at that dtime. A block starting at t averages the samples at
t, t+1h, ..., t+5h. Wind speed and speed times each component are averaged hourly too, because
stress grows with the square of the wind and a mean of the wind would hide gusts.

Output zarr: atm_<field> (time, y, x), float32, time = block start.
Run `python -m hycom_emulator.forcing <system.toml> <out.zarr> <first YYYY-MM-DDTHH> <last YYYY-MM-DDTHH>`.
"""

from __future__ import annotations

import glob
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from gom_da.eval.hycom_archive import REC_ALIGN
from gom_da.eval.timeline import from_hycom_days

from hycom_emulator.system import SystemConfig

BLOCK = timedelta(hours=6)
_LINE = re.compile(r"^\s*\S+:\s*day,span,range\s*=\s*([0-9.]+)")
DERIVED = ("wndspd", "wndspd_ewd", "wndspd_nwd")


class HourlyField:
    """All monthly files of one forcing field, indexed by valid time (to the second)."""

    def __init__(self, pattern: str, idm: int, jdm: int):
        self.idm, self.jdm = idm, jdm
        n2d = idm * jdm
        self.rec_words = n2d + (-n2d) % REC_ALIGN
        self.where: dict[datetime, tuple[Path, int]] = {}
        for a in sorted(glob.glob(pattern)):
            a = Path(a)
            recs = [m.group(1) for m in map(_LINE.match, a.with_suffix(".b").read_text().splitlines()) if m]
            for irec, day in enumerate(recs):
                t = from_hycom_days(float(day)).replace(microsecond=0)
                t = datetime(t.year, t.month, t.day, t.hour) + timedelta(hours=round(t.minute / 60))
                self.where.setdefault(t, (a, irec))
        self._maps: dict[Path, np.memmap] = {}

    def at(self, t: datetime) -> np.ndarray:
        path, irec = self.where[t]
        m = self._maps.get(path)
        if m is None:
            m = self._maps[path] = np.memmap(path, dtype=">f4", mode="r")
        off = irec * self.rec_words
        return np.asarray(m[off : off + self.idm * self.jdm], dtype=np.float64).reshape(self.jdm, self.idm)


def block_means(fields: dict[str, HourlyField], start: datetime) -> dict[str, np.ndarray]:
    hours = [start + timedelta(hours=h) for h in range(6)]
    out = {n: np.mean([f.at(t) for t in hours], axis=0) for n, f in fields.items()}
    u = [fields["wndewd"].at(t) for t in hours]
    v = [fields["wndnwd"].at(t) for t in hours]
    spd = [np.hypot(a, b) for a, b in zip(u, v)]
    out["wndspd"] = np.mean(spd, axis=0)
    out["wndspd_ewd"] = np.mean([s * a for s, a in zip(spd, u)], axis=0)
    out["wndspd_nwd"] = np.mean([s * b for s, b in zip(spd, v)], axis=0)
    return out


def build(cfg: SystemConfig, out: Path, first: datetime, last: datetime) -> None:
    import xarray as xr

    from hycom_emulator.background import grid_of

    grid = grid_of(cfg)
    fields = {n: HourlyField(str(cfg.expt_dir / p), grid.idm, grid.jdm) for n, p in cfg.forcing_files.items()}
    starts = []
    t = first
    while t <= last:
        starts.append(t)
        t += BLOCK
    for n, t in enumerate(starts):
        means = block_means(fields, t)
        ds = xr.Dataset(
            {f"atm_{k}": (("time", "y", "x"), v[None].astype(np.float32)) for k, v in means.items()},
            coords={"time": [np.datetime64(t, "ns")]},
            attrs={"system": cfg.name, "block_hours": 6, "files": str(dict(cfg.forcing_files))},
        )
        if n == 0:
            ds.to_zarr(out, mode="w", consolidated=False)
        else:
            ds.to_zarr(out, append_dim="time", consolidated=False)


def main(argv: list[str]) -> None:
    cfg = SystemConfig.from_toml(Path(argv[1]))
    build(cfg, Path(argv[2]), datetime.fromisoformat(argv[3]), datetime.fromisoformat(argv[4]))


if __name__ == "__main__":
    main(sys.argv)
