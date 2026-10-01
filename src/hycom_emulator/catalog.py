"""Per-cycle index of the files one system kept.

A cycle is named by its analysis time t_a (18Z). HYCOM ran t_a-24h -> t_a applying the previous
increment over the whole window, TSIS analysed the 24 h mean of that run, and xa2inc wrote the
increment valid at t_a. Every role below is a product at a fixed offset from t_a.

Run `python -m hycom_emulator.catalog configs/systems/<name>.toml` for a coverage summary.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from gom_da.eval.timeline import from_hycom_days, stamp_time

from hycom_emulator.system import SystemConfig

H = timedelta(hours=1)
ROLES: dict[str, tuple[str, timedelta]] = {
    "restart_start": ("restart", -24 * H),  # instantaneous state the run started from
    "incupd_applied": ("incupd", -24 * H),  # increment applied during the run
    "archm_21": ("archm", -21 * H),  # 6 h mean, t_a-24h .. t_a-18h
    "archv_00": ("archv", -18 * H),  # snapshot 6 h into the run
    "archm_09": ("archm", -9 * H),  # 18 h mean, t_a-18h .. t_a; with archm_21 the TSIS background
    "inc_nc": ("inc_nc", 0 * H),  # TSIS increment (target of module A)
    "incupd": ("incupd", 0 * H),  # same increment as HYCOM reads it
    "restart_end": ("restart", 0 * H),
    "tsis_obs": ("tsis_obs", 6 * H),  # TSIS labels the analysis t_a+6h (time_offset 0.25)
    "tsis_inov": ("tsis_inov", 6 * H),
}
_TSIS_STAMP = re.compile(r"_(\d{10})\.nc$")
_RESTART_DTIME = re.compile(r"nstep,dtime,thbase\s*=\s*\d+\s+([0-9.]+)")


@dataclass(frozen=True)
class Cycle:
    analysis: datetime
    files: dict[str, Path | None]

    def missing(self, roles=ROLES) -> list[str]:
        return [r for r in roles if self.files[r] is None]


def valid_time(product: str, path: Path) -> datetime:
    if product == "restart":
        m = _RESTART_DTIME.search(path.with_suffix(".b").read_text(errors="replace"))
        if m is None:
            raise ValueError(f"no dtime in {path.with_suffix('.b')}")
        return from_hycom_days(float(m.group(1)))
    if product.startswith("tsis_"):
        m = _TSIS_STAMP.search(path.name)
        if m is None:
            raise ValueError(f"no YYYYMMDDHH stamp in {path.name}")
        return datetime.strptime(m.group(1), "%Y%m%d%H")
    return stamp_time(path.name)


def unreadable_dirs(pattern: str) -> list[Path]:
    """Directories the product glob would search but cannot list. glob skips them silently."""
    return [
        Path(d)
        for d in glob.glob(os.path.dirname(pattern))
        if os.path.isdir(d) and not os.access(d, os.R_OK | os.X_OK)
    ]


def index(cfg: SystemConfig) -> dict[str, dict[datetime, Path]]:
    out: dict[str, dict[datetime, Path]] = {}
    for product in cfg.products:
        files: dict[datetime, Path] = {}
        for name in sorted(glob.glob(cfg.product_glob(product))):
            path = Path(name)
            when = valid_time(product, path)
            # Monthly tar dirs symlink the boundary file of the previous month.
            if when in files and files[when].resolve() != path.resolve():
                raise ValueError(f"{product}: {files[when]} and {path} are both valid at {when}")
            files.setdefault(when, path)
        out[product] = files
    return out


def cycles(cfg: SystemConfig, idx: dict[str, dict[datetime, Path]] | None = None) -> list[Cycle]:
    idx = index(cfg) if idx is None else idx
    first = datetime.combine(cfg.first_cycle, datetime.min.time()) + 18 * H
    last = datetime.combine(cfg.last_cycle, datetime.min.time()) + 18 * H
    out = []
    t = first
    while t <= last:
        files = {role: idx.get(product, {}).get(t + offset) for role, (product, offset) in ROLES.items()}
        out.append(Cycle(t, files))
        t += 24 * H
    return out


def summary(cfg: SystemConfig) -> str:
    idx = index(cfg)
    lines = [f"system {cfg.name} (canonical={cfg.canonical})", "", "product      files  first             last"]
    for product, files in idx.items():
        times = sorted(files)
        span = f"{times[0]:%Y-%m-%d %H}Z  {times[-1]:%Y-%m-%d %H}Z" if times else "-"
        lines.append(f"{product:<12} {len(files):>5}  {span}")
        for d in unreadable_dirs(cfg.product_glob(product)):
            lines.append(f"{'':<12} UNREADABLE {d}")
    cyc = cycles(cfg, idx)
    lines += ["", f"cycles {cyc[0].analysis:%Y-%m-%d} .. {cyc[-1].analysis:%Y-%m-%d} ({len(cyc)})", "role            present  missing analysis dates"]
    for role in ROLES:
        gaps = [c.analysis for c in cyc if c.files[role] is None]
        lines.append(f"{role:<15} {len(cyc) - len(gaps):>7}  {_ranges(gaps)}")
    return "\n".join(lines)


def _ranges(times: list[datetime]) -> str:
    if not times:
        return "-"
    spans, start, prev = [], times[0], times[0]
    for t in times[1:] + [None]:
        if t is not None and t - prev == 24 * H:
            prev = t
            continue
        spans.append(f"{start:%m-%d}" if start == prev else f"{start:%m-%d}..{prev:%m-%d}")
        if t is not None:
            start = prev = t
    return ", ".join(spans)


if __name__ == "__main__":
    print(summary(SystemConfig.from_toml(Path(sys.argv[1]))))
