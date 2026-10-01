"""Check a built store against the TSIS files it came from, cycle by cycle.

For each cycle: max |H(xb) - (val - inov)| over unclipped T and S layer obs whose stored layer mass
is at least MASSLESS_PA (must be below ATOL), and whether the stored increment is non-trivial over
ocean. Exits 1 if any cycle fails.

Run `python -m hycom_emulator.verify_store <system.toml> <store.zarr>`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hycom_emulator.background import MASSLESS_PA
from hycom_emulator.catalog import cycles
from hycom_emulator.system import SystemConfig
from hycom_emulator.tsis import read_layer_obs

ATOL = 2e-5


def verify(cfg: SystemConfig, store: Path) -> list[str]:
    import xarray as xr

    by_time = {np.datetime64(c.analysis, "ns"): c for c in cycles(cfg)}
    failures = []
    with xr.open_zarr(store, consolidated=False) as ds:
        ocean = ds["ocean"].values
        print("cycle        n_T   n_S   max_dT     max_dS     inc_ok")
        for t in ds["cycle"].values:
            cyc = by_time[t]
            obs = read_layer_obs(cyc.files["tsis_obs"])
            sel = ds.sel(cycle=t)
            dw = (sel["xb_thknss"] * sel["xb_oneta"]).values
            worst, counts = {}, {}
            for typ in ("temp", "salin"):
                o = obs.of(typ)
                keep = ~o.clipped & (dw[o.k, o.j, o.i] >= MASSLESS_PA)
                x = sel[f"xb_{typ}"].values[o.k[keep], o.j[keep], o.i[keep]]
                worst[typ] = float(np.abs(o.hxb[keep] - x).max()) if keep.any() else np.nan
                counts[typ] = int(keep.sum())
            inc = sel["inc_thknss"].values[:, ocean]
            inc_ok = bool(np.isfinite(inc).any() and np.nanmax(np.abs(inc)) > 0)
            stamp = f"{str(t)[:13]}"
            print(f"{stamp}  {counts['temp']:5d} {counts['salin']:5d}  {worst['temp']:.2e}  {worst['salin']:.2e}  {inc_ok}")
            if not (worst["temp"] < ATOL and worst["salin"] < ATOL and inc_ok):
                failures.append(stamp)
    return failures


def main(argv: list[str]) -> None:
    failures = verify(SystemConfig.from_toml(Path(argv[1])), Path(argv[2]))
    print(f"FAIL {len(failures)} cycles: {failures}" if failures else "PASS all cycles")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv)
