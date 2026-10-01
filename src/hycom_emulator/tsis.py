"""TSIS observation files as flat tables.

`tsis_obs_*.nc` (xprep) holds layer obs on (prof, layer, type). `val - inov` is H(xb) at grid
point (grdj, grdi), one-based, in layer `layer`. Types are T, S, rho, pin (top-interface depth).
xprep clips innovations at the solver's rejection limits; a clipped `inov` no longer gives H(xb).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FILL_ABS = 1e20
TYPES = ("temp", "salin", "rho", "pin")
INOV_CLIP = {"temp": 5.0, "salin": 3.0, "pin": 300.0}


@dataclass(frozen=True)
class LayerObs:
    """One row per valid (profile, layer, type). i, j, k are zero-based array indices."""

    typ: np.ndarray
    prof: np.ndarray
    k: np.ndarray
    j: np.ndarray
    i: np.ndarray
    val: np.ndarray
    inov: np.ndarray
    err: np.ndarray
    qc: np.ndarray
    id: np.ndarray
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    @property
    def hxb(self) -> np.ndarray:
        return self.val - self.inov

    @property
    def clipped(self) -> np.ndarray:
        limit = np.array([INOV_CLIP.get(t, np.inf) for t in TYPES])[self.typ]
        return np.isclose(np.abs(self.inov), limit, rtol=0, atol=1e-4)

    def of(self, typ: str) -> LayerObs:
        sel = self.typ == TYPES.index(typ)
        return LayerObs(**{f: getattr(self, f)[sel] for f in self.__dataclass_fields__})


def read_layer_obs(path: Path) -> LayerObs:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        v = {n: np.asarray(ds[n][:]) for n in ("val", "inov", "err", "qc", "id", "time", "lon", "lat", "grdi", "grdj")}
    ok = np.abs(v["val"]) < FILL_ABS
    prof, k, typ = np.nonzero(ok)
    return LayerObs(
        typ=typ,
        prof=prof,
        k=k,
        j=v["grdj"][ok].astype(np.int64) - 1,
        i=v["grdi"][ok].astype(np.int64) - 1,
        **{n: v[n][ok].astype(np.float64) for n in ("val", "inov", "err", "time", "lon", "lat")},
        qc=v["qc"][ok],
        id=v["id"][ok],
    )
