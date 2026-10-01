"""Which HYCOM + TSIS system a dataset came from, and whether two systems share configuration.

A system is one cycled DA run: HYCOM executable, runtime blkdat, TSIS build and namelist,
initial restart and relaxation mask. `structural` covers everything except step sizes, output
frequencies and labels, so the June segment of 05.4 (baclin 120 s) matches the rest of the run.
`strict` adds those numbers and the sha256 of the executables and the initial restart.

Run `python -m hycom_emulator.system configs/systems/<name>.toml` to print a fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from gom_da.eval.setup_card import sha256_file
from gom_da.eval.timeline import parse_blkdat, parse_nlist, parse_runtime_echo

NUMERICAL_KEYS = frozenset(
    {"baclin", "batrop", "incstp", "dsurfq", "diagfq", "proffq", "tilefq", "meanfq", "rstrfq", "cplifq"}
)
LABEL_KEYS = frozenset({"iexpt", "itest", "jtest"})
# Namelist paths that are climatologies the analysis reads, not per-run obs or output directories.
NLIST_CONFIG_LOCATIONS = frozenset({"tsis_stat_location", "sclim_data_location", "dclim_data_location"})


@dataclass(frozen=True)
class SystemConfig:
    name: str
    canonical: bool
    note: str
    expt_dir: Path
    hycom_exe: Path
    blkdat: Path
    nlist: Path
    initial_restart: Path
    runtime_log: Path | None
    relax_rmu: Path
    tsis_bin_dir: Path
    tsis_bins: tuple[str, ...]
    forcing: str
    obs_dir: Path
    first_cycle: date
    last_cycle: date
    products: dict[str, str]
    forcing_files: dict[str, str]

    @classmethod
    def from_toml(cls, path: Path) -> SystemConfig:
        raw = tomllib.loads(Path(path).read_text())
        expt = Path(raw["expt_dir"])
        return cls(
            name=raw["name"],
            canonical=raw["canonical"],
            note=raw["note"],
            expt_dir=expt,
            hycom_exe=expt / raw["hycom_exe"],
            blkdat=expt / raw["blkdat"],
            nlist=expt / raw["nlist"],
            initial_restart=expt / raw["initial_restart"],
            runtime_log=expt / raw["runtime_log"] if "runtime_log" in raw else None,
            relax_rmu=expt / raw["relax_rmu"],
            tsis_bin_dir=Path(raw["tsis_bin_dir"]),
            tsis_bins=tuple(raw["tsis_bins"]),
            forcing=raw["forcing"],
            obs_dir=Path(raw["obs_dir"]),
            first_cycle=raw["first_cycle"],
            last_cycle=raw["last_cycle"],
            products=dict(raw["products"]),
            forcing_files=dict(raw.get("forcing_files", {})),
        )

    def product_glob(self, product: str) -> str:
        """Glob for one product's files. Relative patterns are under expt_dir; `{obs_dir}` expands."""
        pattern = self.products[product].format(obs_dir=self.obs_dir)
        return pattern if pattern.startswith("/") else str(self.expt_dir / pattern)


@dataclass(frozen=True)
class Fingerprint:
    structural_fields: dict[str, str]
    strict_fields: dict[str, str]

    @property
    def structural(self) -> str:
        return _digest(self.structural_fields)

    @property
    def strict(self) -> str:
        return _digest({**self.structural_fields, **self.strict_fields})

    def to_dict(self) -> dict:
        return {
            "structural": self.structural,
            "strict": self.strict,
            "structural_fields": self.structural_fields,
            "strict_fields": self.strict_fields,
        }


def _digest(fields: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def fingerprint(cfg: SystemConfig) -> Fingerprint:
    """Integer flags HYCOM echoed at run time win over the blkdat on disk, which can be edited after
    a run. Echoed reals are printed to 4 decimals (diapyc 1e-7 echoes as 0), so blkdat keeps those."""
    blk = parse_blkdat(cfg.blkdat.read_text())
    if cfg.runtime_log is not None:
        echo = parse_runtime_echo(cfg.runtime_log.read_text(errors="replace"))
        blk.update({k: echo[k] for k in blk if k in echo and blk[k].is_integer() and echo[k].is_integer()})

    structural = {
        f"blkdat.{k}": f"{v:g}" for k, v in blk.items() if k not in NUMERICAL_KEYS | LABEL_KEYS
    }
    for key, value in parse_nlist(cfg.nlist.read_text()).items():
        if not key.endswith("_location") or key in NLIST_CONFIG_LOCATIONS:
            structural[f"nlist.{key}"] = value
    structural["tsis.build"] = cfg.tsis_bin_dir.name
    structural["forcing"] = cfg.forcing
    structural["relax_rmu"] = sha256_file(cfg.relax_rmu)

    strict = {f"blkdat.{k}": f"{blk[k]:g}" for k in sorted(NUMERICAL_KEYS & blk.keys())}
    strict["hycom_exe"] = sha256_file(cfg.hycom_exe)
    for name in cfg.tsis_bins:
        strict[f"tsis.{name}"] = sha256_file(cfg.tsis_bin_dir / name)
    strict["initial_restart"] = sha256_file(cfg.initial_restart)
    return Fingerprint(structural, strict)


def main(argv: list[str]) -> None:
    cfg = SystemConfig.from_toml(Path(argv[1]))
    print(json.dumps({"name": cfg.name, "canonical": cfg.canonical, **fingerprint(cfg).to_dict()}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
