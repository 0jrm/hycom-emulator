"""Collapse kill for B00 training: stop a stage whose loss sits at the level of a forecast with no spatial information.

A collapsed network outputs (nearly) the same change everywhere; the unet arm of emu-b00-053-conv did, with an output
spatial std of 6e-5 and a training loss of 1.1094. No spatially constant forecast can score below L0, the training
loss of persistence on the run's own data, loss and rollout length (the best constant change scores within 0.5% of
it): 1.106 for that run, so the collapse sat exactly on it. Healthy runs end at 0.22-0.41 of L0. The rule (R2):
from epoch 10 of a stage, the 5-epoch mean of the epoch train loss is at least 0.9 L0 and improved by less than 1%
over the previous 5 epochs. Replayed on 19 stage histories, it stops only the collapsed run (at epoch 11). The rule
it replaces, "the loss changes by less than 0.001 over five epochs", also stopped a healthy run converging at 0.143.

    python -m hycom_emulator.collapse check <run out dir> <ARM> [AR2]   exit 0 and print the stage if collapsed

`check` caches L0 in <out>/l0.json; it reads the run's nlam.yaml and mlflow.db, one MLflow run (stage) at a time.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from hycom_emulator.persistence import persistence_losses

RATIO, GAIN, WINDOW, FIRST_EPOCH = 0.9, 0.01, 5, 10


def collapsed_at(y, l0: float) -> int | None:
    """First epoch (1-based) at which the epoch train losses y of one stage meet the rule, or None."""
    y = np.asarray(y, dtype=float)
    ma = np.convolve(y, np.ones(WINDOW) / WINDOW, mode="valid")  # ma[j]: mean of epochs j+1 .. j+WINDOW
    for j in range(WINDOW, len(ma)):
        epoch = j + WINDOW
        if epoch >= FIRST_EPOCH and ma[j] >= RATIO * l0 and ma[j] >= (1 - GAIN) * ma[j - WINDOW]:
            return epoch
    return None


def stage_histories(db: Path) -> dict[str, np.ndarray]:
    """Epoch train loss per MLflow run (one per training stage), in step order within each run."""
    q = ("select r.name, m.value from metrics m join runs r on m.run_uuid = r.run_uuid "
         "where m.key = 'train_loss_epoch' order by r.name, m.step")
    out: dict[str, list[float]] = {}
    for name, v in sqlite3.connect(db).execute(q):
        out.setdefault(name, []).append(v)
    return {n: np.array(v) for n, v in out.items()}


def trivial_loss(config_path: str, loss_name: str, n: int = 24) -> tuple[float, float]:
    """(one-step, two-step mean) training loss of persistence on n train samples of the run's datastore."""
    s = persistence_losses(config_path, "train", 2, loss_name, n)
    return float(s[:, 0].mean()), float(s.mean(1).mean())


def check(out: Path, arm: str, ar2: int = 2) -> str | None:
    """'<stage> collapsed at epoch E (MA5 = r L0)' for the first collapsed stage of the run in out, else None."""
    from hycom_emulator.physics import ARMS

    cache = out / "l0.json"
    if not cache.is_file():
        l0 = trivial_loss(str(out / "nlam.yaml"), ARMS[arm]["loss"])
        cache.write_text(json.dumps({"one_step": l0[0], "two_step": l0[1]}))
    l0 = json.loads(cache.read_text())
    for name, y in sorted(stage_histories(out / "mlflow.db").items()):
        ref = l0["one_step"] if name.endswith("-s1") or ar2 == 1 else l0["two_step"]
        epoch = collapsed_at(y, ref)
        if epoch:
            return f"{name} collapsed at epoch {epoch} (MA5 = {y[epoch - WINDOW:epoch].mean() / ref:.2f} L0, L0 = {ref:.3f})"
    return None


if __name__ == "__main__":
    if sys.argv[1:2] != ["check"] or len(sys.argv) not in (4, 5):
        raise SystemExit("usage: python -m hycom_emulator.collapse check <run out dir> <ARM> [AR2]")
    verdict = check(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) == 5 else 2)
    if verdict:
        print(verdict)
    sys.exit(0 if verdict else 1)
