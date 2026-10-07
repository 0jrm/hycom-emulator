"""collapse: the R2 collapse rule and its L0 anchor. Synthetic histories and a synthetic pack, no RCC data."""

import sqlite3

import numpy as np
from test_stack import SPLITS, T, _run, write_pack

from hycom_emulator.collapse import check, collapsed_at, trivial_loss

L0 = 1.106


def test_a_loss_parked_at_l0_is_a_collapse():
    y = [1.4618, 1.1134, 1.112, 1.1108, 1.1102, 1.1099] + [1.1095] * 20 + [1.1094] * 24  # emu-b00-053-conv-unet
    assert collapsed_at(y, L0) == 11


def test_healthy_runs_are_not_stopped():
    epochs = np.arange(300)
    finetune = 0.33 + 0.36 * np.exp(-epochs[:50] / 3)
    converged = 0.143 + 0.5 * np.exp(-epochs / 30)  # flat to < 0.001 per epoch late: the old rule stopped this
    scratch_start = L0 * (0.95 - 0.004 * epochs[:60])  # near L0 at first, improving > 1% per 5 epochs
    for y in (finetune, converged, scratch_start):
        assert collapsed_at(y, L0) is None


def test_a_stage_that_never_learns_is_stopped():
    assert collapsed_at(np.full(30, 0.98 * L0), L0) == 10


def test_check_reads_each_stage_against_its_l0(tmp_path):
    rng = np.random.default_rng(0)
    pack = write_pack(tmp_path / "pack", *_run(rng, False, rng.normal(size=(T, 12, 1))))
    out = tmp_path / "run"
    out.mkdir()
    (out / "b00.yaml").write_text(f"zarr: {pack}\n{SPLITS}")
    (out / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    one, two = trivial_loss(str(out / "nlam.yaml"), "wmse")
    assert np.isfinite([one, two]).all() and min(one, two) > 0
    db = sqlite3.connect(out / "mlflow.db")
    db.execute("create table runs (run_uuid text, name text)")
    db.execute("create table metrics (run_uuid text, key text, value real, step integer)")
    db.executemany("insert into runs values (?, ?)", [("a", "x-s1"), ("b", "x-s2")])
    rows = [("a", "train_loss_epoch", 0.3 * one, s) for s in range(30)]
    rows += [("b", "train_loss_epoch", max(one, two), 100 + s) for s in range(30)]
    db.executemany("insert into metrics values (?, ?, ?, ?)", rows)
    db.commit()
    verdict = check(out, "control")
    assert verdict and verdict.startswith("x-s2 collapsed at epoch 10"), "stage 1 is healthy; stage 2 sits at L0"
    assert (out / "l0.json").is_file()


def test_persistence_scores_every_lead_of_a_split(tmp_path, monkeypatch):
    import json

    from hycom_emulator import persistence
    from hycom_emulator.persistence import persistence_losses

    rng = np.random.default_rng(0)
    pack = write_pack(tmp_path / "pack", *_run(rng, False, rng.normal(size=(T, 12, 1))))
    (tmp_path / "b00.yaml").write_text(f"zarr: {pack}\n{SPLITS}")
    (tmp_path / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    s = persistence_losses(str(tmp_path / "nlam.yaml"), "train", 2)
    assert s.shape[1] == 2 and np.isfinite(s).all() and (s > 0).all()
    assert np.isclose(s[:, 0].mean(), trivial_loss(str(tmp_path / "nlam.yaml"), "wmse", n=len(s))[0])
    out = tmp_path / "p.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["persistence", str(tmp_path / "nlam.yaml"), str(out), "--split", "train", "--ar-steps", "2"])
    persistence.main()
    r = json.loads(out.read_text())
    assert set(r["per_lead"]) == {"1", "2"} and r["samples"] == len(s)
