#!/bin/bash
# Progress and ETA of a train_b00.sh run. usage: train_status.sh <run_id>
# Stage caps match train_b00.sh: stage 1 300 epochs or 7 h, stage 2 100 epochs or 3.5 h.
OUT=/scratch/jmiranda/hycom-emulator-runs/${1:?run_id}
pgrep -f "train_b00.sh $1 " > /dev/null && STATE=running || STATE=stopped
/conda/jmiranda/venvs/hycom-emulator/bin/python - "$OUT" "$STATE" <<'PY'
import re, sqlite3, sys, time
out, state = sys.argv[1:]
log = open(f"{out}/train.log", errors="replace").read()
marks = re.findall(r"^== (\S+) (.+)$", log, re.M)
stage = marks[-1][1] if marks else "starting"
if "done" in stage or state == "stopped":
    print(f"{stage}; {state}; scores: {out}/scores_test.json"); sys.exit()
caps = {"stage 1": (300, 7 * 3600), "stage 2": (100, 210 * 60)}
key = next((k for k in caps if stage.startswith(k)), None)
if key is None:
    print(f"{stage}; {state}"); sys.exit()
try:
    db = sqlite3.connect(f"file:{out}/mlflow.db?mode=ro", uri=True)
    run = db.execute("select run_uuid, start_time from runs order by start_time desc limit 1").fetchone()
    rows = db.execute("select step, timestamp, value from metrics where run_uuid=? and key='train_loss_epoch' order by step", (run[0],)).fetchall()
    vals = db.execute("select value from metrics where run_uuid=? and key='val_mean_loss' order by step", (run[0],)).fetchall()
except Exception:
    rows, vals, run = [], [], None
max_ep, cap_s = caps[key]
if not rows:
    print(f"{key}: no epoch finished yet; {state}"); sys.exit()
t0 = run[1] / 1000
done, elapsed = len(rows), rows[-1][1] / 1000 - t0
per = elapsed / done
total = min(max_ep, cap_s / per)
eta = max(0, t0 + total * per - time.time())
best = min(v[0] for v in vals) if vals else float("nan")
print(f"{key}: {done} epochs ({100*done/total:.0f}% of {total:.0f} reachable), {per/60:.1f} min/epoch, "
      f"train {rows[-1][2]:.4f}, best val {best:.4f}, stage ETA {int(eta//3600):02d}:{int(eta%3600//60):02d}, {state}")
PY
