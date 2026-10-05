#!/bin/bash
# Train and score B00 on skynet. usage: train_b00.sh <run_id> <out_dir>
# Stages the B00 data (a pack_b00 or stack_b00 folder; a zarr also works) into /dev/shm and
# removes it on exit. Each stage has a hard time cap,
# so the whole run stays under the card's GPU-hour budget: graph 30 min, stage 1 (1-step) 7 h,
# stage 2 (2-step) 3.5 h, evaluation 30 min. Stage 2 resumes stage 1's best val checkpoint
# (neural-lam --load restores weights and epoch; the optimizer restarts without --restore_opt), so
# it runs EPOCHS2 more epochs. Each stage
# writes to a new run dir, where Lightning does not restore the best val score, so a stage saves
# its own best even when it is worse than the stage before: lowest() compares the scores stored
# in the checkpoints, and the scored checkpoint is the lowest val loss over both stages.
# RESUME1=<checkpoints dir of a stopped stage 1 of the same run design> continues that stage from
# its last.ckpt; give CAP1 the stage-1 time it has left, so the card's GPU budget holds.
# INIT=<checkpoint> skips stage 1 and fine-tunes INIT as stage 2 (EPOCHS2 epochs, CAP2 time cap,
# fresh optimizer); the scored checkpoint is then stage 2's own best, since its loss may differ
# from INIT's. ARM=<arm of hycom_emulator.physics.ARMS> sets the model, loss, loss weights, conv
# settings and graph (unet builds graph/multiscale_s4 from graph/multiscale).
# EVAL_MODEL=<neural-lam model> scores the checkpoint with that step predictor (hycom_graph_lam,
# hycom_conv_graph_lam: the thickness projection at prediction time).
# EPOCHS1, EPOCHS2, CAP1, CAP2, DATA, PY and GPU override the defaults (used by the CPU smoke test).
# Run detached: setsid nohup train_b00.sh <run_id> <out_dir> > <out_dir>/train.log 2>&1 &
set -euo pipefail
RUN_ID=$1
OUT=$2
DATA=${DATA:-/scratch/jmiranda/hycom-emulator-data/abozec_054_b00atm2.pack}
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
SHM=/dev/shm/$USER/$RUN_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# MLflow 3 refuses the file store; SQLite keeps metrics in one file:
#   sqlite3 <out>/mlflow.db "select key, step, value from metrics"
export MLFLOW_TRACKING_URI=sqlite:///$OUT/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=8
MODEL=(--graph multiscale --hidden_dim 128 --processor_layers 4 --batch_size 4
       --lr 1e-3 --ar_steps_eval 2 --val_steps_to_log 1 2 --val_interval 1 --n_example_pred 0
       --num_workers 8 --logger mlflow --runs_root "$OUT/runs")

mkdir -p "$OUT" "$SHM"
trap 'rm -rf "$SHM"' EXIT
echo "== $(date -Is) stage data to $SHM"
rsync -a "$DATA/" "$SHM/b00/"
cat > "$OUT/b00.yaml" <<YAML
zarr: $SHM/b00
splits:
  train: [2025-03-04, 2025-07-31]
  val: [2025-08-06, 2025-08-15]
  test: [2025-08-21, 2025-09-01]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: b00.yaml\n' > "$OUT/nlam.yaml"
cd "$OUT"
ARM_ARGS=$($PY -m hycom_emulator.physics arm "${ARM:-control}" b00.yaml nlam.yaml); MODEL+=($ARM_ARGS); echo "arm ${ARM:-control}: $ARM_ARGS"
best() { find "$OUT/runs" -path "*$1*/checkpoints/min_val_loss.ckpt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2; }
lowest() {  # the checkpoint with the lowest stored val loss; empty arguments are skipped
  $PY - "$@" <<'PY'
import sys, torch
def score(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return min(float(v["best_model_score"]) for k, v in d["callbacks"].items() if "val_mean_loss" in k)
paths = [p for p in sys.argv[1:] if p]
print(min(paths, key=score) if paths else "")
PY
}

echo "== $(date -Is) graph"
[ -f graph/multiscale/metainfo.yaml ] || timeout 30m $PY -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
timeout 30m $PY -m hycom_emulator.convnet coarse_graph nlam.yaml multiscale
if [ -n "${INIT:-}" ]; then S1=$INIT; echo "== $(date -Is) no stage 1: fine-tuning $INIT"; else
echo "== $(date -Is) stage 1: 1-step training"
LOAD1=(); [ -z "${RESUME1:-}" ] || { LOAD1=(--load "$RESUME1/last.ckpt"); echo "resuming $RESUME1/last.ckpt"; }
timeout --signal=INT ${CAP1:-7h} $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" "${LOAD1[@]}" \
  --epochs ${EPOCHS1:-300} --ar_steps_train 1 --logger_run_name "$RUN_ID-s1" || echo "stage 1 exit $? (124 = time cap reached)"
S1=$(lowest "$(best "$RUN_ID-s1")" ${RESUME1:+"$RESUME1/min_val_loss.ckpt"})
[ -n "$S1" ] || { echo "stage 1 left no checkpoint"; exit 1; }; echo "stage 1 best: $S1"
fi
echo "== $(date -Is) stage 2: 2-step fine-tune"
E1=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$S1")
timeout --signal=INT ${CAP2:-210m} $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" \
  --epochs $(( E1 + 1 + ${EPOCHS2:-100} )) --ar_steps_train 2 --load "$S1" --logger_run_name "$RUN_ID-s2" || echo "stage 2 exit $? (124 = time cap reached)"
if [ -n "${INIT:-}" ]; then S2=$(best "$RUN_ID-s2"); [ -n "$S2" ] || { echo "fine-tune left no checkpoint"; exit 1; }
else
S2=$(lowest "$S1" "$(best "$RUN_ID-s2")")  # val uses --ar_steps_eval 2 in both stages, so the scores compare
[ "$S2" != "$S1" ] || echo "stage 2 did not beat stage 1 on val; scoring stage 1"
fi
echo "scored checkpoint: $S2"
echo "== $(date -Is) evaluate"
for split in val test; do
  timeout 30m $PY -m hycom_emulator.evaluate_b00 nlam.yaml "$S2" "$OUT/scores_$split.json" --split $split --ar-steps 2 ${EVAL_MODEL:+--model $EVAL_MODEL} > /dev/null
done
timeout 30m $PY -m hycom_emulator.physcheck_b00 nlam.yaml "$S2" "$OUT/physcheck_test.json" --split test ${EVAL_MODEL:+--model $EVAL_MODEL}
$PY - "$OUT/scores_test.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for member, m in r.get("members", {"": r}).items():
    print(f"verdict {member}:", "PASS" if m["verdict"]["pass"] else "FAIL")
    for k, ok in m["verdict"]["rows"].items():
        field, lead = k.split()
        s = m["scores"][field][lead]
        print(f"{k:12s} model {s['rmse_model']:.4g} persistence {s['rmse_persistence']:.4g} persistence+inc {s['rmse_persistence_inc']:.4g} corr_change {s['corr_change']:.3f} {'ok' if ok else 'FAIL'}")
PY
echo "== $(date -Is) done"
