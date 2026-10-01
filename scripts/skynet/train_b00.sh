#!/bin/bash
# Train and score B00 on skynet. usage: train_b00.sh <run_id> <out_dir>
# Stages the B00 zarr into /dev/shm and removes it on exit. Each stage has a hard time cap,
# so the whole run stays under the card's GPU-hour budget: graph 30 min, stage 1 (1-step) 7 h,
# stage 2 (2-step) 3.5 h, evaluation 30 min. Stage 2 resumes stage 1's best val checkpoint
# (neural-lam --load restores epoch, optimizer and the best val score), so it runs EPOCHS2 more
# epochs and saves a checkpoint only if val loss improves. The scored checkpoint is the lowest
# val loss over both stages.
# EPOCHS1, EPOCHS2, DATA, PY and GPU override the defaults (used by the CPU smoke test).
# Run detached: setsid nohup train_b00.sh <run_id> <out_dir> > <out_dir>/train.log 2>&1 &
set -euo pipefail
RUN_ID=$1
OUT=$2
DATA=${DATA:-/scratch/jmiranda/hycom-emulator-data/abozec_054_b00atm.zarr}
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
SHM=/dev/shm/$USER/$RUN_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# MLflow 3 refuses the file store; SQLite keeps metrics in one file:
#   sqlite3 <out>/mlflow.db "select key, step, value from metrics"
export MLFLOW_TRACKING_URI=sqlite:///$OUT/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=8
MODEL=(--model graph_lam --graph multiscale --hidden_dim 128 --processor_layers 4 --batch_size 4
       --lr 1e-3 --ar_steps_eval 2 --val_steps_to_log 1 2 --val_interval 1 --n_example_pred 0
       --num_workers 8 --logger mlflow --runs_root "$OUT/runs")

mkdir -p "$OUT" "$SHM"
trap 'rm -rf "$SHM"' EXIT
echo "== $(date -Is) stage data to $SHM"
rsync -a "$DATA/" "$SHM/b00.zarr/"
cat > "$OUT/b00.yaml" <<YAML
zarr: $SHM/b00.zarr
splits:
  train: [2025-03-04, 2025-07-31]
  val: [2025-08-06, 2025-08-15]
  test: [2025-08-21, 2025-09-01]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: b00.yaml\n' > "$OUT/nlam.yaml"
cd "$OUT"
best() { find "$OUT/runs" -path "*$1*/checkpoints/min_val_loss.ckpt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2; }

echo "== $(date -Is) graph"
[ -f graph/multiscale/metainfo.yaml ] || timeout 30m $PY -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
echo "== $(date -Is) stage 1: 1-step training"
timeout --signal=INT 7h $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" \
  --epochs ${EPOCHS1:-300} --ar_steps_train 1 --logger_run_name "$RUN_ID-s1" || echo "stage 1 exit $? (124 = time cap reached)"
S1=$(best "$RUN_ID-s1"); [ -n "$S1" ] || { echo "stage 1 left no checkpoint"; exit 1; }; echo "stage 1 best: $S1"
echo "== $(date -Is) stage 2: 2-step fine-tune"
E1=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$S1")
timeout --signal=INT 210m $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" \
  --epochs $(( E1 + 1 + ${EPOCHS2:-100} )) --ar_steps_train 2 --load "$S1" --logger_run_name "$RUN_ID-s2" || echo "stage 2 exit $? (124 = time cap reached)"
S2=$(best "$RUN_ID-s2")
if [ -z "$S2" ]; then echo "stage 2 did not beat stage 1 on val; scoring stage 1"; S2=$S1; fi
echo "scored checkpoint: $S2"
echo "== $(date -Is) evaluate"
for split in val test; do
  timeout 30m $PY -m hycom_emulator.evaluate_b00 nlam.yaml "$S2" "$OUT/scores_$split.json" --split $split --ar-steps 2 > /dev/null
done
$PY - "$OUT/scores_test.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("verdict:", "PASS" if r["verdict"]["pass"] else "FAIL")
for k, ok in r["verdict"]["rows"].items():
    field, lead = k.split()
    s = r["scores"][field][lead]
    print(f"{k:12s} model {s['rmse_model']:.4g} persistence {s['rmse_persistence']:.4g} persistence+inc {s['rmse_persistence_inc']:.4g} corr_change {s['corr_change']:.3f} {'ok' if ok else 'FAIL'}")
PY
echo "== $(date -Is) done"
