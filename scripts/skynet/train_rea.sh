#!/bin/bash
# Train the reanalysis emulator on skynet with a 1 -> 2 -> 4 day rollout curriculum. usage: train_rea.sh <run_id> <out_dir>
# DATA is a rea_pack folder. STAGE=1 copies it to /dev/shm (removed on exit); STAGE=0 reads it in place.
# Each stage loads the previous stage's best checkpoint (weights and epoch; fresh optimizer) and has a hard time cap.
# Validation unrolls 4 days in every stage, so the stored val losses compare across stages and the scored
# checkpoint is the lowest over all three. The test split is then scored with neural-lam's own per-lead metrics.
# Overrides: GPU, BS, HIDDEN, LAYERS, LR, WORKERS, EPOCHS1..3, CAP1..3, TRAIN, VAL, TEST (each "start end"), PY;
# EXTRA_ARGS (space-separated) is appended to every train_model call, e.g. a model or loss of an experiment arm.
# Run detached: setsid nohup train_rea.sh <run_id> <out_dir> < /dev/null > <out_dir>/train.log 2>&1 &
set -euo pipefail
RUN_ID=$1
OUT=$2
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
SHM=/dev/shm/$USER/$RUN_ID
read -r TR0 TR1 <<<"${TRAIN:-2001-01-16 2021-12-31}"
read -r VA0 VA1 <<<"${VAL:-2022-01-01 2023-12-31}"
read -r TE0 TE1 <<<"${TEST:-2024-01-01 2024-08-31}"
export CUDA_VISIBLE_DEVICES=${GPU-3}  # GPU= (empty) means CPU
export MLFLOW_TRACKING_URI=sqlite:///$OUT/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=8
MODEL=(--model graph_lam --graph multiscale --hidden_dim ${HIDDEN:-128} --processor_layers ${LAYERS:-4}
       --batch_size ${BS:-8} --lr ${LR:-1e-3} --ar_steps_eval 4 --val_steps_to_log 1 2 4 --val_interval 1
       --n_example_pred 0 --num_workers ${WORKERS:-8} --logger mlflow --runs_root "$OUT/runs")
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"; MODEL+=("${EXTRA[@]}")

mkdir -p "$OUT" "$SHM"
trap 'rm -rf "$SHM"' EXIT
if [ "${STAGE:-1}" = 1 ]; then echo "== $(date -Is) stage data to $SHM"; rsync -a "$DATA/" "$SHM/rea/"; ZARR=$SHM/rea
else echo "== $(date -Is) reading $DATA in place (STAGE=0)"; ZARR=$DATA; fi
cat > "$OUT/rea.yaml" <<YAML
zarr: $ZARR
splits:
  train: [$TR0, $TR1]
  val: [$VA0, $VA1]
  test: [$TE0, $TE1]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: rea.yaml\n' > "$OUT/nlam.yaml"
cd "$OUT"
best() { find "$OUT/runs" -path "*$1*/checkpoints/min_val_loss.ckpt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2; }
lowest() {
  $PY - "$@" <<'PY'
import sys, torch
def score(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return min(float(v["best_model_score"]) for k, v in d["callbacks"].items() if "val_mean_loss" in k)
paths = [p for p in sys.argv[1:] if p]
print(min(paths, key=score) if paths else "")
PY
}
epoch_of() { $PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$1"; }

echo "== $(date -Is) graph"
timeout 30m $PY -m hycom_emulator.nlam build_graph nlam.yaml multiscale
PREV=""; E=0
for s in 1 2 3; do
  AR=$(( 1 << (s - 1) )); EP=EPOCHS$s; CAP=CAP$s
  LOAD=(); [ -z "$PREV" ] || { LOAD=(--load "$PREV"); E=$(( $(epoch_of "$PREV") + 1 )); }
  echo "== $(date -Is) stage $s: ${AR}-day rollouts"
  timeout --signal=INT ${!CAP:-3h} $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" "${LOAD[@]}" \
    --epochs $(( E + ${!EP:-20} )) --ar_steps_train $AR --logger_run_name "$RUN_ID-s$s" || echo "stage $s exit $? (124 = time cap reached)"
  PREV=$(lowest "$PREV" "$(best "$RUN_ID-s$s")")
  [ -n "$PREV" ] || { echo "stage $s left no checkpoint"; exit 1; }
  echo "after stage $s best: $PREV"
done
echo "scored checkpoint: $PREV"
echo "== $(date -Is) test"
timeout 60m $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --load "$PREV" --eval test \
  --logger_run_name "$RUN_ID-test"
echo "== $(date -Is) done"
