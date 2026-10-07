#!/bin/bash
# Fine-tune a reanalysis emulator checkpoint on skynet in one stage, then score the test split.
# usage: finetune_rea.sh <run_id> <out_dir> <checkpoint>
# The run keeps the checkpoint's architecture and graph (GRAPH_DIR is linked next to the config) and loads its
# weights and epoch with a fresh optimizer, as train_rea.sh's stages do. Validation unrolls EVAL_AR days (default 4,
# as emu-rea-s2b), so the val loss compares with the start checkpoint's when the loss is the same.
# rea.yaml carries exclude_source_changes (EXCLUDE, default true): train windows that straddle a source-experiment
# change are dropped (hycom_emulator.datastore.SOURCE_CHANGES).
# Overrides: GPU, BS, LR, AR, EVAL_AR, EPOCHS, CAP, WORKERS, TRAIN, VAL, TEST (each "start end"), DATA, GRAPH_DIR, PY;
# EXTRA_ARGS (space-separated) is appended to every train_model call: the arm's loss and rollout flags
# (hycom_emulator.rea_train), e.g. "--loss rea_wmse --mean_penalty 0.0016 --mean_scales <series.npz> --pushforward 4".
# Run detached: setsid nohup finetune_rea.sh <run_id> <out_dir> <ckpt> < /dev/null > <out_dir>/train.log 2>&1 &
set -euo pipefail
RUN_ID=$1
OUT=$2
CKPT=$3
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
DATA=${DATA:-/scratch/jmiranda/hycom-emulator-data/rea_s2.pack}
read -r TR0 TR1 <<<"${TRAIN:-2001-01-16 2021-12-31}"
read -r VA0 VA1 <<<"${VAL:-2022-01-01 2023-12-31}"
read -r TE0 TE1 <<<"${TEST:-2024-01-01 2024-08-31}"
export CUDA_VISIBLE_DEVICES=${GPU-1}  # GPU= (empty) means CPU
export MLFLOW_TRACKING_URI=sqlite:///$OUT/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=8
EVAL_AR=${EVAL_AR:-4}
MODEL=(--model graph_lam --graph multiscale --hidden_dim ${HIDDEN:-128} --processor_layers ${LAYERS:-4}
       --batch_size ${BS:-8} --lr ${LR:-1e-3} --ar_steps_eval $EVAL_AR --val_steps_to_log $(seq -s " " 1 $EVAL_AR)
       --val_interval 1 --n_example_pred 0 --num_workers ${WORKERS:-8} --logger mlflow --runs_root "$OUT/runs")
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"; MODEL+=("${EXTRA[@]}")

mkdir -p "$OUT"
cat > "$OUT/rea.yaml" <<YAML
zarr: $DATA
exclude_source_changes: ${EXCLUDE:-true}
splits:
  train: [$TR0, $TR1]
  val: [$VA0, $VA1]
  test: [$TE0, $TE1]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: rea.yaml\n' > "$OUT/nlam.yaml"
[ -z "${GRAPH_DIR:-}" ] || [ -e "$OUT/graph" ] || ln -s "$GRAPH_DIR" "$OUT/graph"
cd "$OUT"
timeout 30m $PY -m hycom_emulator.nlam build_graph nlam.yaml multiscale
E=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'] + 1)" "$CKPT")
echo "== $(date -Is) fine-tune from $CKPT (epoch $E): ${AR:-8}-day rollouts, ${EXTRA_ARGS:-no arm flags}"
timeout --signal=INT ${CAP:-12h} $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --load "$CKPT" \
  --epochs $(( E + ${EPOCHS:-10} )) --ar_steps_train ${AR:-8} --logger_run_name "$RUN_ID" || echo "fine-tune exit $? (124 = time cap reached)"
BEST=$(find "$OUT/runs/$RUN_ID" -name min_val_loss.ckpt | head -1)
[ -n "$BEST" ] || { echo "fine-tune left no checkpoint"; exit 1; }
echo "== $(date -Is) test $BEST"
timeout 60m $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --load "$BEST" --eval test \
  --logger_run_name "$RUN_ID-test"
echo "== $(date -Is) done"
