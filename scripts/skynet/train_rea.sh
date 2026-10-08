#!/bin/bash
# Train the reanalysis emulator on skynet with a rollout curriculum, STAGES (rollout days, default "1 2 4 8 16").
# usage: train_rea.sh <run_id> <out_dir>
# DATA is a rea_pack folder. STAGE=1 copies it to /dev/shm (removed on exit); STAGE=0 reads it in place.
# Each stage loads the previous stage's best checkpoint (weights and epoch; fresh optimizer) and has an epoch cap
# EPOCHS_<days> and a hard time cap CAP_<days>; with --plateau in EXTRA_ARGS a stage also stops when validation stalls.
# Stage <days> validates EVAL_<days> days (default max(4, days)) and keeps its own best; the scored checkpoint is the
# last stage's best. PF_<days> is the stage's --pushforward (default 4 from 8 days on, else 0; FIRST=1 adds
# --train_first_step), CKPT_<days>=1 its --checkpoint_steps (default from 16 days on). The test split is then scored
# over the last stage's validation horizon with neural-lam's own per-lead metrics.
# Overrides: GPU, BS, HIDDEN, LAYERS, LR, WORKERS, PRECISION, STAGES, EPOCHS_/CAP_/EVAL_/PF_/CKPT_<days>, FIRST,
# TRAIN, VAL, TEST (each "start end"), PY; EXTRA_ARGS (space-separated) is appended to every train_model call, e.g. a
# model or loss of an experiment arm, --compile --nondeterministic --plateau 2 3.
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
       --batch_size ${BS:-8} --lr ${LR:-1e-3} --precision ${PRECISION:-32} --val_interval 1
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
epoch_of() { $PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$1"; }
declare -A EPOCHS_DEF=([1]=30 [2]=15 [4]=10 [8]=8 [16]=6) CAP_DEF=([1]=4h [2]=3h [4]=4h [8]=6h [16]=10h)

echo "== $(date -Is) graph"
timeout 30m $PY -m hycom_emulator.nlam build_graph nlam.yaml multiscale
PREV=""; E=0
for AR in ${STAGES:-1 2 4 8 16}; do
  EP=EPOCHS_$AR; CP=CAP_$AR; EV=EVAL_$AR; PFV=PF_$AR; CKV=CKPT_$AR
  EVAL=${!EV:-$(( AR > 4 ? AR : 4 ))}; PF=${!PFV:-$(( AR >= 8 ? 4 : 0 ))}; CKPT=${!CKV:-$(( AR >= 16 ? 1 : 0 ))}
  STAGE_ARGS=(--ar_steps_train $AR --ar_steps_eval $EVAL --val_steps_to_log $(seq -s " " 1 $EVAL))
  [ "$PF" = 0 ] || STAGE_ARGS+=(--pushforward $PF); [ "$PF" = 0 ] || [ "${FIRST:-0}" = 0 ] || STAGE_ARGS+=(--train_first_step)
  [ "$CKPT" = 0 ] || STAGE_ARGS+=(--checkpoint_steps)
  LOAD=(); [ -z "$PREV" ] || { LOAD=(--load "$PREV"); E=$(( $(epoch_of "$PREV") + 1 )); }
  echo "== $(date -Is) stage ${AR}-day: ${STAGE_ARGS[*]}"
  timeout --signal=INT ${!CP:-${CAP_DEF[$AR]:-6h}} $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" "${LOAD[@]}" \
    "${STAGE_ARGS[@]}" --epochs $(( E + ${!EP:-${EPOCHS_DEF[$AR]:-6}} )) --logger_run_name "$RUN_ID-d$AR" \
    || echo "stage ${AR}-day exit $? (124 = time cap reached)"
  PREV=$(best "$RUN_ID-d$AR")
  [ -n "$PREV" ] || { echo "stage ${AR}-day left no checkpoint"; exit 1; }
  echo "after stage ${AR}-day best: $PREV"
done
echo "scored checkpoint: $PREV"
echo "== $(date -Is) test"
timeout 60m $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --load "$PREV" --eval test \
  --ar_steps_eval $EVAL --val_steps_to_log $(seq -s " " 1 $EVAL) --logger_run_name "$RUN_ID-test"
echo "== $(date -Is) done"
