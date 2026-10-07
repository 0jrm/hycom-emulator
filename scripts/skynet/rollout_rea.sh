#!/bin/bash
# Queued long free runs of the final emu-rea-s2b checkpoint (hycom_emulator.rollout_rea). usage: rollout_rea.sh <code_dir> <out_dir>
# Waits for "eval done" in the chain log, reads the scored checkpoint from the training log, waits for an idle GPU
# (2, else 3; never 0 or 1), then runs every default start for HORIZON days, the summary figures and the movies.
# A guard kills the job if host memory in use (MemTotal - MemAvailable) passes LIMIT_GB. Start and end go to the chain log.
# Overrides: HORIZON, CHUNK, BATCH, LIMIT_GB, CHAIN_LOG, TRAIN_LOG, PY.
# Run detached: setsid nohup rollout_rea.sh <code_dir> <out_dir> < /dev/null > <out_dir>.queue.log 2>&1 &
set -uo pipefail
CODE=$1
OUT=$2
X=/scratch/jmiranda/hycom-emulator-runs
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
CHAIN_LOG=${CHAIN_LOG:-$X/emu-rea-s2b.log}
TRAIN_LOG=${TRAIN_LOG:-$X/emu-rea-s2b/train.log}
LIMIT_GB=${LIMIT_GB:-520}
say() { echo "$(date -Is) rollout_rea: $*" >> "$CHAIN_LOG"; echo "$(date -Is) $*"; }
in_use_gb() { awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print int((t-a)/1048576)}' /proc/meminfo; }
idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")" -lt 1000 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1")" ]; }

until grep -q "eval done" "$CHAIN_LOG"; do
  grep -qE "eval (gulf|interior) failed|eval: no scored checkpoint|left no checkpoint" "$CHAIN_LOG" && { say "chain failed before eval done; not running"; exit 1; }
  sleep 120
done
CKPT=$(sed -n 's/^scored checkpoint: //p' "$TRAIN_LOG" | tail -1)
[ -f "$CKPT" ] || { say "no scored checkpoint in $TRAIN_LOG"; exit 1; }
until GPU=$(for g in 2 3; do idle $g && { echo $g; break; }; done) && [ -n "$GPU" ]; do sleep 60; done

mkdir -p "$OUT"
cd "$OUT"
cat > rea.yaml <<YAML
zarr: /scratch/jmiranda/hycom-emulator-data/rea_s2.pack
splits:
  train: [2001-01-16, 2024-08-31]
  val: [2022-01-01, 2023-12-31]
  test: [2024-01-01, 2024-08-31]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: rea.yaml\n' > nlam.yaml
ln -sfn "$X/emu-rea-s2b/graph" graph

export PYTHONPATH=$CODE/src HYCOM_EMULATOR_SHA=$(git -C "$CODE" rev-parse HEAD) CUDA_VISIBLE_DEVICES=$GPU MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=8
H=${HORIZON:-90}
R=(nice "$PY" -m hycom_emulator.rollout_rea)
cat > job.sh <<JOB
#!/bin/bash
set -e
${R[*]} run nlam.yaml "$CKPT" . --horizon $H --chunk ${CHUNK:-15} --batch ${BATCH:-8}
${R[*]} figures stats.nc figs
${R[*]} movie nlam.yaml "$CKPT" movies --horizon $H --chunk ${CHUNK:-15}
JOB
chmod +x job.sh
say "started on GPU $GPU, checkpoint $CKPT, horizon $H, out $OUT, host memory in use $(in_use_gb) GB"
setsid ./job.sh < /dev/null > run.log 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  used=$(in_use_gb)
  (( used > peak )) && peak=$used
  if [ "$used" -gt "$LIMIT_GB" ]; then kill -- -$pid; say "killed at $used GB host memory in use; partial parts kept in $OUT/parts (rerun resumes)"; exit 1; fi
  sleep 5
done
wait $pid
rc=$?
say "finished, exit $rc, peak host memory in use $peak GB, $(du -sh "$OUT" | cut -f1) in $OUT; timing $(tr -d ' \n' < timing.json 2>/dev/null)"
exit $rc
