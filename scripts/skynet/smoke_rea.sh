#!/bin/bash
# CPU smoke of the reanalysis pipeline on skynet. usage: smoke_rea.sh <code_dir> <out_dir>
# Builds 10-day packs at stride 1, 2 and 4 from a frozen code checkout, on different days so the page
# cache does not flatter the later timings. The stride-4 window holds the 2018-05-01..02 gap. It checks
# them, then trains graph_lam (tiny) on the stride-4 pack with 2-step rollouts and evaluates a 2-step
# rollout on test. The splits overlap: this proves the datastore and rollout path, not skill. No GPU.
set -euo pipefail
CODE=$1
OUT=$2
PY=${PY:-/conda/jmiranda/venvs/hycom-emulator/bin/python}
export PYTHONPATH=$CODE/src:/conda/jmiranda/pylib-netcdf4 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
export MLFLOW_TRACKING_URI=sqlite:///$OUT/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p "$OUT"
cd "$OUT"
S1=(s1.pack --start 2024-08-01 --end 2024-08-10 --train-end 2024-08-07 --stride 1)
S2=(s2.pack --start 2024-07-01 --end 2024-07-10 --train-end 2024-07-07 --stride 2)
S4=(s4.pack --start 2018-04-27 --end 2018-05-06 --train-end 2018-05-03 --stride 4)
peak_rss() {  # summed RSS of a process and its children, sampled until it exits
  local m=0 r
  while kill -0 "$1" 2>/dev/null; do
    r=$(ps --no-headers -o rss -p "$1" --ppid "$1" | awk '{s += $1} END {print s + 0}')
    (( r > m )) && m=$r
    sleep 0.5
  done
  echo "peak RSS $(( m / 1024 )) MB"
}
for args in "S1[@]" "S2[@]" "S4[@]"; do
  echo "== $(date -Is) build ${!args}"
  nice $PY -m hycom_emulator.rea_pack build "${!args}" &
  peak_rss $!
  wait $!
done
echo "== $(date -Is) rebuild stride 1 (must read nothing)"
nice $PY -m hycom_emulator.rea_pack build "${S1[@]}"

for p in s1.pack s4.pack; do
echo "== $(date -Is) check $p"
$PY - $p <<'PY'
import sys
import numpy as np
import xarray as xr
p = sys.argv[1]
m = xr.open_zarr(f"{p}/meta.zarr", consolidated=True, chunks=None).load()
st = np.load(f"{p}/state.npy", mmap_mode="r")
fo = np.load(f"{p}/forcing.npy", mmap_mode="r")
print("state", st.shape, "forcing", fo.shape, "static", m["static"].shape)
print("finite", bool(np.isfinite(st).all() and np.isfinite(fo).all()), "max |state|", float(np.abs(st).max()))
names = m["state_feature"].values.tolist()
lev = m["level_ocean"].values
ocean = m["static"].sel(static_feature="ocean").values.astype(bool)
for i, n in enumerate(names):
    v, _, d = n.rpartition("_")
    mask = lev[:, list(m["level"].values).index(float(d[:-1]))].astype(bool) if d.endswith("m") and v else ocean
    x = st[:, mask, i]
    print(f"{n:12s} min {x.min():9.4g} max {x.max():9.4g} mean {m['state_mean'].values[i]:9.4g} std {m['state_std'].values[i]:9.4g} diff_std {m['state_diff_std'].values[i]:9.4g}")
for i, n in enumerate(m["forcing_feature"].values):
    x = fo[:, ocean, i]
    print(f"{n:12s} min {x.min():9.4g} max {x.max():9.4g} mean {m['forcing_mean'].values[i]:9.4g} std {m['forcing_std'].values[i]:9.4g}")
print("level ocean fraction", dict(zip(m["level"].values.tolist(), np.round(lev.mean(axis=0), 3).tolist())))
b = m["boundary_mask"].values.astype(bool)
print("ocean frac", ocean.mean(), "level 0 == ocean", bool((lev[:, 0].astype(bool) == ocean).all()),
      "boundary ocean pts", int((b & ocean).sum()), "time_filled", m["time_filled"].values.tolist())
gulf = m["static"].sel(static_feature="gulf").values.astype(bool)
x, y = m["x"].values, m["y"].values
print("gulf pts", int(gulf.sum()), "lon", x[gulf].min(), x[gulf].max(), "lat", y[gulf].min(), y[gulf].max(),
      "gulf pts south of 21.5N east of 87W", int((gulf & (y < 21.5) & (x > -87)).sum()), "east of 81W", int((gulf & (x > -81)).sum()))
if m.attrs["stride"] != 1:
    sys.exit()
ref = "/scratch/jmiranda/hycom-emulator-data/abozec_053_b00atm.pack/meta.zarr"
r = xr.open_zarr(ref, consolidated=True, chunks=None)
rb = r["boundary_mask"].values.astype(bool) & r["static"].sel(static_feature="ocean").values.astype(bool)
same_grid = np.allclose(r["x"].values, x) and np.allclose(r["y"].values, y, atol=1e-4)
print("B00 grid matches", same_grid, "band Jaccard vs B00", float((rb & b & ocean).sum() / (rb | (b & ocean)).sum()))
PY
done

echo "== $(date -Is) neural-lam on the stride-4 pack"
cat > rea.yaml <<YAML
zarr: $OUT/s4.pack
splits:
  train: [2018-04-27, 2018-05-06]
  val: [2018-04-27, 2018-05-06]
  test: [2018-04-27, 2018-05-06]
YAML
printf 'datastore:\n  kind: hycom\n  config_path: rea.yaml\n' > nlam.yaml
nice $PY -m hycom_emulator.nlam build_graph nlam.yaml multiscale
MODEL=(--model graph_lam --graph multiscale --hidden_dim 8 --processor_layers 1 --batch_size 2 --ar_steps_eval 2
       --val_steps_to_log 1 2 --n_example_pred 0 --num_workers 2 --logger mlflow --runs_root "$OUT/runs")
nice $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --epochs 2 --ar_steps_train 2 --logger_run_name rea-smoke
CKPT=$(find "$OUT/runs" -name last.ckpt | head -1)
echo "== $(date -Is) evaluate $CKPT"
nice $PY -m hycom_emulator.nlam train_model --config_path nlam.yaml "${MODEL[@]}" --eval test --load "$CKPT" --logger_run_name rea-smoke-eval
echo "== $(date -Is) done"
