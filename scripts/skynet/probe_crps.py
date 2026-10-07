"""GPU memory and speed of crps_graph_lam training steps; nothing is saved.
usage: probe_crps.py <nlam.yaml> <graph_lam.ckpt> --members 2 --ar 4 --batch 4 [--steps 8] [--workers 4] [--checkpoint_steps]
The model takes the checkpoint's architecture and weights (as --init_from). Prints peak allocated and reserved
GPU memory, the median GPU time per optimizer step after two warm-up steps, and the median wait for data."""

import argparse
import statistics
import time

import torch
from neural_lam.config import load_config_and_datastore
from neural_lam.models import MODELS, ARForecaster
from neural_lam.train_model import build_predictor
from neural_lam.weather_dataset import WeatherDataset

import hycom_emulator.datastore  # noqa: F401
from hycom_emulator.ensemble import module_factory

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("ckpt")
p.add_argument("--members", type=int, default=2)
p.add_argument("--ar", type=int, default=4)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--steps", type=int, default=8)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--checkpoint_steps", action="store_true")
a = p.parse_args()

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
config, ds = load_config_and_datastore(config_path=a.config)
args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
forecaster = ARForecaster(build_predictor(MODELS["crps_graph_lam"], args, config, ds), ds)
module = module_factory(a.members, a.ckpt, a.checkpoint_steps)(forecaster=forecaster, config=config, datastore=ds, loss="afcrps").cuda().train()
opt = torch.optim.AdamW(module.parameters(), lr=1e-4, betas=(0.9, 0.95))
data = WeatherDataset(ds, split="train", ar_steps=a.ar, num_past_forcing_steps=1, num_future_forcing_steps=1)
loader = torch.utils.data.DataLoader(data, batch_size=a.batch, shuffle=True, num_workers=a.workers, multiprocessing_context="fork")

gpu, wait, t = [], [], time.perf_counter()
for i, batch in enumerate(loader):
    if i == a.steps:
        break
    wait.append(time.perf_counter() - t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = module.on_after_batch_transfer(tuple(x.cuda(non_blocking=True) for x in batch), 0)
    _, _, _, loss = module._compute_prediction_and_loss(batch)
    loss.mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    gpu.append(time.perf_counter() - t0)
    print(f"step {i} loss {loss.mean().item():.4f} gpu {gpu[-1]:.2f} s wait {wait[-1]:.2f} s", flush=True)
    t = time.perf_counter()

gib = 2**30
print(f"members {a.members} ar {a.ar} batch {a.batch} checkpoint_steps {a.checkpoint_steps}: peak allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
      f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB, gpu {statistics.median(gpu[2:]):.2f} s/step, "
      f"data wait {statistics.median(wait[2:]):.2f} s/step")
