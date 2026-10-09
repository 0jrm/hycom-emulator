"""Where does the fp16 forward (no_grad, train mode) of a trained crps checkpoint go non-finite? Train-mode 8-day batches under
autocast fp16 (Lightning 16-mixed), hooks on the predictor's film and processor layers report the max |activation|
and the first non-finite tensor. usage: probe_nan.py <nlam.yaml> <ckpt> [--batches 12] [--ar 8] [--batch 4]"""

import argparse
import json

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
p.add_argument("--batches", type=int, default=12)
p.add_argument("--ar", type=int, default=8)
p.add_argument("--batch", type=int, default=4)
a = p.parse_args()
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
config, ds = load_config_and_datastore(config_path=a.config)
args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
forecaster = ARForecaster(build_predictor(MODELS["crps_graph_lam"], args, config, ds), ds)
module = module_factory(2, a.ckpt, False)(forecaster=forecaster, config=config, datastore=ds, loss="afcrps").cuda().train()
pred = module.forecaster.predictor
stats = {}


def watch(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        m = t.detach().abs().max().item() if t.numel() else 0.0
        s = stats.setdefault(name, {"max": 0.0, "nonfinite": 0, "dtype": str(t.dtype)})
        s["max"] = max(s["max"], m) if m == m else s["max"]
        if not torch.isfinite(t).all():
            s["nonfinite"] += 1
    return hook


for i, net in enumerate(pred.processor.children()):
    net.register_forward_hook(watch(f"processor.{i}"))
for i, f in enumerate(pred.film):
    f.register_forward_hook(watch(f"film.{i}"))
pred.g2m_gnn.register_forward_hook(watch("g2m_gnn"))
pred.m2g_gnn.register_forward_hook(watch("m2g_gnn"))
pred.output_map.register_forward_hook(watch("output_map"))

train = WeatherDataset(ds, split="train", ar_steps=a.ar, num_past_forcing_steps=1, num_future_forcing_steps=1)
loader = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=4, multiprocessing_context="fork")
for i, b in enumerate(loader):
    if i == a.batches:
        break
    batch = module.on_after_batch_transfer(tuple(x.cuda(non_blocking=True) for x in b), 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pr, _, _, loss = module._compute_prediction_and_loss(batch)
    print(json.dumps({"batch": i, "loss": [round(v, 3) for v in loss.tolist()], "pred_finite": torch.isfinite(pr).all().item(),
                      "pred_max": pr.abs().max().item()}), flush=True)
print(json.dumps(stats, indent=1))
stats.clear()
print("== same batches, fp32")
for i, b in enumerate(loader):
    if i == 4:
        break
    batch = module.on_after_batch_transfer(tuple(x.cuda(non_blocking=True) for x in b), 0)
    with torch.no_grad():
        pr, _, _, loss = module._compute_prediction_and_loss(batch)
    print(json.dumps({"batch": i, "loss": [round(v, 3) for v in loss.tolist()], "pred_max": pr.abs().max().item()}), flush=True)
print(json.dumps(stats, indent=1))
