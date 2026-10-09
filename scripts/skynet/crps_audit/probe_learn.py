"""Does crps_graph_lam learn? Short optimizer runs of the control (graph_lam, wmse) and the ensemble (crps_graph_lam,
afcrps) from the same start, scored every few steps on a fixed val subset with the same yardsticks: wmse of the
(ensemble-mean) forecast, wmse and wmae of one member, afCRPS, spread/skill. One JSON line per evaluation.
usage: probe_learn.py <nlam.yaml> <template.ckpt> --arm control|crps [--members 2] [--steps 300] [--eval_every 50]
       [--lr 1e-3] [--batch 4] [--ar 1] [--fp16] [--init_from <ckpt>] [--seed 0] [--val_n 24] [--compile] [--noise_scale 1] [--film update|state] [--mean_weight w]
The logs in logs/ were made with the pre-audit code (787d8d5), where --film input was that code and --film update this
fix, except fixed_*.log, made with this branch (9f2141d)."""

import argparse
import json
import sys
import time

import torch
from neural_lam import metrics
from neural_lam.config import load_config_and_datastore
from neural_lam.models import MODELS, ARForecaster, ForecasterModule
from neural_lam.train_model import build_predictor
from neural_lam.weather_dataset import WeatherDataset

import hycom_emulator.datastore  # noqa: F401
from hycom_emulator.ensemble import NOISE_DIM, CRPSGraphLAM, crps_ensemble, module_factory


def process_step_state(self, mesh_rep):
    """The modulation before the audit (fd3f082): the noise scaled the mesh state before every layer."""
    batch_size = mesh_rep.shape[0]
    z = self.noise_scale * torch.randn(batch_size, 1, NOISE_DIM, device=mesh_rep.device, dtype=mesh_rep.dtype)
    edge_rep = self.expand_to_batch(self.m2m_embedder(self.m2m_features), batch_size)
    for net, film in zip(self.processor.children(), self.film):
        scale, shift = film(z).chunk(2, dim=-1)
        mesh_rep = mesh_rep * (1 + scale) + shift
        mesh_rep, edge_rep = net(mesh_rep, mesh_rep, edge_rep)
    return mesh_rep

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("template")
p.add_argument("--arm", choices=("control", "crps"), required=True)
p.add_argument("--members", type=int, default=2)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--ar", type=int, default=1)
p.add_argument("--fp16", action="store_true")
p.add_argument("--init_from", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--val_n", type=int, default=24)
p.add_argument("--compile", action="store_true")
p.add_argument("--noise_scale", type=float, default=1.0)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--film", choices=("update", "state"), default="update", help="state: the pre-audit modulation of the mesh state")
p.add_argument("--mean_weight", type=float, default=0.0, help="crps: add this times the wmse of the z=0 forecast")
a = p.parse_args()

if a.film == "state":
    CRPSGraphLAM.process_step = process_step_state
torch.manual_seed(a.seed)
torch.set_float32_matmul_precision("high")
config, ds = load_config_and_datastore(config_path=a.config)
args = torch.load(a.template, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
name = "graph_lam" if a.arm == "control" else "crps_graph_lam"
forecaster = ARForecaster(build_predictor(MODELS[name], args, config, ds), ds)
if a.arm == "control":
    module = ForecasterModule(forecaster=forecaster, config=config, datastore=ds, loss="wmse", lr=a.lr)
    if a.init_from:
        module.load_state_dict(torch.load(a.init_from, map_location="cpu", weights_only=False)["state_dict"], strict=True)
else:
    module = module_factory(a.members, a.init_from, False)(forecaster=forecaster, config=config, datastore=ds, loss="afcrps", lr=a.lr)
    module.forecaster.predictor.noise_scale = a.noise_scale
module = module.cuda().train()
if a.compile:
    module.forecaster.predictor.compile()
opt = torch.optim.AdamW(module.parameters(), lr=a.lr, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)

train = WeatherDataset(ds, split="train", ar_steps=a.ar, num_past_forcing_steps=1, num_future_forcing_steps=1)
loader = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, multiprocessing_context="fork", drop_last=True)
val = WeatherDataset(ds, split="val", ar_steps=a.ar, num_past_forcing_steps=1, num_future_forcing_steps=1)
val_idx = torch.linspace(0, len(val) - 1, a.val_n).round().long().tolist()
val_batches = [tuple(torch.stack([torch.as_tensor(val[i][k]) for i in val_idx[j:j + a.batch]]) for k in range(3)) for j in range(0, a.val_n, a.batch)]
std, mask = module.per_var_std, module.interior_mask_bool


def to_dev(b):
    return module.on_after_batch_transfer(tuple(x.cuda(non_blocking=True) for x in b[:3]) + (None,), 0)


def score(prediction, target):
    return {"wmse": metrics.wmse(prediction, target, std, mask=mask).mean().item(),
            "wmae": metrics.wmae(prediction, target, std, mask=mask).mean().item()}


@torch.no_grad()
def evaluate(step):
    module.eval()
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)
    acc = {}
    for b in val_batches:
        init, target, forcing, _ = to_dev(b)
        if a.arm == "control":
            pred, _ = module.forecaster(init, forcing, target)
            out = score(pred, target)
        else:
            ens = module.forecast_members(init, forcing, target, 8)
            out = {f"mean8_{k}": v for k, v in score(ens.mean(1), target).items()}
            out |= {f"mean{a.members}_{k}": v for k, v in score(ens[:, :a.members].mean(1), target).items()}
            out |= {f"member_{k}": v for k, v in score(ens[:, 0], target).items()}
            out["afcrps2"] = crps_ensemble(ens[:, :2].transpose(1, 2), target, std, mask=mask).mean().item()
            out["afcrps8"] = crps_ensemble(ens.transpose(1, 2), target, std, mask=mask).mean().item()
            out["spread_skill8"] = module.spread_skill(ens, target).item()
            module.forecaster.predictor.noise_scale = 0.0
            det, _ = module.forecaster(init, forcing, target)
            module.forecaster.predictor.noise_scale = a.noise_scale
            out |= {f"z0_{k}": v for k, v in score(det, target).items()}
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + v / len(val_batches)
    torch.random.set_rng_state(g)
    module.train()
    film = [layer.weight.norm().item() for layer in module.forecaster.predictor.film] if a.arm == "crps" else None
    print(json.dumps({"step": step, "arm": a.arm, "members": a.members, "fp16": a.fp16, "lr": a.lr, "ar": a.ar,
                      "film": a.film, "mean_weight": a.mean_weight, "film_norms": film, "scale": scaler.get_scale() if a.fp16 else None, **{k: round(v, 5) for k, v in acc.items()}}), flush=True)


evaluate(0)
step, losses, t0, nonfinite = 0, [], time.perf_counter(), 0
while step < a.steps:
    for b in loader:
        if step >= a.steps:
            break
        batch = to_dev(b)
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            loss = module._compute_prediction_and_loss(batch)[3].mean()
            if a.mean_weight:
                module.forecaster.predictor.noise_scale = 0.0
                det, _ = module.forecaster(batch[0], batch[2], batch[1])
                module.forecaster.predictor.noise_scale = a.noise_scale
                loss = loss + a.mean_weight * metrics.wmse(det, batch[1], std, mask=mask).mean()
        if not torch.isfinite(loss):
            nonfinite += 1
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        step += 1
        if step % a.eval_every == 0:
            print(json.dumps({"step": step, "train_loss_mean_last": round(sum(losses[-a.eval_every:]) / a.eval_every, 4),
                              "nonfinite": nonfinite, "s_per_step": round((time.perf_counter() - t0) / step, 3)}), flush=True)
            evaluate(step)
print(json.dumps({"done": step, "nonfinite": nonfinite, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
