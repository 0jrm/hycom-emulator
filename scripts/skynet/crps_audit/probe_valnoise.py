"""How much of val_mean_loss is the noise draw? The 4-day afCRPS (2 members) of one checkpoint over the val split,
repeated with different seeds. usage: probe_valnoise.py <nlam.yaml> <ckpt> [--seeds 3] [--limit N] [--batch 8]"""

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
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--batch", type=int, default=8)
a = p.parse_args()
torch.set_float32_matmul_precision("high")
config, ds = load_config_and_datastore(config_path=a.config)
args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
forecaster = ARForecaster(build_predictor(MODELS["crps_graph_lam"], args, config, ds), ds)
module = module_factory(2, a.ckpt, False)(forecaster=forecaster, config=config, datastore=ds, loss="afcrps").cuda().eval()
val = WeatherDataset(ds, split="val", ar_steps=4, num_past_forcing_steps=1, num_future_forcing_steps=1)
if a.limit:
    val = torch.utils.data.Subset(val, list(range(0, len(val), max(1, len(val) // a.limit)))[:a.limit])
loader = torch.utils.data.DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=4, multiprocessing_context="fork")
for seed in range(a.seeds):
    torch.manual_seed(seed)
    total, n = torch.zeros(4, device="cuda"), 0
    with torch.no_grad():
        for b in loader:
            batch = module.on_after_batch_transfer(tuple(x.cuda(non_blocking=True) for x in b), 0)
            init, target, forcing, _ = batch
            ens = module.forecast_members(init, forcing, target, 2)
            step_loss = module.loss(ens.transpose(1, 2), target, module.per_var_std, mask=module.interior_mask_bool).mean(0)
            total += step_loss * batch[0].shape[0]
            n += batch[0].shape[0]
    per_step = (total / n).tolist()
    print(json.dumps({"seed": seed, "samples": n, "val_mean_loss": round(sum(per_step) / 4, 5), "per_step": [round(v, 4) for v in per_step]}), flush=True)
