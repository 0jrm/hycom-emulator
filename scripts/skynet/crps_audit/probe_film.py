"""FiLM weight norms of crps checkpoints; member spread, compile and fp16 sanity of the trained predictor.
usage: probe_film.py <nlam.yaml> <ckpt>... (run from the probe dir; the first ckpt drives the model probes)"""

import sys

import torch
from neural_lam.config import load_config_and_datastore
from neural_lam.models import MODELS, ARForecaster
from neural_lam.train_model import build_predictor
from neural_lam.weather_dataset import WeatherDataset

import hycom_emulator.datastore  # noqa: F401
from hycom_emulator.ensemble import module_factory

config_path, ckpts = sys.argv[1], sys.argv[2:]
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

for path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print(f"== {path} epoch {ck.get('epoch')} step {ck.get('global_step')}")
    for k in sorted(sd):
        if ".film." in k:
            w = sd[k].float()
            h = w.shape[0] // 2
            scale, shift = w[:h], w[h:]
            # per-output std of W z for z ~ N(0, I) is the row norm; report the rms over outputs
            print(f"  {k}: |W|={w.norm():.4f} rms scale={scale.norm(dim=1).pow(2).mean().sqrt():.4f} "
                  f"max scale row={scale.norm(dim=1).max():.4f} rms shift={shift.norm(dim=1).pow(2).mean().sqrt():.4f}")
    ref = [k for k in sd if "processor" in k and k.endswith("weight") and sd[k].ndim == 2][:2]
    for k in ref:
        print(f"  ref {k}: |W|={sd[k].float().norm():.4f} shape {tuple(sd[k].shape)}")

config, ds = load_config_and_datastore(config_path=config_path)
args = torch.load(ckpts[0], map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
forecaster = ARForecaster(build_predictor(MODELS["crps_graph_lam"], args, config, ds), ds)
module = module_factory(4, ckpts[0], False)(forecaster=forecaster, config=config, datastore=ds, loss="afcrps").cuda().eval()
data = WeatherDataset(ds, split="val", ar_steps=4, num_past_forcing_steps=1, num_future_forcing_steps=1)
batch = tuple(torch.as_tensor(x).unsqueeze(0).cuda() for x in data[10])
batch = module.on_after_batch_transfer(batch, 0)
init, target, forcing, _ = batch
std = module.per_var_std
mask = module.interior_mask_bool


def spread_report(tag, ens):
    """ens (B, M, T, N, F) standardized. Member spread and ensemble-mean error per step, in per_var_std units."""
    m = ens.shape[1]
    for t in range(ens.shape[2]):
        e = ens[:, :, t][..., mask, :]
        y = target[:, t][..., mask, :]
        sp = ((e.var(1) * (m + 1) / m) / std**2).mean(-2).sum(-1).mean().sqrt()
        rm = (((e.mean(1) - y) ** 2) / std**2).mean(-2).sum(-1).mean().sqrt()
        md = (e[:, 0] - e[:, 1]).abs().max()
        print(f"  {tag} step {t + 1}: spread {sp:.4f} rmse_mean {rm:.4f} spread/skill {sp / rm:.3f} max|m0-m1| {md:.4e}")


with torch.no_grad():
    ens = module.forecast_members(init, forcing, target, 4)
    spread_report("eager", ens)
    ens2 = module.forecast_members(init, forcing, target, 4)
    print(f"  eager: two calls differ by max {(ens - ens2).abs().max():.4e}")
    module.forecaster.predictor.noise_scale = 0.0
    ens0 = module.forecast_members(init, forcing, target, 2)
    print(f"  z=0: members differ by max {(ens0[:, 0] - ens0[:, 1]).abs().max():.4e}")
    module.forecaster.predictor.noise_scale = 1.0

    module.forecaster.predictor.compile()
    ensc = module.forecast_members(init, forcing, target, 4)
    spread_report("compiled", ensc)
    ensc2 = module.forecast_members(init, forcing, target, 4)
    print(f"  compiled: two calls differ by max {(ensc - ensc2).abs().max():.4e}")

    with torch.autocast("cuda", dtype=torch.float16):
        ensh = module.forecast_members(init, forcing, target, 4)
    print(f"  compiled fp16 autocast: finite {torch.isfinite(ensh).all().item()} dtype {ensh.dtype}")
    spread_report("fp16", ensh)
