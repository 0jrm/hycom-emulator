"""Arm switches for reanalysis-emulator training: our flags, taken off train_model's arguments before neural-lam parses them.

    python -m hycom_emulator.nlam train_model --config_path nlam.yaml ... --loss rea_wmse --mean_penalty 0.0016 \\
        --mean_scales truth_series.npz [--amse 0.1] [--pushforward 1] [--input_noise 0.1] [--checkpoint_steps] [--log_domain_means]

- --mean_penalty, --mean_scales, --amse: the weights of rea_loss.rea_wmse (needs --loss rea_wmse; --mean_penalty 0
  is the control). --mean_scales is a `conservation series` npz; its train rows give the scales.
- --pushforward K: the first K rollout steps run without gradient (Brandstetter et al. 2022), so step K learns from a
  state the model made itself instead of backpropagating through the whole chain.
- --input_noise a: Gaussian noise of a times each channel's one-day change std on the two initial states (interior).
- --checkpoint_steps: recompute each step's activations in the backward pass (memory for time).
- --log_domain_means: also log, in validation, the Gulf-mean error and squared error of each rea_loss quantity in SI
  units per step of --val_steps_to_log (`val_<quantity>_err_unroll<k>`, `val_<quantity>_sqerr_unroll<k>`).

Each switch is off by default and changes no weights, so checkpoints load with or without them.

    python -m hycom_emulator.rea_train probe --config_path nlam.yaml --load <ckpt> --ar_steps_train 4 --batch_size 4 \\
        --steps 10 [--num_workers 8] [--loss rea_wmse ...] [arm flags]

`probe` builds the model train_model would build for these flags, loads the checkpoint's weights and takes `steps`
optimizer steps on train batches. It prints one JSON line: peak GPU memory (GiB; null on CPU), the median time per
step and per data wait over the steps after the first, and the losses. It exits 1 on a non-finite loss.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from neural_lam.models import ARForecaster, ForecasterModule
from torch.utils.checkpoint import checkpoint

from hycom_emulator import rea_loss


@dataclass(frozen=True)
class Options:
    mean_penalty: float | None = None
    mean_scales: Path | None = None
    amse: float = 0.0
    pushforward: int = 0
    input_noise: float = 0.0
    checkpoint_steps: bool = False
    log_domain_means: bool = False


def _value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument(flag, default=default)
    return getattr(p.parse_known_args(argv)[0], flag.lstrip("-"))


def split_args(argv: list[str]) -> tuple[Options, list[str]]:
    """(our options, the arguments left for neural-lam). Exits with a usage error on an inconsistent set."""
    p = argparse.ArgumentParser(prog="hycom_emulator.rea_train", add_help=False, allow_abbrev=False)
    p.add_argument("--mean_penalty", type=float)
    p.add_argument("--mean_scales", type=Path)
    p.add_argument("--amse", type=float, default=0.0)
    p.add_argument("--pushforward", type=int, default=0)
    p.add_argument("--input_noise", type=float, default=0.0)
    p.add_argument("--checkpoint_steps", action="store_true")
    p.add_argument("--log_domain_means", action="store_true")
    a, rest = p.parse_known_args(argv)
    opts = Options(**vars(a))
    rea = _value(rest, "--loss", "wmse") == "rea_wmse"
    if rea and opts.mean_penalty is None:
        p.error("--loss rea_wmse needs --mean_penalty (0 for the control)")
    if not rea and (opts.mean_penalty is not None or opts.amse):
        p.error("--mean_penalty and --amse need --loss rea_wmse")
    if opts.mean_penalty and opts.mean_scales is None:
        p.error("--mean_penalty > 0 needs --mean_scales")
    if opts.log_domain_means and _value(rest, "--model", "graph_lam") == "crps_graph_lam":
        p.error("--log_domain_means replaces ForecasterModule, which crps_graph_lam's ensemble module also replaces")
    if min(opts.mean_penalty or 0.0, opts.amse, opts.pushforward, opts.input_noise) < 0:
        p.error("weights, --pushforward and --input_noise are >= 0")
    return opts, rest


def install(opts: Options, rest: list[str]) -> None:
    """Configure rea_wmse and swap neural-lam's forecaster and module classes for what opts asks."""
    import neural_lam.train_model as tm
    from neural_lam.config import load_config_and_datastore

    on = []
    if _value(rest, "--loss", "wmse") == "rea_wmse":
        _, ds = load_config_and_datastore(config_path=_value(rest, "--config_path"))
        means = inv_scale = None
        on.append(f"rea_wmse mean_penalty {opts.mean_penalty:g} amse {opts.amse:g}")
        if opts.mean_penalty:
            means = rea_loss.DomainMeans.from_meta(ds.meta)
            scales = rea_loss.scales_from_series(opts.mean_scales, ds.config["splits"]["train"][1])
            inv_scale = 1.0 / scales
            on[-1] += " (scales " + " ".join(f"{q} {s:.3g}" for q, s in zip(means.names, scales)) + ")"
        shape = ds.grid_shape_state
        rea_loss.configure(means, inv_scale, opts.mean_penalty, opts.amse, (shape.x, shape.y))
    if opts.pushforward or opts.input_noise or opts.checkpoint_steps:
        tm.ARForecaster = functools.partial(ReaForecaster, pushforward=opts.pushforward, input_noise=opts.input_noise,
                                            checkpoint_steps=opts.checkpoint_steps)
        on.append(f"ReaForecaster pushforward {opts.pushforward} input_noise {opts.input_noise:g} checkpoint_steps {opts.checkpoint_steps}")
    if opts.log_domain_means:
        tm.ForecasterModule = ReaForecasterModule
        on.append("validation logs Gulf-mean errors")
    print("rea_train: " + ("; ".join(on) if on else "no arm switch on"), flush=True)


class ReaForecaster(ARForecaster):
    """ARForecaster with the training-time switches; in eval mode it is ARForecaster exactly.

    Training unrolls like ARForecaster (boundary overwritten with the truth at each step), except:
    - the first `pushforward` steps run under no_grad, so step `pushforward` starts from a detached state the model
      made. All steps are returned and the loss still averages over all of them; the first ones add no gradient.
    - `input_noise` adds Gaussian noise of that many one-day change stds per channel to both initial states, on
      interior points only.
    - `checkpoint_steps` calls the predictor of each gradient step through activation checkpointing.
    """

    def __init__(self, predictor, datastore, pushforward: int = 0, input_noise: float = 0.0, checkpoint_steps: bool = False):
        super().__init__(predictor, datastore)
        self.pushforward, self.input_noise, self.checkpoint_steps = pushforward, input_noise, checkpoint_steps
        stats = datastore.get_standardization_dataarray("state")
        noise_std = input_noise * torch.tensor(stats.state_diff_std_standardized.values, dtype=torch.float32)
        self.register_buffer("noise_std", noise_std, persistent=False)

    def forward(self, init_states, forcing_features, boundary_states):
        if not self.training:
            return super().forward(init_states, forcing_features, boundary_states)
        pred_steps = forcing_features.shape[1]
        if self.pushforward >= pred_steps:
            raise ValueError(f"pushforward {self.pushforward} leaves no gradient step in a {pred_steps}-step rollout")
        if self.input_noise:
            init_states = init_states + torch.randn_like(init_states) * self.noise_std * self.interior_mask.unsqueeze(1)
        prev_prev_state, prev_state = init_states[:, 0], init_states[:, 1]
        predictions, stds = [], []
        for i in range(pred_steps):
            grad = i >= self.pushforward
            with nullcontext() if grad else torch.no_grad():
                args = (prev_state, prev_prev_state, forcing_features[:, i])
                if grad and self.checkpoint_steps:
                    pred_state, pred_std = checkpoint(self.predictor, *args, use_reentrant=False)
                else:
                    pred_state, pred_std = self.predictor(*args)
                new_state = self.boundary_mask * boundary_states[:, i] + self.interior_mask * pred_state
            predictions.append(new_state)
            if pred_std is not None:
                stds.append(pred_std)
            prev_prev_state, prev_state = prev_state, new_state
        return torch.stack(predictions, dim=1), torch.stack(stds, dim=1) if stds else None


class ReaForecasterModule(ForecasterModule):
    """ForecasterModule that also logs, in validation, the batch-mean Gulf-mean error and squared error of each
    rea_loss quantity (SI units) at each step of val_steps_to_log. No __init__ of its own: Lightning's
    save_hyperparameters reads ForecasterModule's."""

    _domain_means: rea_loss.DomainMeans | None = None

    def _compute_prediction_and_loss(self, batch):
        out = super()._compute_prediction_and_loss(batch)
        if self._trainer is not None and self._trainer.validating:
            self._log_domain_means(out[0], out[1])
        return out

    def _log_domain_means(self, prediction, target):
        if self._domain_means is None:
            self._domain_means = rea_loss.DomainMeans.from_meta(self.datastore.meta).to(prediction.device)
        e = self._domain_means.errors(prediction, target)  # (batch, step, quantity)
        logs = {}
        for k in self.hparams.val_steps_to_log:
            if k <= e.shape[1]:
                for q, name in enumerate(self._domain_means.names):
                    logs[f"val_{name}_err_unroll{k}"] = e[:, k - 1, q].mean()
                    logs[f"val_{name}_sqerr_unroll{k}"] = (e[:, k - 1, q] ** 2).mean()
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True, batch_size=e.shape[0])


def probe(argv: list[str]) -> int:
    import neural_lam.train_model as tm
    from neural_lam.config import load_config_and_datastore
    from neural_lam.models import MODELS
    from neural_lam.weather_dataset import WeatherDataset

    import hycom_emulator.convnet  # noqa: F401  registers our models
    import hycom_emulator.datastore  # noqa: F401  registers the hycom kind

    opts, rest = split_args(argv)
    p = argparse.ArgumentParser(prog="hycom_emulator.rea_train probe", allow_abbrev=False)
    p.add_argument("--config_path", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--ar_steps_train", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--loss", default="wmse")
    a = p.parse_args(rest)
    install(opts, rest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    ckpt = torch.load(a.load, map_location="cpu", weights_only=False)
    args = ckpt["hyper_parameters"]["args"]
    config, ds = load_config_and_datastore(config_path=a.config_path)
    forecaster = tm.ARForecaster(tm.build_predictor(MODELS[args.model], args, config, ds), ds)
    module = tm.ForecasterModule(forecaster=forecaster, config=config, datastore=ds, loss=a.loss, lr=args.lr)
    module.load_state_dict(ckpt["state_dict"])
    module = module.to(device).train()
    opt = module.configure_optimizers()
    data = WeatherDataset(ds, split="train", ar_steps=a.ar_steps_train, num_past_forcing_steps=args.num_past_forcing_steps,
                          num_future_forcing_steps=args.num_future_forcing_steps)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                                         multiprocessing_context="fork" if a.num_workers else None, pin_memory=device.type == "cuda")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batches, losses, waits, steps = iter(loader), [], [], []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        batch = next(batches, None)
        if batch is None:
            batches = iter(loader)
            batch = next(batches)
        batch = module.on_after_batch_transfer([x.to(device, non_blocking=True) for x in batch], 0)
        sync()
        t1 = time.perf_counter()
        loss = module._compute_prediction_and_loss(batch)[3].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync()
        waits.append(t1 - t0)
        steps.append(time.perf_counter() - t1)
        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            break
    gib = lambda f: round(f() / 2**30, 3) if device.type == "cuda" else None  # noqa: E731
    later = lambda v: statistics.median(v[1:]) if len(v) > 1 else None  # noqa: E731
    print(json.dumps({
        "ar_steps": a.ar_steps_train, "batch": a.batch_size, "loss": a.loss, "device": str(device),
        "flags": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated), "peak_reserved_gib": gib(torch.cuda.max_memory_reserved),
        "s_per_step": later(steps), "data_wait_s": later(waits), "losses": losses,
    }), flush=True)
    return 0 if all(map(math.isfinite, losses)) else 1


if __name__ == "__main__":
    if sys.argv[1:2] != ["probe"]:
        raise SystemExit("usage: python -m hycom_emulator.rea_train probe --config_path <nlam.yaml> --load <ckpt> "
                         "--ar_steps_train T --batch_size B --steps N [--num_workers W] [--loss ...] [arm flags]")
    sys.exit(probe(sys.argv[2:]))
