"""Arm switches for reanalysis-emulator training: our flags, taken off train_model's arguments before neural-lam parses them.

    python -m hycom_emulator.nlam train_model --config_path nlam.yaml ... --loss rea_wmse --mean_penalty 0.0016 \\
        --mean_scales truth_series.npz [--amse 0.1] [--pushforward 1] [--input_noise 0.1] [--checkpoint_steps]

- --mean_penalty, --mean_scales, --amse: the weights of rea_loss.rea_wmse (needs --loss rea_wmse; --mean_penalty 0
  is the control). --mean_scales is a `conservation series` npz; its train rows give the scales.
- --pushforward K: the first K rollout steps run without gradient (Brandstetter et al. 2022), so step K learns from a
  state the model made itself instead of backpropagating through the whole chain.
- --input_noise a: Gaussian noise of a times each channel's one-day change std on the two initial states (interior).
- --checkpoint_steps: recompute each step's activations in the backward pass (memory for time).

Each switch is off by default and changes no weights, so checkpoints load with or without them.
"""

from __future__ import annotations

import argparse
import functools
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from neural_lam.models import ARForecaster
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hycom_emulator.rea_train", add_help=False, allow_abbrev=False)
    p.add_argument("--mean_penalty", type=float)
    p.add_argument("--mean_scales", type=Path)
    p.add_argument("--amse", type=float, default=0.0)
    p.add_argument("--pushforward", type=int, default=0)
    p.add_argument("--input_noise", type=float, default=0.0)
    p.add_argument("--checkpoint_steps", action="store_true")
    return p


def _value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument(flag, default=default)
    return getattr(p.parse_known_args(argv)[0], flag.lstrip("-"))


def split_args(argv: list[str]) -> tuple[Options, list[str]]:
    """(our options, the arguments left for neural-lam). Exits with a usage error on an inconsistent set."""
    p = _parser()
    a, rest = p.parse_known_args(argv)
    opts = Options(**vars(a))
    rea = _value(rest, "--loss", "wmse") == "rea_wmse"
    if rea and opts.mean_penalty is None:
        p.error("--loss rea_wmse needs --mean_penalty (0 for the control)")
    if not rea and (opts.mean_penalty is not None or opts.amse):
        p.error("--mean_penalty and --amse need --loss rea_wmse")
    if opts.mean_penalty and opts.mean_scales is None:
        p.error("--mean_penalty > 0 needs --mean_scales")
    if min(opts.mean_penalty or 0.0, opts.amse, opts.pushforward, opts.input_noise) < 0:
        p.error("weights, --pushforward and --input_noise are >= 0")
    return opts, rest


def install(opts: Options, rest: list[str]) -> None:
    """Configure rea_wmse and swap neural-lam's forecaster class for what opts asks."""
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
        self.pushforward, self.checkpoint_steps = pushforward, checkpoint_steps
        stats = datastore.get_standardization_dataarray("state")
        noise_std = input_noise * torch.tensor(stats.state_diff_std_standardized.values, dtype=torch.float32)
        self.register_buffer("noise_std", noise_std, persistent=False)
        self.input_noise = input_noise

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
