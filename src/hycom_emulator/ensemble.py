"""A stochastic GraphLAM trained with an ensemble CRPS (card arm 2 of the reanalysis emulator).

Each step draws one noise vector z ~ N(0, I_32) per ensemble member. Each processor layer's update d (the
layer-normed output it adds to the mesh representation h) is modulated as h + d * (1 + W_s z) + W_b z (FiLM on
the update, as the conditional layer norms of AIFS-CRPS and FGN, Alet et al. 2025). W_s and W_b have no bias and
start at zero, so the model starts as the GraphLAM checkpoint it loads, and z = 0 reproduces that GraphLAM with
the current weights. z is global: it is the same at every mesh node, and the processor turns it into spatially
structured perturbations.

Members ride on the batch axis. EnsembleForecasterModule repeats each sample M times, unrolls
neural-lam's ARForecaster unchanged (the true boundary band overwrites every member), and scores the M
forecasts with the almost-fair CRPS of AIFS-CRPS (Lang et al. 2024):

    afCRPS = mean_j |x_j - y| - (1 - (1 - alpha) / M) / (2 M (M - 1)) sum_{j != k} |x_j - x_k|

alpha = 1 is the fair CRPS, alpha = 0 the plain ensemble CRPS. Each entry is divided by neural-lam's
per_var_std (the 1-day change std, as in wmae) and reduced like wmse: mean over interior grid points,
sum over channels. Validation and test metrics see the ensemble mean, and draw the same noise at every validation
(EVAL_SEED), so val_mean_loss compares weights.

    train_model --model crps_graph_lam --loss afcrps --members 2 --init_from <graph_lam.ckpt>

`--members` (default 2), `--init_from` and `--checkpoint_steps` are ours: hycom_emulator.nlam strips them
before neural-lam parses the rest. `--init_from` loads a checkpoint's weights only, so training starts at
epoch 0; `--load` keeps its usual meaning (weights and epoch) and also takes a graph_lam checkpoint.
`--checkpoint_steps` recomputes each step's activations in the backward pass: GPU memory then grows with the
rollout only by the saved states, which long (8-21 day) rollouts need. Importing this module registers `crps_graph_lam`,
`afcrps` (alpha 0.95) and `fcrps` (alpha 1).
"""

from __future__ import annotations

import argparse
from functools import partial

import torch
from neural_lam import metrics
from neural_lam.models import MODELS, ForecasterModule
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM
from torch import nn
from torch.utils.checkpoint import checkpoint

NOISE_DIM = 32
EVAL_SEED = 31
FILM = "film."
LOSSES = ("afcrps", "fcrps")


class CRPSGraphLAM(GraphLAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film = nn.ModuleList(nn.Linear(NOISE_DIM, 2 * self.hidden_dim, bias=False) for _ in self.processor.children())
        for layer in self.film:
            nn.init.zeros_(layer.weight)
        self.noise_scale = 1.0
        self.checkpoint_steps = False
        self.register_load_state_dict_pre_hook(_fill_film)

    def forward(self, prev_state, prev_prev_state, forcing):
        """With checkpoint_steps, a step's activations are recomputed in the backward pass (with the same noise:
        checkpoint restores the RNG state), so a rollout holds one step's activations instead of all of them."""
        if self.checkpoint_steps and torch.is_grad_enabled():
            return checkpoint(super().forward, prev_state, prev_prev_state, forcing, use_reentrant=False)
        return super().forward(prev_state, prev_prev_state, forcing)

    def process_step(self, mesh_rep):
        """Each processor layer's update (its layer-normed MLP output) is modulated, not the residual stream it is
        added to: the noise then enters additively and stays bounded by the update's scale. Modulating the stream
        itself compounded through the layers (mesh values 200-300x the deterministic model's after four layers)
        and overflowed fp16 on 0.1-2% of the training steps."""
        batch_size = mesh_rep.shape[0]
        z = self.noise_scale * torch.randn(batch_size, 1, NOISE_DIM, device=mesh_rep.device, dtype=mesh_rep.dtype)
        edge_rep = self.expand_to_batch(self.m2m_embedder(self.m2m_features), batch_size)
        for net, film in zip(self.processor.children(), self.film):
            scale, shift = film(z).chunk(2, dim=-1)
            new_rep, edge_rep = net(mesh_rep, mesh_rep, edge_rep)
            mesh_rep = mesh_rep + (new_rep - mesh_rep) * (1 + scale) + shift
        return mesh_rep


def _fill_film(module, state_dict, prefix, *_):
    """A GraphLAM state dict (no FiLM weights) gets the zero FiLM weights: the start is that GraphLAM."""
    own = {k: v for k, v in module.state_dict().items() if k.startswith(FILM)}
    if not any(prefix + k in state_dict for k in own):
        state_dict.update({prefix + k: torch.zeros_like(v) for k, v in own.items()})


def crps_ensemble(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, alpha=0.95):
    """pred (..., M, N, F) members on the axis before the grid, or (..., N, F) as one member (then this is
    wmae). target (..., N, F); pred_std (F,) or broadcastable. Reduced as neural-lam's metrics."""
    if pred.ndim == target.ndim:
        pred = pred.unsqueeze(-3)
    m = pred.shape[-3]
    skill = (pred - target.unsqueeze(-3)).abs().mean(-3)
    if m == 1:
        entry = skill
    else:
        ranked = pred.sort(dim=-3).values  # sum_{j<k} |x_j - x_k| = sum_i (2i - M + 1) x_(i)
        coef = (2 * torch.arange(m, device=pred.device, dtype=pred.dtype) - m + 1).view(m, 1, 1)
        pair_sum = (coef * ranked).sum(-3)
        entry = skill - (1 - (1 - alpha) / m) * pair_sum / (m * (m - 1))
    return metrics.mask_and_reduce_metric(entry / pred_std, mask=mask, average_grid=average_grid, sum_vars=sum_vars)


class EnsembleForecasterModule(ForecasterModule):
    def __init__(self, *args, members: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.forecaster.predictor, CRPSGraphLAM):
            raise ValueError(f"an ensemble needs a noisy predictor (crps_graph_lam), not {type(self.forecaster.predictor).__name__}")
        if self.loss not in (metrics.DEFINED_METRICS[n] for n in LOSSES):
            raise ValueError(f"an ensemble trains with --loss {' or '.join(LOSSES)}")
        if members < 2:
            raise ValueError("--members must be at least 2")
        self.members = members

    def on_validation_start(self):
        """Validation and test draw the same noise every time, so val_mean_loss compares weights, not draws: with 2
        members one checkpoint's val_mean_loss moved 1% between draws, twice the plateau threshold. The training
        stream (noise, the next epoch's shuffle) resumes where it was."""
        self._training_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
        torch.manual_seed(EVAL_SEED)

    def on_validation_end(self):
        torch.random.set_rng_state(self._training_rng[0])
        torch.cuda.set_rng_state_all(self._training_rng[1])

    on_test_start, on_test_end = on_validation_start, on_validation_end

    def forecast_members(self, init_states, forcing, boundary_states, members: int):
        """(B, M, T, N, F) standardized forecasts: member j of sample b is batch row b * M + j."""
        rep = lambda x: x.repeat_interleave(members, dim=0)  # noqa: E731
        prediction, _ = self.forecaster(rep(init_states), rep(forcing), rep(boundary_states))
        return prediction.unflatten(0, (init_states.shape[0], members))

    def _compute_prediction_and_loss(self, batch):
        init_states, target_states, forcing, _ = batch
        ens = self.forecast_members(init_states, forcing, target_states, self.members)
        time_step_loss = self.loss(ens.transpose(1, 2), target_states, self.per_var_std, mask=self.interior_mask_bool).mean(0)
        if not self.training:
            self.log(f"{'test' if self.trainer.testing else 'val'}_spread_skill", self.spread_skill(ens, target_states),
                     on_epoch=True, sync_dist=True, batch_size=target_states.shape[0])
        return ens.mean(1), target_states, self.per_var_std, time_step_loss

    def spread_skill(self, ens, target):
        """sqrt((M + 1) / M x member variance) over the RMSE of the mean, both weighted like the loss: 1 when
        the truth is statistically one more member (Fortin et al. 2014)."""
        m = ens.shape[1]
        weigh = lambda x: metrics.mask_and_reduce_metric(x / self.per_var_std**2, self.interior_mask_bool, True, True)  # noqa: E731
        return (weigh(ens.var(1) * (m + 1) / m).mean() / weigh((ens.mean(1) - target) ** 2).mean()).sqrt()


def module_factory(members: int, init_from: str | None, checkpoint_steps: bool = False):
    """What train_model calls in place of ForecasterModule (it passes keywords only)."""

    def build(**kwargs):
        module = EnsembleForecasterModule(**kwargs, members=members)
        module.forecaster.predictor.checkpoint_steps = checkpoint_steps
        if init_from:
            state = torch.load(init_from, map_location="cpu", weights_only=False)["state_dict"]
            module.load_state_dict(state, strict=True)
        return module

    return build


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our flags out of a train_model command line; the rest goes to neural-lam."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--members", type=int, default=2)
    p.add_argument("--init_from", default=None)
    p.add_argument("--checkpoint_steps", action="store_true")
    p.add_argument("--model", default="graph_lam")
    ours, rest = p.parse_known_args(argv)
    if ours.model != "crps_graph_lam" and ({"--members", "--init_from", "--checkpoint_steps"} & {a.split("=")[0] for a in argv}):
        raise SystemExit("--members, --init_from and --checkpoint_steps need --model crps_graph_lam")
    return ours, rest + ["--model", ours.model]


MODELS["crps_graph_lam"] = CRPSGraphLAM
metrics.DEFINED_METRICS["afcrps"] = partial(crps_ensemble, alpha=0.95)
metrics.DEFINED_METRICS["fcrps"] = partial(crps_ensemble, alpha=1.0)
