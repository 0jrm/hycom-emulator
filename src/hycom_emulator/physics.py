"""Physical structure for B00 training: a thickness projection, a reshaped loss and the ablation arms.

HYCOM's layer thicknesses are never negative and each column sums to the bottom depth (within 1 mm
in the 05.3 archives). B00 trained with neural-lam's channel-wise wmse breaks both (explore-physcheck,
2026-10-02): 11% of (point, layer) thicknesses below -1 cm, column sums off by 0.4 m RMS.

- project_thickness: relu on every layer thickness; a column's excess is clipped from the bottom, and
  a shortfall is spread over layers with mass by their 24 h change variance (the least-squares
  correction under wmse); exact, differentiable.
  `hycom_graph_lam` is GraphLAM with this projection after neural-lam's residual update; it has
  GraphLAM's weights, so it loads GraphLAM checkpoints. Use it at prediction time only
  (evaluate_b00 --model hycom_graph_lam): on E1 it costs no skill and removes the negative
  thicknesses and column errors, while training through it diverged (emu-b00-053-phys and the
  explore-phys2-gate run, 2026-10-02/03): closing a column makes one layer absorb the summed error
  of the others, and on the shelf every layer has a tiny change std, so wmse's gradients explode.
- hycom_wmse: neural-lam's wmse with T, S, u and v layer entries weighted by the true layer thickness
  relative to the column mean (empty layers weigh 0, as in evaluate_b00), plus optional penalties on
  sigma2 error (couples T and S, holds isopycnal layers at their density) and on static instability
  of the prediction, each in units of RHO_REF.
- group_weights: each layered field (41 channels) and each surface field gets the same total weight,
  so SSH is 1/9 of the loss instead of 1/209.

ARMS names the loss arms of card emu-b00-053-phys2 (trained with GraphLAM) and the convnet arms
(conv, conv_noca, unet: hycom_emulator.convnet), all scored with the projection.
`python -m hycom_emulator.physics arm <arm> <b00.yaml> <nlam.yaml>` adds the arm's settings to both files and prints the train_model arguments it needs.
Importing this module registers `hycom_graph_lam` and `hycom_wmse` with neural-lam.
"""

from __future__ import annotations

import sys

import torch
import yaml
from neural_lam import metrics
from neural_lam.models import MODELS
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

ONEM = 9806.0  # Pa of layer thickness per metre
GRADIENT_WEIGHT = 3.59  # gradient term 1/10 of the loss at emu-b00-053-phys2-reweight over all 115 train samples (base 0.358, raw 0.0111)
DENSITY_WEIGHT, STABILITY_WEIGHT = 4.00e-4, 3.90e-2  # 1/9 of base 0.427 each: sigma2 term 118.6, stability 1.22 at weight 1
MASSLESS = 1e-3 * ONEM  # Pa: a layer thinner than 1 mm holds no mass
RHO_REF = 0.01  # kg/m3: density error that costs as much as one standardized change
LAYERED = ("temp", "salin", "thknss", "u", "v")
# hycom_wmse's gradient term: T and S of the top five layers, and SSH.
GRADIENT_CHANNELS = tuple(f"{v}_k{k:02d}" for v in ("temp", "salin") for k in range(1, 6)) + ("srfhgt",)
WEIGHTED_BY_THICKNESS = ("temp", "salin", "u", "v")
# HYCOM's 7-term sigma-2 polynomial (stmt_fns.h, Brydon & Sun fit): sigma2(T, S) in kg/m3 - 1000.
SIGMA2_7T = (9.77093e00, -2.26493e-02, 7.89879e-01, -6.43205e-03, -2.62983e-03, 2.75835e-05, 3.15235e-05)

REWEIGHT = {"loss": "hycom_wmse", "group_weights": True,
            "physics": {"thickness_weighted": True, "density": 0.0, "stability": 0.0}}
ARMS = {
    "control": {"model": "graph_lam", "loss": "wmse"},
    "reweight": REWEIGHT | {"model": "graph_lam"},
    # Penalty weights: each penalty is 1/9 of the base loss (one variable group's share) at E1's
    # checkpoint over all 115 train samples (explore-physcheck/preflight3, 2026-10-03).
    "penalties": REWEIGHT | {"model": "graph_lam",
                             "physics": {"thickness_weighted": True, "density": DENSITY_WEIGHT, "stability": STABILITY_WEIGHT}},
    # convnet: option A (full-resolution conv blocks) with and without channel attention; option B (stride-4 level).
    "conv": REWEIGHT | {"model": "conv_graph_lam", "conv": {"blocks": 3, "channel_attention": True, "stride": 1}},
    "conv_noca": REWEIGHT | {"model": "conv_graph_lam", "conv": {"blocks": 3, "channel_attention": False, "stride": 1}},
    "unet": REWEIGHT | {"model": "conv_graph_lam", "conv": {"blocks": 3, "channel_attention": True, "stride": 4},
                        "graph": "multiscale_s4"},
    # Sharpness arms (card emu-b00-053-sharp): the gradient term alone, on option A, and a 3-cell mesh.
    "grad": REWEIGHT | {"model": "graph_lam", "physics": REWEIGHT["physics"] | {"gradient": GRADIENT_WEIGHT}},
    "conv_grad": REWEIGHT | {"model": "conv_graph_lam", "physics": REWEIGHT["physics"] | {"gradient": GRADIENT_WEIGHT},
                             "conv": {"blocks": 3, "channel_attention": True, "stride": 1}},
    "mesh3": REWEIGHT | {"model": "graph_lam", "graph": "mesh3"},
    "unet_norm": REWEIGHT | {"model": "conv_graph_lam",
                             "conv": {"blocks": 3, "channel_attention": True, "stride": 4, "norm": True},
                             "graph": "multiscale_s4"},
}

def sigma2(t, s):
    c1, c2, c3, c4, c5, c6, c7 = SIGMA2_7T
    return c1 + c3 * s + t * (c2 + c5 * s + t * (c4 + c7 * s + c6 * t))


def layer_columns(names: list[str], var: str) -> list[int]:
    return sorted((i for i, n in enumerate(names) if n.startswith(f"{var}_k")), key=lambda i: names[i])


def project_thickness(state, prev_state, mean, std, th, change_var):
    """state, prev_state: (..., grid, feature) standardized; change_var: (thickness layers,) variance of
    each layer's 24 h thickness change (Pa^2). Thickness columns th become >= 0 and each column sums to
    prev_state's column sum, in two steps:
    1. Excess: interfaces accumulated from the top are clipped at the column sum, as HYCOM closes a
       column. This removes first the spurious mass the network scatters into layers that are empty
       below the bottom (about +-1.7 m in E1), not mass from real layers.
    2. Shortfall: what is still missing is spread over the layers that hold mass in prev_state in
       proportion to change_var, the least-squares correction under wmse's own metric, so the fixed
       z-level layers at the top (tiny change variance) barely move and empty layers stay empty.
    Spreading a column's excess over its real layers instead (the first version) took it out of the
    near-fixed shelf layers, whose tiny change std made the fine-tune diverge (emu-b00-053-phys,
    2026-10-02). Other features are untouched."""
    dp = torch.relu(state[..., th] * std[th] + mean[th])
    prev = torch.relu(prev_state[..., th] * std[th] + mean[th])
    total = prev.sum(-1, keepdim=True)
    interfaces = torch.cat([torch.zeros_like(total), torch.minimum(torch.cumsum(dp, -1), total)], -1)
    dp = interfaces[..., 1:] - interfaces[..., :-1]
    w = (prev > MASSLESS) * change_var
    dp = dp + (total - interfaces[..., -1:]) * w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    out = state.clone()
    out[..., th] = (dp - mean[th]) / std[th]
    return out


class ThicknessProjection:
    """Mixin for a neural-lam step predictor: project_thickness after the residual update."""

    def __init__(self, *args, datastore, **kwargs):
        super().__init__(*args, datastore=datastore, **kwargs)
        names = datastore.get_vars_names("state")
        th = torch.tensor(layer_columns(names, "thknss"))
        self.register_buffer("thickness_idx", th, persistent=False)
        stats = datastore.get_standardization_dataarray("state")
        change_std = (stats.state_diff_std_standardized * stats.state_std).values[th.numpy()]  # Pa
        self.register_buffer("thickness_change_var", torch.tensor(change_std, dtype=torch.float32) ** 2, persistent=False)
        CONTEXT.configure(datastore)

    def get_clamped_new_state(self, state_delta, prev_state):
        new_state = super().get_clamped_new_state(state_delta, prev_state)
        return project_thickness(new_state, prev_state, self.state_mean, self.state_std, self.thickness_idx, self.thickness_change_var)


class HycomGraphLAM(ThicknessProjection, GraphLAM):
    pass


class _Context:
    """What hycom_wmse needs beyond neural-lam's metric arguments: feature columns, statistics and
    the physics settings of the datastore config. Set by HycomDatastore when its config has a
    `physics` section, and by HycomGraphLAM."""

    def configure(self, datastore):
        names = datastore.get_vars_names("state")
        stats = datastore.get_standardization_dataarray("state")
        self.cols = {v: torch.tensor(layer_columns(names, v)) for v in LAYERED}
        self.mean = torch.tensor(stats.state_mean.values, dtype=torch.float32)
        self.std = torch.tensor(stats.state_std.values, dtype=torch.float32)
        self.grad_cols = torch.tensor([names.index(n) for n in GRADIENT_CHANNELS if n in names])
        shape = datastore.grid_shape_state
        self.grid = (shape.x, shape.y)
        self.settings = {"thickness_weighted": False, "density": 0.0, "stability": 0.0, "gradient": 0.0}
        self.settings |= datastore.config.get("physics", {})

    def to(self, device):
        if self.mean.device != device:
            self.mean, self.std = self.mean.to(device), self.std.to(device)
            self.cols = {k: v.to(device) for k, v in self.cols.items()}
            self.grad_cols = self.grad_cols.to(device)
        return self


CONTEXT = _Context()


def hycom_wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """neural-lam's wmse, reshaped per CONTEXT.settings. Penalties enter as extra 'variables' only
    when sum_vars is True (the training loss); per-variable logging sees the reweighted wmse."""
    c = CONTEXT.to(pred.device)
    entry = (pred - target) ** 2 / pred_std**2
    phys = lambda x, v: x[..., c.cols[v]] * c.std[c.cols[v]] + c.mean[c.cols[v]]  # noqa: E731
    dp_true = phys(target, "thknss").clamp_min(0.0)
    rel = dp_true / dp_true.mean(-1, keepdim=True).clamp_min(1e-6)  # 0 for empty layers, mean 1 per column
    if c.settings["thickness_weighted"]:
        w = torch.ones_like(entry)
        for v in WEIGHTED_BY_THICKNESS:
            w[..., c.cols[v]] = rel
        entry = entry * w
    loss = metrics.mask_and_reduce_metric(entry, mask=mask, average_grid=average_grid, sum_vars=sum_vars)
    if sum_vars and average_grid and c.settings["gradient"]:
        loss = loss + c.settings["gradient"] * gradient_error(pred, target, pred_std, mask, c)
    if not sum_vars or not (c.settings["density"] or c.settings["stability"]):
        return loss
    rho_pred = sigma2(phys(pred, "temp"), phys(pred, "salin"))
    extra = torch.zeros_like(entry[..., 0])
    if c.settings["density"]:
        rho_true = sigma2(phys(target, "temp"), phys(target, "salin"))
        extra = extra + c.settings["density"] * (rel * ((rho_pred - rho_true) / RHO_REF) ** 2).mean(-1)
    if c.settings["stability"]:
        both = torch.minimum(rel[..., 1:], rel[..., :-1])
        inversion = torch.relu(rho_pred[..., :-1] - rho_pred[..., 1:]) / RHO_REF
        extra = extra + c.settings["stability"] * (both * inversion**2).mean(-1)
    return loss + metrics.mask_and_reduce_metric(extra[..., None], mask=mask, average_grid=average_grid, sum_vars=True)


def gradient_error(pred, target, pred_std, mask, c):
    """Mean |horizontal difference| of the standardized error over GRADIENT_CHANNELS, between grid
    neighbours that are both interior. pred - target is also the error of the predicted 24 h change,
    so this is an L1 loss on the gradient of the change: it pays for a front that is in the right
    place but blurred, which the squared error alone prefers when the front's position is uncertain."""
    e = (pred - target)[..., c.grad_cols] / pred_std[..., c.grad_cols]
    e = e.reshape(*e.shape[:-2], *c.grid, e.shape[-1])
    m = torch.ones(c.grid, dtype=torch.bool, device=e.device) if mask is None else mask.reshape(c.grid)
    total, count = 0.0, 0
    for axis in (-3, -2):
        n = e.shape[axis] - 1
        d = (e.narrow(axis, 1, n) - e.narrow(axis, 0, n)).abs()
        both = m.narrow(axis + 1, 1, n) & m.narrow(axis + 1, 0, n)
        total = total + (d * both[..., None]).sum((-3, -2))
        count += int(both.sum())
    return (total / count).mean(-1)


def group_weights(names: list[str]) -> dict[str, float]:
    """Each layered field and each surface field gets total weight 1/groups."""
    layered = {n: n.split("_k")[0] for n in names if "_k" in n}
    groups = len(set(layered.values())) + sum(1 for n in names if n not in layered)
    per = {v: sum(1 for g in layered.values() if g == v) for v in set(layered.values())}
    return {n: 1.0 / (groups * per[layered[n]]) if n in layered else 1.0 / groups for n in names}


def apply_arm(arm: str, b00_yaml, nlam_yaml) -> list[str]:
    """Write arm settings into the two config files; return the train_model arguments."""
    from hycom_emulator.datastore import HycomDatastore

    spec = ARMS[arm]
    sections = {k: spec[k] for k in ("physics", "conv") if k in spec}
    if sections:
        cfg = yaml.safe_load(open(b00_yaml)) | sections
        yaml.safe_dump(cfg, open(b00_yaml, "w"), sort_keys=False)
    if spec.get("group_weights"):
        names = HycomDatastore(b00_yaml).get_vars_names("state")
        cfg = yaml.safe_load(open(nlam_yaml))
        cfg["training"] = {"state_feature_weighting": {"__config_class__": "ManualStateFeatureWeighting",
                                                       "weights": group_weights(names)}}
        yaml.safe_dump(cfg, open(nlam_yaml, "w"), sort_keys=False)
    return ["--model", spec["model"], "--loss", spec["loss"]] + (["--graph", spec["graph"]] if "graph" in spec else [])


MODELS["hycom_graph_lam"] = HycomGraphLAM
metrics.DEFINED_METRICS["hycom_wmse"] = hycom_wmse

if __name__ == "__main__":
    if sys.argv[1:2] != ["arm"] or len(sys.argv) != 5:
        raise SystemExit("usage: python -m hycom_emulator.physics arm <arm> <b00.yaml> <nlam.yaml>")
    print(" ".join(apply_arm(sys.argv[2], sys.argv[3], sys.argv[4])))
