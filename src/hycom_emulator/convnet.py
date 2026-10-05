"""GraphLAM with a convolutional grid path: B00 forecasts that can carry sub-mesh structure.

GraphLAM never mixes a grid point with its neighbours: each point is embedded by an MLP of its own
features, sent to the mesh (one node per ~6x5 grid cells) and decoded from it, so the predicted
change below ~25 km is a pointwise function of local features and fronts come out smooth.
ConvGraphLAM keeps every GraphLAM module and adds residual 3x3 conv blocks on the 525x385 grid,
in index space, before the grid-to-mesh step (encoder) and before the output MLP (decoder):

    e = grid_embedder(x); f = encoder(e)                       full resolution
    stride 1 (option A): the mesh exchanges with f, as GraphLAM with its embedding
    stride s (option B): the mesh exchanges with c = coarse_encoder(down(f)) on s x s cells, and its
                         message comes back through coarse_decoder and bilinear upsampling
    rep = f + encoding_grid_mlp(f) + mesh message; out = output_map(decoder(rep))

Every block's second conv starts at zero, so each block starts as the identity: with stride 1 the
model reproduces a GraphLAM checkpoint exactly at step 0 and a fine-tune learns only what the
neighbourhood adds. `down` starts as the block mean. Blocks see the boundary mask as an extra
channel; land and nest-band points are not zeroed (band values are boundary data). Channel
attention is RCAN's squeeze-excite. A GraphLAM checkpoint loads into either stride: missing conv
weights take their initial values. Option B needs a graph whose grid side is the s x s cells:
`python -m hycom_emulator.convnet coarse_graph <nlam.yaml> <graph>` writes graph/<graph>_s<s>
with the mesh of graph/<graph> and grid edges rebuilt by neural-lam's rules for the cell centres.

Settings: the datastore config's `conv` section, {blocks, channel_attention, stride}.
Importing this module registers `conv_graph_lam` and `hycom_conv_graph_lam` (with the
prediction-time thickness projection) with neural-lam, and everything physics registers.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from neural_lam.models import MODELS
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM
from torch import nn

from hycom_emulator.physics import ThicknessProjection

NEW_MODULES = ("encoder.", "decoder.", "coarse_encoder.", "coarse_decoder.", "down.")
G2M_RADIUS = 0.67  # neural-lam create_graph: grid nodes within 0.67 mesh spacings feed a mesh node
M2G_NEIGHBOURS = 4


@dataclass(frozen=True)
class ConvSettings:
    blocks: int = 3
    channel_attention: bool = True
    stride: int = 1


class ResBlock(nn.Module):
    def __init__(self, width: int, channel_attention: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(width + 1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        squeeze = max(width // 16, 1)
        self.attention = (
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, squeeze, 1), nn.SiLU(), nn.Conv2d(squeeze, width, 1), nn.Sigmoid())
            if channel_attention else None
        )

    def forward(self, h, mask):
        r = self.conv2(F.silu(self.conv1(torch.cat([h, mask.expand(h.shape[0], -1, -1, -1)], 1))))
        return h + (r * self.attention(r) if self.attention is not None else r)


def _run(blocks, h, mask):
    for block in blocks:
        h = block(h, mask)
    return h


def _image(x, nx, ny):
    return x.reshape(x.shape[0], nx, ny, -1).permute(0, 3, 1, 2)


def _nodes(img):
    return img.permute(0, 2, 3, 1).reshape(img.shape[0], -1, img.shape[1])


class ConvGraphLAM(GraphLAM):
    def __init__(self, *args, datastore, **kwargs):
        super().__init__(*args, datastore=datastore, **kwargs)
        self.settings = cfg = ConvSettings(**datastore.config.get("conv", {}))
        shape = datastore.grid_shape_state
        self.nx, self.ny, s = shape.x, shape.y, cfg.stride
        self.cx, self.cy = -(-self.nx // s), -(-self.ny // s)
        cells = self.cx * self.cy
        receivers, senders = int(self.m2g_edge_index[1].max()) + 1, int(self.g2m_edge_index[0].max()) + 1
        if receivers != cells or senders > cells:
            raise ValueError(f"graph {kwargs.get('graph_name')!r} has {receivers} grid nodes; conv stride {s} on a "
                             f"{self.nx}x{self.ny} grid needs {cells} (build it with `convnet coarse_graph`)")
        mask = torch.tensor(datastore.boundary_mask.values, dtype=torch.float32).reshape(1, 1, self.nx, self.ny)
        self.register_buffer("mask", mask, persistent=False)
        width = self.hidden_dim
        blocks = lambda: nn.ModuleList(ResBlock(width, cfg.channel_attention) for _ in range(cfg.blocks))  # noqa: E731
        self.encoder, self.decoder = blocks(), blocks()
        if s > 1:
            self.register_buffer("coarse_mask", F.max_pool2d(self._pad(mask), s), persistent=False)
            self.down = nn.Conv2d(width, width, s, stride=s)
            with torch.no_grad():
                self.down.weight.copy_(torch.eye(width)[:, :, None, None].expand(-1, -1, s, s) / s**2)
                self.down.bias.zero_()
            self.coarse_encoder, self.coarse_decoder = blocks(), blocks()
        self.register_load_state_dict_pre_hook(_fill_new_modules)

    def _pad(self, img):
        s = self.settings.stride
        return F.pad(img, (0, self.cy * s - self.ny, 0, self.cx * s - self.nx), mode="replicate")

    def forward(self, prev_state, prev_prev_state, forcing):
        batch_size = prev_state.shape[0]
        s = self.settings.stride
        grid_features = torch.cat(
            (prev_state, prev_prev_state, forcing, self.expand_to_batch(self.grid_static_features, batch_size)), dim=-1
        )
        f = _run(self.encoder, _image(self.grid_embedder(grid_features), self.nx, self.ny), self.mask)
        fn = _nodes(f)
        fine = fn + self.encoding_grid_mlp(fn)
        if s == 1:
            cn, rep = fn, fine
        else:
            cn = _nodes(_run(self.coarse_encoder, self.down(self._pad(f)), self.coarse_mask))
            rep = cn + self.encoding_grid_mlp(cn)
        mesh_emb = self.expand_to_batch(self.embedd_mesh_nodes(), batch_size)
        mesh_rep = self.g2m_gnn(cn, mesh_emb, self.expand_to_batch(self.g2m_embedder(self.g2m_features), batch_size))
        mesh_rep = self.process_step(mesh_rep)
        message = self.m2g_gnn(mesh_rep, rep, self.expand_to_batch(self.m2g_embedder(self.m2g_features), batch_size)) - rep
        if s > 1:
            up = F.interpolate(_run(self.coarse_decoder, _image(message, self.cx, self.cy), self.coarse_mask),
                               scale_factor=s, mode="bilinear", align_corners=False)
            message = _nodes(up[..., : self.nx, : self.ny])
        net_output = self.output_map(_nodes(_run(self.decoder, _image(fine + message, self.nx, self.ny), self.mask)))
        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(2, dim=-1)
            pred_std = F.softplus(pred_std_raw)
        else:
            pred_delta_mean, pred_std = net_output, None
        return self.get_clamped_new_state(pred_delta_mean * self.diff_std + self.diff_mean, prev_state), pred_std


def _fill_new_modules(module, state_dict, prefix, *_):
    """A GraphLAM state dict (none of the conv weights) gets the module's initial conv weights."""
    own = {k: v for k, v in module.state_dict().items() if k.startswith(NEW_MODULES)}
    if not any(prefix + k in state_dict for k in own):
        state_dict.update({prefix + k: v for k, v in own.items()})


class HycomConvGraphLAM(ThicknessProjection, ConvGraphLAM):
    pass


def cell_centres(xy: np.ndarray, s: int) -> np.ndarray:
    """(nx, ny, 2) grid positions -> (ceil(nx/s), ceil(ny/s), 2) mean position of each s x s block."""
    nx, ny, _ = xy.shape
    cx, cy = -(-nx // s), -(-ny // s)
    padded = np.full((cx * s, cy * s, 2), np.nan)
    padded[:nx, :ny] = xy
    return np.nanmean(padded.reshape(cx, s, cy, s, 2), axis=(1, 3))


def grid_edges(grid_xy: np.ndarray, mesh_xy: np.ndarray, spacing: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """g2m and m2g (edge_index, features) between grid nodes (N, 2) and mesh nodes, by neural-lam's
    create_graph rules: every grid node within G2M_RADIUS x spacing feeds a mesh node; every grid node
    reads its M2G_NEIGHBOURS nearest mesh nodes. Features are [length, sender - receiver]."""
    from scipy.spatial import KDTree

    def edges(senders, receivers, s_xy, r_xy):
        diff = s_xy[senders] - r_xy[receivers]
        feats = np.concatenate([np.hypot(diff[:, 0], diff[:, 1])[:, None], diff], 1)
        return torch.tensor(np.stack([senders, receivers]), dtype=torch.int64), torch.tensor(feats, dtype=torch.float32)

    near = KDTree(grid_xy).query_ball_point(mesh_xy, spacing * G2M_RADIUS)
    g2m_mesh = np.repeat(np.arange(len(mesh_xy)), [len(n) for n in near])
    g2m_grid = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    m2g_mesh = KDTree(mesh_xy).query(grid_xy, M2G_NEIGHBOURS)[1].reshape(-1)
    m2g_grid = np.repeat(np.arange(len(grid_xy)), M2G_NEIGHBOURS)
    return {"g2m": edges(g2m_grid, g2m_mesh, grid_xy, mesh_xy), "m2g": edges(m2g_mesh, m2g_grid, mesh_xy, grid_xy)}


def mesh_spacing(mesh_xy: np.ndarray) -> float:
    """The spacing create_graph scales the g2m radius by: the x step of the finest mesh level."""
    x = np.unique(mesh_xy[:, 0])
    steps = np.diff(x)
    return float(steps[steps > 1e-6 * (x[-1] - x[0])].min())


def coarse_graph(graph_dir: Path, xy: np.ndarray, stride: int) -> Path:
    """graph_dir's mesh with grid edges for the stride x stride cells of xy (nx, ny, 2); idempotent."""
    out = graph_dir.with_name(f"{graph_dir.name}_s{stride}")
    if (out / "m2g_features.pt").is_file():
        return out
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(graph_dir, tmp)
    mesh_xy = torch.load(graph_dir / "mesh_features.pt", weights_only=True)[0].numpy()
    for name, (index, feats) in grid_edges(cell_centres(xy, stride).reshape(-1, 2), mesh_xy, mesh_spacing(mesh_xy)).items():
        torch.save(index, tmp / f"{name}_edge_index.pt")
        torch.save(feats, tmp / f"{name}_features.pt")
    tmp.rename(out)
    return out


MODELS["conv_graph_lam"] = ConvGraphLAM
MODELS["hycom_conv_graph_lam"] = HycomConvGraphLAM

if __name__ == "__main__":
    if sys.argv[1:2] != ["coarse_graph"] or len(sys.argv) != 4:
        raise SystemExit("usage: python -m hycom_emulator.convnet coarse_graph <nlam.yaml> <graph name>")
    from neural_lam.config import load_config_and_datastore

    import hycom_emulator.datastore  # noqa: F401  registers the hycom kind

    _, ds = load_config_and_datastore(config_path=sys.argv[2])
    stride = ConvSettings(**ds.config.get("conv", {})).stride
    if stride > 1:
        print(coarse_graph(ds.root_path / "graph" / sys.argv[3], ds.get_xy("state", stacked=False), stride))
