"""GraphLAM inside an RCAN pyramid: conv levels from the grid down to the mesh-coupled cells and back up.

GraphLAM's grid path is pointwise, so change below the mesh scale (~28 km) is a function of local features and
fronts come out smooth (docs/plan-2026-11-04-rcan-graph-emulator.md). RcanGraphLAM keeps every GraphLAM module and
puts index-space conv levels between the grid and the mesh:

    h0 = grid_embedder(x) + head(x)                   head: conv3x3 over the 2 states, forcing windows and statics,
                                                      LayerNorm, zero-initialised 1x1 conv
    e0 = enc0(h0); ek = enck(down_k(e(k-1)))          down: space-to-depth by s_k, 1x1 conv, starts as the cell mean
    cn = nodes(eK); rep = cn + encoding_grid_mlp(cn)  K: the mesh-coupled level (its cells are the graph's grid nodes)
    msg = m2g_gnn(process(g2m_gnn(cn)), rep) - rep    GraphLAM, unchanged
    dK = decK(image(msg) + skipK(eK))
    dk = deck(up_k(d(k+1)) + skip_k(ek))              up: conv3x3 to s^2 F channels and pixel shuffle, starts as nearest
    out = output_map(nodes(e0) + encoding_grid_mlp(nodes(e0)) + nodes(d0))

Each enc/dec is RCAN's residual-in-residual module: groups of convnet.ResBlock (pre-norm, conv-SiLU-conv, squeeze-excite)
with a group conv on each group's skip and a zero-initialised conv on the long skip. With `norm`, every conv whose
output joins the residual stream reads it through a per-point LayerNorm: the group and long-skip convs and the level
skips too, not only the blocks, and the head reads the raw input through a conv and writes into the stream through a
LayerNorm and a 1x1 conv, so what it adds is bounded by its weights whatever the input, as GraphLAM's grid embedder
(an MLP ending in a LayerNorm) is. With a bare conv head, a 1250 m bottom cell whose truth sits at 35 sigma entered
the stream linearly, its error fed back through every free-run day, and rcan_s2 blew up 3-5x a day from day 7
(emu-rea-rcan, 2026-11-04). Without the norms, the long-skip conv's first Adam steps move all 1152 taps by about lr with
one sign, its output grows with the stream it reads, and rcan_s2 diverged from scratch at lr 1e-3 (probe 2026-11-04:
1-day train loss 1.5 to 5.4 in three steps while graph_lam fell to 0.8; with the long-skip convs held at zero it
tracked graph_lam). With the head's 1x1 conv, the long-skip convs and
the level skips at zero, every level starts as the identity and a GraphLAM checkpoint loads into the model as
GraphLAM's step with the mesh reading cell means and its message written back by nearest replication. With one level
(coupled at the grid) that is GraphLAM's step exactly. The decoders carry the mesh message, and the fine
representation enters at the output as in GraphLAM, so a model grown by a finer level starts as the coarser model.

Levels are keyed by cell size (`cell_<deg>`), so a checkpoint loads into a model with more (finer) levels: weights of
the levels it has load, the new level starts at the identity, and every GraphLAM weight carries over.

Settings: the datastore config's `rcan` section, {width, levels: [{cell_deg, enc: [groups, blocks], dec: [groups,
blocks]}], channel_attention, reduction, norm, checkpoint_blocks, tile: {size, halo} | null}. Levels run fine to
coarse; the first is the grid spacing, each next one a whole multiple of the previous; the graph's grid side must be
the last level's cells (`<graph>_s<total stride>`, built by `nlam build_graph`). `tile` runs the finest level's encoder
and decoder on tiles with a halo, each tile checkpointed when checkpoint_blocks is set; a tile is exact when the halo
covers the module's receptive radius (`RIR.radius`) and channel attention is off, since squeeze-excite pools over the
tile it sees. Importing this module registers `rcan_graph_lam` with neural-lam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from neural_lam.models import MODELS
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM
from torch import nn
from torch.utils.checkpoint import checkpoint

from hycom_emulator.convnet import ResBlock, _image, _nodes

RCAN_MODULES = ("head.", "head_norm.", "head_out.", "levels.")
STRIDE_TOLERANCE = 0.02  # relative: a level's cell may differ from a whole multiple of the previous one by this much


@dataclass(frozen=True)
class Level:
    cell_deg: float
    enc: tuple[int, int]  # groups, blocks per group
    dec: tuple[int, int]


@dataclass(frozen=True)
class Tile:
    size: int
    halo: int


@dataclass(frozen=True)
class RcanSettings:
    levels: tuple[Level, ...]
    width: int | None = None
    channel_attention: bool = True
    reduction: int = 16
    norm: bool = True
    checkpoint_blocks: bool = True
    tile: Tile | None = None

    @classmethod
    def parse(cls, cfg: dict) -> RcanSettings:
        levels = tuple(Level(float(v["cell_deg"]), tuple(v["enc"]), tuple(v["dec"])) for v in cfg["levels"])
        rest = {k: v for k, v in cfg.items() if k not in ("levels", "tile")}
        return cls(levels=levels, tile=Tile(**cfg["tile"]) if cfg.get("tile") else None, **rest)


def level_key(cell_deg: float) -> str:
    return "cell_" + f"{cell_deg:g}".replace(".", "p")


def grid_spacing(xy: np.ndarray) -> float:
    """Spacing in degrees along x of an (nx, ny, 2) lon/lat grid (Mercator cells are square, so x is enough)."""
    return float(np.median(np.abs(np.diff(xy[:, 0, 0]))))


def level_strides(levels: tuple[Level, ...], spacing: float) -> list[int]:
    """[1, s1, s2, ...]: each level's stride from the previous one, checked against the grid spacing."""
    cells = [v.cell_deg for v in levels]
    if not cells:
        raise ValueError("rcan needs at least one level")
    if abs(cells[0] / spacing - 1) > STRIDE_TOLERANCE:
        raise ValueError(f"the finest rcan level is {cells[0]:g} deg; the grid spacing is {spacing:.4g} deg")
    strides = [1]
    for a, b in zip(cells, cells[1:]):
        s = round(b / a)
        if s < 2 or abs(b / a - s) > STRIDE_TOLERANCE * s:
            raise ValueError(f"rcan level {b:g} deg is not a whole multiple (>= 2) of the level {a:g} deg before it")
        strides.append(s)
    return strides


def _norm(norm: nn.LayerNorm | None, h: torch.Tensor) -> torch.Tensor:
    """Per-point LayerNorm over channels of an image (B, F, nx, ny), or h itself."""
    return norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) if norm is not None else h


def _pad(img: torch.Tensor, s: int) -> torch.Tensor:
    """Replicate the last rows and columns up to a multiple of s."""
    nx, ny = img.shape[-2:]
    return F.pad(img, (0, -ny % s, 0, -nx % s), mode="replicate")


class ResidualGroup(nn.Module):
    def __init__(self, width: int, blocks: int, channel_attention: bool, norm: bool, reduction: int):
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock(width, channel_attention, norm, reduction) for _ in range(blocks))
        self.norm = nn.LayerNorm(width) if norm else None
        self.conv = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, h, mask, checkpoint_blocks: bool = False):
        r = h
        for block in self.blocks:
            r = checkpoint(block, r, mask, use_reentrant=False) if checkpoint_blocks else block(r, mask)
        return h + self.conv(_norm(self.norm, r))


class RIR(nn.Module):
    """Residual in residual: groups, then a conv on the long skip that starts at zero (so the module starts as the
    identity)."""

    def __init__(self, width: int, groups: int, blocks: int, channel_attention: bool, norm: bool, reduction: int):
        super().__init__()
        self.groups = nn.ModuleList(ResidualGroup(width, blocks, channel_attention, norm, reduction) for _ in range(groups))
        self.norm = nn.LayerNorm(width) if norm else None
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    @property
    def radius(self) -> int:
        """Cells an output reads from in each direction: one per 3x3 conv on the longest path."""
        return sum(2 * len(g.blocks) + 1 for g in self.groups) + 1

    def forward(self, h, mask, checkpoint_blocks: bool = False):
        r = h
        for group in self.groups:
            r = group(r, mask, checkpoint_blocks)
        return h + self.conv(_norm(self.norm, r))


class SpaceToDepth(nn.Module):
    """Each s x s cell's s^2 F values -> F by a 1x1 conv that starts as the cell mean."""

    def __init__(self, width: int, s: int):
        super().__init__()
        self.s = s
        self.proj = nn.Conv2d(width * s * s, width, 1)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width).repeat_interleave(s * s, dim=1)[:, :, None, None] / (s * s))
            self.proj.bias.zero_()

    def forward(self, h):
        return self.proj(F.pixel_unshuffle(h, self.s))


class PixelShuffleUp(nn.Module):
    """conv3x3 F -> s^2 F and pixel shuffle; starts as nearest replication (centre tap 1 for each sub-pixel)."""

    def __init__(self, width: int, s: int):
        super().__init__()
        self.conv = nn.Conv2d(width, width * s * s, 3, padding=1)
        self.shuffle = nn.PixelShuffle(s)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.weight[:, :, 1, 1] = torch.eye(width).repeat_interleave(s * s, dim=0)

    def forward(self, h):
        return self.shuffle(self.conv(h))


class RcanLevel(nn.Module):
    def __init__(self, width: int, level: Level, down: int | None, up: int | None, cfg: RcanSettings):
        super().__init__()
        rir = lambda groups, blocks: RIR(width, groups, blocks, cfg.channel_attention, cfg.norm, cfg.reduction)  # noqa: E731
        self.enc, self.dec = rir(*level.enc), rir(*level.dec)
        self.skip_norm = nn.LayerNorm(width) if cfg.norm else None
        self.skip = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        if down is not None:
            self.down = SpaceToDepth(width, down)
        if up is not None:
            self.up = PixelShuffleUp(width, up)


def _tiled(fn, h, mask, tile: Tile, checkpoint_tiles: bool):
    """fn(h, mask) computed tile by tile, each tile read with `halo` cells around it and cropped back."""
    nx, ny = h.shape[-2:]
    size, halo = tile.size, tile.halo
    rows = []
    for x0 in range(0, nx, size):
        x1, a, b = min(x0 + size, nx), max(x0 - halo, 0), min(x0 + size + halo, nx)
        cols = []
        for y0 in range(0, ny, size):
            y1, c, d = min(y0 + size, ny), max(y0 - halo, 0), min(y0 + size + halo, ny)
            args = (h[..., a:b, c:d], mask[..., a:b, c:d])
            t = checkpoint(fn, *args, use_reentrant=False) if checkpoint_tiles else fn(*args)
            cols.append(t[..., x0 - a : x1 - a, y0 - c : y1 - c])
        rows.append(torch.cat(cols, -1))
    return torch.cat(rows, -2)


class RcanGraphLAM(GraphLAM):
    def __init__(self, *args, datastore, **kwargs):
        super().__init__(*args, datastore=datastore, **kwargs)
        self.settings = cfg = RcanSettings.parse(datastore.config["rcan"])
        width = self.hidden_dim
        if cfg.width not in (None, width):
            raise ValueError(f"rcan width {cfg.width} must equal hidden_dim {width}")
        shape = datastore.grid_shape_state
        self.strides = level_strides(cfg.levels, grid_spacing(datastore.get_xy("state", stacked=False)))
        self.sizes = [(shape.x, shape.y)]
        for s in self.strides[1:]:
            nx, ny = self.sizes[-1]
            self.sizes.append((-(-nx // s), -(-ny // s)))
        cells = self.sizes[-1][0] * self.sizes[-1][1]
        receivers, senders = int(self.m2g_edge_index[1].max()) + 1, int(self.g2m_edge_index[0].max()) + 1
        if receivers != cells or senders > cells:
            total = int(np.prod(self.strides))
            raise ValueError(f"graph {kwargs.get('graph_name')!r} has {receivers} grid nodes; the coupled rcan level "
                             f"({cfg.levels[-1].cell_deg:g} deg, stride {total}) has {cells} (use <graph>_s{total})")
        mask = torch.tensor(datastore.boundary_mask.values, dtype=torch.float32).reshape(1, 1, shape.x, shape.y)
        for k, s in enumerate(self.strides):
            mask = F.max_pool2d(_pad(mask, s), s) if k else mask
            self.register_buffer(f"mask{k}", mask, persistent=False)
        self.head = nn.Conv2d(self.grid_input_dim, width, 3, padding=1)
        self.head_norm = nn.LayerNorm(width) if cfg.norm else None
        self.head_out = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.head_out.weight)
        nn.init.zeros_(self.head_out.bias)
        last = len(cfg.levels) - 1
        self.levels = nn.ModuleDict({
            level_key(v.cell_deg): RcanLevel(width, v, self.strides[k] if k else None,
                                             self.strides[k + 1] if k < last else None, cfg)
            for k, v in enumerate(cfg.levels)
        })
        self.register_load_state_dict_pre_hook(_fill_new_modules)

    def _rir(self, module: RIR, h, k: int):
        mask, ckpt = getattr(self, f"mask{k}"), self.settings.checkpoint_blocks and torch.is_grad_enabled()
        if k == 0 and self.settings.tile is not None:
            return _tiled(module, h, mask.expand(h.shape[0], -1, -1, -1), self.settings.tile, ckpt)
        return module(h, mask, ckpt)

    def forward(self, prev_state, prev_prev_state, forcing):
        batch_size = prev_state.shape[0]
        x = torch.cat((prev_state, prev_prev_state, forcing, self.expand_to_batch(self.grid_static_features, batch_size)), -1)
        nx, ny = self.sizes[0]
        levels = list(self.levels.values())
        h = _image(self.grid_embedder(x), nx, ny) + self.head_out(_norm(self.head_norm, self.head(_image(x, nx, ny))))
        enc = []
        for k, level in enumerate(levels):
            if k:
                h = level.down(_pad(enc[-1], self.strides[k]))
            enc.append(h := self._rir(level.enc, h, k))
        cn = _nodes(enc[-1])
        rep = cn + self.encoding_grid_mlp(cn)
        mesh_emb = self.expand_to_batch(self.embedd_mesh_nodes(), batch_size)
        mesh_rep = self.g2m_gnn(cn, mesh_emb, self.expand_to_batch(self.g2m_embedder(self.g2m_features), batch_size))
        mesh_rep = self.process_step(mesh_rep)
        message = self.m2g_gnn(mesh_rep, rep, self.expand_to_batch(self.m2g_embedder(self.m2g_features), batch_size)) - rep
        d = _image(message, *self.sizes[-1])
        for k in reversed(range(len(levels))):
            if k < len(levels) - 1:
                d = levels[k].up(d)[..., : self.sizes[k][0], : self.sizes[k][1]]
            d = self._rir(levels[k].dec, d + levels[k].skip(_norm(levels[k].skip_norm, enc[k])), k)
        fine = _nodes(enc[0])
        net_output = self.output_map(fine + self.encoding_grid_mlp(fine) + _nodes(d))
        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(2, dim=-1)
            pred_std = F.softplus(pred_std_raw)
        else:
            pred_delta_mean, pred_std = net_output, None
        return self.get_clamped_new_state(pred_delta_mean * self.diff_std + self.diff_mean, prev_state), pred_std


def _fill_new_modules(module, state_dict, prefix, *_):
    """Every rcan weight the state dict lacks (a GraphLAM checkpoint, or one with fewer levels) takes its initial
    value; weights it has load as usual, and unexpected keys still fail a strict load."""
    for k, v in module.state_dict().items():
        if k.startswith(RCAN_MODULES) and prefix + k not in state_dict:
            state_dict[prefix + k] = v


MODELS["rcan_graph_lam"] = RcanGraphLAM
