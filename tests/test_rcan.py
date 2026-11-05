"""rcan: the RCAN pyramid around GraphLAM on a tiny synthetic rea_pack (0.5 deg grid, 14 x 12; stride 2: 1 deg, 7 x 6).
Identity start from a GraphLAM checkpoint, space-to-depth and pixel shuffle at init, growth by a finer level, tiling and
block checkpointing. No RCC data."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from test_rea_pack import NX, NY, archive, pack, plan_for  # noqa: F401  fixtures
from test_rea_unet import HIDDEN, _config, _datastore

from hycom_emulator.convnet import REA_ARCHS, _image, _nodes, apply_arch
from hycom_emulator.rcan import (RCAN_MODULES, PixelShuffleUp, RcanGraphLAM, SpaceToDepth, Level, level_key, level_strides)
from hycom_emulator.rea_pack import build

pytest.importorskip("neural_lam")


def _rcan(*levels, **kw):
    cfg = {"channel_attention": True, "reduction": 4, "norm": True, "checkpoint_blocks": False, "tile": None,
           "levels": [{"cell_deg": c, "enc": list(e), "dec": list(d)} for c, e, d in levels]} | kw
    return yaml.safe_dump({"rcan": cfg}, sort_keys=False)


ONE = ((0.5, (1, 2), (1, 2)),)
TWO = ((0.5, (2, 2), (1, 2)), (1.0, (1, 1), (1, 1)))
SMALL = ((0.5, (1, 1), (1, 1)), (1.0, (1, 1), (1, 1)))  # level-0 radius 4: 3-row tiles read part of the 14 x 12 grid


@pytest.fixture(scope="module")
def rea(pack, archive, tmp_path_factory):  # noqa: F811
    """(root, stride-1 pack, stride-2 pack) with the graphs the tests use built next to each config."""
    from hycom_emulator.nlam import build_graph

    out, _ = pack
    root = tmp_path_factory.mktemp("rcan")
    out2 = root / "p2"
    build(plan_for(archive[0], stride=2), out2, workers=1)
    build_graph(str(_config(root, out, "g1")), "multiscale_s2")  # builds multiscale too
    (root / "c2").mkdir()  # 7 x 6 is too small for create_graph's mesh levels: take the 0.5 deg pack's mesh
    build_graph(str(_config(root / "c2", out2, "g2")), "multiscale", mesh_from=str(root / "graph" / "multiscale"))
    return root, out, out2


def _model(root, out, name, levels, graph, seed=1, cls=RcanGraphLAM, **kw):
    ds = _datastore(_config(root, out, name, _rcan(*levels, **kw)))
    torch.manual_seed(seed)
    return cls(datastore=ds, graph_name=graph, hidden_dim=HIDDEN, processor_layers=2), ds


def _inputs(ds, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    f = ds.get_num_data_vars("state")
    return (torch.randn(2, n, f, generator=g), torch.randn(2, n, f, generator=g),
            torch.randn(2, n, 3 * ds.get_num_data_vars("forcing"), generator=g))


def _perturb(model, seed=2, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(RCAN_MODULES):
                p.add_(scale * torch.randn(p.shape, generator=g))


def test_one_level_starts_as_graph_lam(rea):
    """Coupled at the grid, the loaded model is GraphLAM's step exactly."""
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    root, out, _ = rea
    parent, ds = _model(root, out, "one", ONE, "multiscale", seed=0, cls=GraphLAM)
    model, _ = _model(root, out, "one", ONE, "multiscale")
    model.load_state_dict(parent.state_dict(), strict=True)
    x = _inputs(ds, NX * NY)
    torch.testing.assert_close(model(*x)[0], parent(*x)[0], rtol=0, atol=1e-5)


@pytest.mark.parametrize("levels, graph", [(ONE, "multiscale"), (TWO, "multiscale_s2")])
def test_an_input_outlier_cannot_grow_the_residual_stream(rea, levels, graph):
    """A state channel at 1000 sigma at one cell moves the fine level's input there by a bounded amount: the head
    writes into the stream through a LayerNorm, as GraphLAM's grid embedder does. Without it a free run fed back a
    1250 m bottom cell at 35 sigma linearly and grew without bound (emu-rea-rcan, 2026-11-04)."""
    root, out, _ = rea
    model, ds = _model(root, out, f"outlier{len(levels)}", levels, graph)
    _perturb(model)
    seen = []
    model.levels[level_key(levels[0][0])].enc.register_forward_hook(lambda mod, i, o: seen.append(i[0][0, :, 3, 4].norm()))
    x = _inputs(ds, NX * NY)
    big = x[0].clone()
    big[0, 3 * NY + 4, 0] = 1000.0
    with torch.no_grad():
        model(*x)
        model(big, *x[1:])
    assert seen[1] < 5 * seen[0], f"stream at the outlier cell grew {float(seen[1] / seen[0]):.0f}x"


def test_two_levels_start_as_graph_lam_on_cell_means(rea):
    """At stride 2 the loaded model is GraphLAM's modules with the mesh reading 2x2 cell means of the grid embedding
    and its message written back to every point of the cell; every GraphLAM weight carries over."""
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    root, out, _ = rea
    parent, ds = _model(root, out, "two", TWO, "multiscale", seed=0, cls=GraphLAM)
    model, _ = _model(root, out, "two", TWO, "multiscale_s2")
    model.load_state_dict(parent.state_dict(), strict=True)
    own = model.state_dict()
    for k, v in parent.state_dict().items():
        torch.testing.assert_close(own[k], v, rtol=0, atol=0)

    m, x = model, _inputs(ds, NX * NY)
    with torch.no_grad():
        f = m.grid_embedder(torch.cat((*x, m.expand_to_batch(m.grid_static_features, 2)), -1))
        cn = _nodes(F.avg_pool2d(_image(f, NX, NY), 2))
        rep = cn + m.encoding_grid_mlp(cn)
        mesh = m.g2m_gnn(cn, m.expand_to_batch(m.embedd_mesh_nodes(), 2), m.expand_to_batch(m.g2m_embedder(m.g2m_features), 2))
        msg = m.m2g_gnn(m.process_step(mesh), rep, m.expand_to_batch(m.m2g_embedder(m.m2g_features), 2)) - rep
        up = _nodes(F.interpolate(_image(msg, NX // 2, NY // 2), scale_factor=2, mode="nearest"))
        delta = m.output_map(f + m.encoding_grid_mlp(f) + up)
        expected = m.get_clamped_new_state(delta * m.diff_std + m.diff_mean, x[0])
        torch.testing.assert_close(model(*x)[0], expected, rtol=0, atol=1e-5)


def test_space_to_depth_then_pixel_shuffle_start_as_the_identity():
    g = torch.Generator().manual_seed(0)
    coarse = torch.randn(2, 8, 5, 4, generator=g)
    fine = F.interpolate(coarse, scale_factor=3, mode="nearest")  # constant on each 3x3 cell
    with torch.no_grad():
        torch.testing.assert_close(SpaceToDepth(8, 3)(fine), coarse, rtol=0, atol=1e-6)
        torch.testing.assert_close(PixelShuffleUp(8, 3)(coarse), fine, rtol=0, atol=1e-6)
        torch.testing.assert_close(PixelShuffleUp(8, 3)(SpaceToDepth(8, 3)(fine)), fine, rtol=0, atol=1e-6)


def test_a_checkpoint_with_fewer_levels_loads_with_the_new_level_at_identity(rea):
    """A one-level model on the 1 deg (stride-2) pack loads into a two-level model on the 0.5 deg pack: its level and
    every GraphLAM weight carry over by cell size, the new 0.5 deg level keeps its initial weights."""
    root, out, out2 = rea
    coarse, _ = _model(root / "c2", out2, "g2", ((1.0, (1, 1), (1, 1)),), "multiscale", seed=0)
    _perturb(coarse)
    fine, ds = _model(root, out, "grow", TWO, "multiscale_s2")
    init = {k: v.clone() for k, v in fine.state_dict().items()}
    fine.load_state_dict(coarse.state_dict(), strict=True)
    loaded, source = fine.state_dict(), coarse.state_dict()
    new = level_key(0.5)
    assert set(source) < set(loaded) and any(k.startswith(f"levels.{level_key(1.0)}.down.") for k in set(loaded) - set(source))
    for k, v in loaded.items():
        expected = init[k] if k.startswith(f"levels.{new}.") or k not in source else source[k]
        torch.testing.assert_close(v, expected, rtol=0, atol=0, msg=k)
    assert torch.isfinite(fine(*_inputs(ds, NX * NY))[0]).all()


def test_tiles_equal_the_whole_field_when_the_halo_covers_the_receptive_field(rea):
    root, out, _ = rea
    whole, ds = _model(root, out, "t0", SMALL, "multiscale_s2", channel_attention=False)
    _perturb(whole)
    level0 = whole.levels[level_key(0.5)]
    radius = max(level0.enc.radius, level0.dec.radius)
    x = _inputs(ds, NX * NY)
    assert radius == 4
    for halo, exact in ((radius, True), (1, False)):
        tiled, _ = _model(root, out, f"t{halo}", SMALL, "multiscale_s2", channel_attention=False, checkpoint_blocks=True,
                          tile={"size": 3, "halo": halo})
        tiled.load_state_dict(whole.state_dict(), strict=True)
        with torch.no_grad():
            a, b = whole(*x)[0], tiled(*x)[0]
        if exact:
            torch.testing.assert_close(b, a, rtol=0, atol=1e-5)
            tiled(*x)[0].square().sum().backward()  # checkpointed tiles under autograd
        else:
            assert (b - a).abs().max() > 1e-4


def test_checkpoint_blocks_changes_no_output_or_gradient(rea):
    root, out, _ = rea
    runs = []
    for ckpt in (False, True):
        model, ds = _model(root, out, f"ck{int(ckpt)}", TWO, "multiscale_s2", checkpoint_blocks=ckpt)
        _perturb(model)
        y = model(*_inputs(ds, NX * NY))[0]
        y.square().mean().backward()
        runs.append((y.detach(), {k: p.grad for k, p in model.named_parameters() if p.grad is not None}))
    (y0, g0), (y1, g1) = runs
    torch.testing.assert_close(y1, y0, rtol=0, atol=1e-6)
    assert g0.keys() == g1.keys() and any(k.startswith("levels.") for k in g0)
    for k in g0:
        torch.testing.assert_close(g1[k], g0[k], rtol=1e-5, atol=1e-7, msg=k)


def test_level_strides_check_the_grid_spacing():
    lv = lambda c: Level(c, (1, 1), (1, 1))  # noqa: E731
    assert level_strides((lv(0.04), lv(0.08), lv(0.16)), 0.04) == [1, 2, 2]
    assert level_strides((lv(0.01), lv(0.04)), 0.01) == [1, 4]
    with pytest.raises(ValueError, match="grid spacing"):
        level_strides((lv(0.08), lv(0.16)), 0.04)
    with pytest.raises(ValueError, match="whole multiple"):
        level_strides((lv(0.04), lv(0.1)), 0.04)


def test_graph_must_be_the_coupled_cells(rea):
    root, out, _ = rea
    with pytest.raises(ValueError, match="stride 2.*use <graph>_s2"):
        _model(root, out, "bad", TWO, "multiscale")


def test_arch_writes_the_rcan_section(rea, tmp_path):
    from hycom_emulator.datastore import HycomDatastore

    _, out, _ = rea
    _config(tmp_path, out, "r")
    assert apply_arch("rcan_s2", tmp_path / "r.yaml") == ["--model", "rcan_graph_lam", "--graph", "multiscale_s2"]
    cfg = HycomDatastore(tmp_path / "r.yaml").config
    assert cfg["rcan"] == REA_ARCHS["rcan_s2"]["rcan"] and "conv" not in cfg and cfg["splits"]["train"]
    for name in ("rcan_s2", "rcan_s2_light", "rcan_s1"):
        spec = REA_ARCHS[name]
        cells = [v["cell_deg"] for v in spec["rcan"]["levels"]]
        assert cells[-1] == 0.16 and spec["graph"] == f"multiscale_s{round(0.16 / cells[0])}"
        assert level_strides(tuple(Level(v["cell_deg"], v["enc"], v["dec"]) for v in spec["rcan"]["levels"]), cells[0])

