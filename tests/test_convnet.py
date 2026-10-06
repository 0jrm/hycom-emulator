"""convnet: conv grid path around GraphLAM's mesh (options A and B). Synthetic pack, no RCC data."""

import numpy as np
import pytest
import torch
from test_stack import SPLITS, T, _run, write_pack

from hycom_emulator.convnet import ConvGraphLAM, HycomConvGraphLAM, coarse_graph, grid_edges, mesh_spacing
from hycom_emulator.physics import apply_arm, layer_columns

NX, NY, HIDDEN = 30, 22, 16


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    from hycom_emulator.nlam import main

    root = tmp_path_factory.mktemp("conv")
    rng = np.random.default_rng(0)
    pack = write_pack(root / "pack", *_run(rng, False, rng.normal(size=(T, NX * NY, 1))), shape=(NX, NY))
    for name, conv in (("a", ""), ("b", "conv:\n  stride: 4\n"), ("n", "conv:\n  norm: true\n")):
        (root / f"{name}.yaml").write_text(f"zarr: {pack}\n{SPLITS}{conv}")
        (root / f"nlam_{name}.yaml").write_text(f"datastore:\n  kind: hycom\n  config_path: {name}.yaml\n")
    main(["nlam", "create_graph", "--config_path", str(root / "nlam_a.yaml"), "--name", "multiscale"])
    return root


def datastore(root, name):
    from neural_lam.config import load_config_and_datastore

    return load_config_and_datastore(config_path=str(root / f"nlam_{name}.yaml"))[1]


def build(cls, ds, graph="multiscale", seed=0):
    torch.manual_seed(seed)
    return cls(datastore=ds, graph_name=graph, hidden_dim=HIDDEN, processor_layers=2)


def inputs(ds, seed=1):
    g = torch.Generator().manual_seed(seed)
    n, f = NX * NY, ds.get_num_data_vars("state")
    return (torch.randn(2, n, f, generator=g), torch.randn(2, n, f, generator=g),
            torch.randn(2, n, 3 * ds.get_num_data_vars("forcing"), generator=g))


def test_option_a_starts_as_the_graph_lam_it_loads(root):
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    ds = datastore(root, "a")
    parent = build(GraphLAM, ds, seed=0)
    conv = build(ConvGraphLAM, ds, seed=1)
    conv.load_state_dict(parent.state_dict(), strict=True)
    x = inputs(ds)
    assert torch.allclose(conv(*x)[0], parent(*x)[0], atol=1e-5), "option A must reproduce GraphLAM at step 0"


def test_pre_norm_blocks_also_start_as_graph_lam(root):
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    ds = datastore(root, "n")
    parent = build(GraphLAM, ds, seed=0)
    conv = build(ConvGraphLAM, ds, seed=1)
    conv.load_state_dict(parent.state_dict(), strict=True)
    assert conv.encoder[0].norm is not None
    x = inputs(ds)
    assert torch.allclose(conv(*x)[0], parent(*x)[0], atol=1e-5), "pre-norm blocks must start as the identity"


def test_a_partial_conv_state_dict_fails(root):
    ds = datastore(root, "a")
    state = build(ConvGraphLAM, ds).state_dict()
    del state["encoder.0.conv2.weight"]
    with pytest.raises(RuntimeError, match="encoder.0.conv2.weight"):
        build(ConvGraphLAM, ds).load_state_dict(state, strict=True)


def test_training_reaches_the_neighbourhood_taps(root):
    ds = datastore(root, "a")
    model = build(ConvGraphLAM, ds)
    prev, prev_prev, forcing = inputs(ds)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ((model(prev, prev_prev, forcing)[0] - prev_prev) ** 2).mean().backward()
    opt.step()
    taps = model.encoder[0].conv2.weight.detach().clone()
    taps[..., 1, 1] = 0
    assert taps.abs().sum() > 0, "off-centre taps of the identity-initialized conv never moved"
    p, q = 10 * NY + 10, 11 * NY + 10
    prev = prev.clone().requires_grad_(True)
    model(prev, prev_prev, forcing)[0][0, p].sum().backward()
    assert prev.grad[0, q].abs().sum() > 0


def test_coarse_graph_rules_reproduce_neural_lams_graph(root):
    ds = datastore(root, "a")
    g = root / "graph" / "multiscale"
    mesh_xy = torch.load(g / "mesh_features.pt", weights_only=True)[0].numpy()
    ours = grid_edges(ds.get_xy("state", stacked=False).reshape(-1, 2), mesh_xy, mesh_spacing(mesh_xy))
    for name, (index, feats) in ours.items():
        theirs_index = torch.load(g / f"{name}_edge_index.pt", weights_only=True)
        theirs_feats = torch.load(g / f"{name}_features.pt", weights_only=True)
        key = lambda i: sorted(map(tuple, i.T.tolist()))  # noqa: E731
        assert key(index) == key(theirs_index), f"{name} edges differ from neural-lam's"
        order = lambda i: np.lexsort(i.numpy()[::-1])  # noqa: E731
        assert torch.allclose(feats[order(index)], theirs_feats[order(theirs_index)], atol=1e-5), "beyond float32 mesh positions near lon -90"


def test_option_b_runs_on_its_coarse_graph(root):
    ds = datastore(root, "b")
    out = coarse_graph(root / "graph" / "multiscale", ds.get_xy("state", stacked=False), 4)
    stamp = (out / "m2g_features.pt").stat().st_mtime_ns
    assert coarse_graph(root / "graph" / "multiscale", ds.get_xy("state", stacked=False), 4) == out
    assert (out / "m2g_features.pt").stat().st_mtime_ns == stamp, "coarse_graph must not rebuild an existing graph"
    cells = -(-NX // 4) * -(-NY // 4)
    m2g = torch.load(out / "m2g_edge_index.pt", weights_only=True)
    assert torch.equal(torch.bincount(m2g[1], minlength=cells), torch.full((cells,), 4))
    assert int(torch.load(out / "g2m_edge_index.pt", weights_only=True)[0].max()) < cells

    model = build(ConvGraphLAM, ds, graph="multiscale_s4")
    x = inputs(ds)
    y, _ = model(*x)
    assert y.shape == x[0].shape
    y.square().mean().backward()
    assert model.down.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="stride 4"):
        build(ConvGraphLAM, ds, graph="multiscale")


def test_projection_closes_conv_columns(root):
    ds = datastore(root, "a")
    model = build(HycomConvGraphLAM, ds)
    prev, prev_prev, forcing = inputs(ds)
    th = layer_columns(ds.get_vars_names("state"), "thknss")
    prev[..., th] = prev[..., th].abs()  # every column holds mass
    phys = lambda x: x[..., th] * model.state_std[th] + model.state_mean[th]  # noqa: E731
    dp = phys(model(prev, prev_prev, forcing)[0])
    assert (dp >= -1e-3).all()
    assert torch.allclose(dp.sum(-1), phys(prev).clamp_min(0).sum(-1), rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("arm, conv, extra", [
    ("conv", {"blocks": 3, "channel_attention": True, "stride": 1}, []),
    ("conv_noca", {"blocks": 3, "channel_attention": False, "stride": 1}, []),
    ("unet", {"blocks": 3, "channel_attention": True, "stride": 4}, ["--graph", "multiscale_s4"]),
    ("conv_norm", {"blocks": 3, "channel_attention": True, "stride": 1, "norm": True}, []),
    ("unet_norm", {"blocks": 3, "channel_attention": True, "stride": 4, "norm": True}, ["--graph", "multiscale_s4"]),
])
def test_conv_arms(root, tmp_path, arm, conv, extra):
    from hycom_emulator.datastore import HycomDatastore

    (tmp_path / "b00.yaml").write_text((root / "a.yaml").read_text())
    (tmp_path / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    args = apply_arm(arm, tmp_path / "b00.yaml", tmp_path / "nlam.yaml")
    assert args == ["--model", "conv_graph_lam", "--loss", "hycom_wmse", *extra]
    cfg = HycomDatastore(tmp_path / "b00.yaml").config
    assert cfg["conv"] == conv and cfg["physics"]["thickness_weighted"]


def test_mesh3_graph_takes_graph_lam_weights(root):
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    from hycom_emulator.nlam import build_graph

    for _ in range(2):
        build_graph(str(root / "nlam_a.yaml"), "mesh3")
    ds = datastore(root, "a")
    coarse, fine = build(GraphLAM, ds), build(GraphLAM, ds, graph="mesh3")
    fine.load_state_dict(coarse.state_dict(), strict=True)
    x = inputs(ds)
    assert fine(*x)[0].shape == x[0].shape


@pytest.mark.parametrize("arm, model, extra", [("grad", "graph_lam", []), ("mesh3", "graph_lam", ["--graph", "mesh3"]),
                                               ("conv_grad", "conv_graph_lam", [])])
def test_sharpness_arms(root, tmp_path, arm, model, extra):
    from hycom_emulator.datastore import HycomDatastore
    from hycom_emulator.physics import GRADIENT_WEIGHT

    (tmp_path / "b00.yaml").write_text((root / "a.yaml").read_text())
    (tmp_path / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    assert apply_arm(arm, tmp_path / "b00.yaml", tmp_path / "nlam.yaml") == ["--model", model, "--loss", "hycom_wmse", *extra]
    physics = HycomDatastore(tmp_path / "b00.yaml").config["physics"]
    assert physics.get("gradient", 0.0) == (0.0 if arm == "mesh3" else GRADIENT_WEIGHT)


@pytest.mark.parametrize("arm, extra", [("conv_norm_wmse", []), ("unet_norm_wmse", ["--graph", "multiscale_s4"])])
def test_wmse_pre_norm_arms(root, tmp_path, arm, extra):
    from hycom_emulator.datastore import HycomDatastore

    (tmp_path / "b00.yaml").write_text((root / "a.yaml").read_text())
    (tmp_path / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
    assert apply_arm(arm, tmp_path / "b00.yaml", tmp_path / "nlam.yaml") == ["--model", "conv_graph_lam", "--loss", "wmse", *extra]
    cfg = HycomDatastore(tmp_path / "b00.yaml").config
    assert cfg["conv"]["norm"] and "physics" not in cfg
    assert "training" not in (tmp_path / "nlam.yaml").read_text(), "wmse arms keep neural-lam's uniform weights"
