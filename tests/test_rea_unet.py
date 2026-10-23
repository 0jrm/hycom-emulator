"""convnet on the reanalysis pack layout: image round trip, boundary mask, the rea architecture config and the
identity start from a graph_lam checkpoint. Tiny synthetic rea_pack folder, no RCC data."""

import numpy as np
import pytest
import torch
from test_rea_pack import AXIS, DAYS, NX, NY, archive, pack, plan_for  # noqa: F401  fixtures

from hycom_emulator.convnet import REA_ARCHS, ConvGraphLAM, _image, _nodes, apply_arch
from hycom_emulator.rea_pack import Grid

pytest.importorskip("neural_lam")
HIDDEN = 16


def _config(root, out, name, conv=""):
    (root / f"{name}.yaml").write_text(
        f"zarr: {out}\nsplits:\n  train: [2010-12-29, 2011-01-02]\n  val: [2011-01-03, 2011-01-04]\n"
        f"  test: [2011-01-03, 2011-01-04]\n{conv}"
    )
    (root / f"nlam_{name}.yaml").write_text(f"datastore:\n  kind: hycom\n  config_path: {name}.yaml\n")
    return root / f"nlam_{name}.yaml"


@pytest.fixture(scope="module")
def rea(pack, tmp_path_factory):  # noqa: F811
    """(root, pack folder, source fields) with graph/multiscale built for the pack's grid."""
    from hycom_emulator.nlam import build_graph

    out, source = pack
    root = tmp_path_factory.mktemp("rea_unet")
    build_graph(str(_config(root, out, "a")), "multiscale")
    return root, out, source


def _datastore(nlam_yaml):
    from neural_lam.config import load_config_and_datastore

    return load_config_and_datastore(config_path=str(nlam_yaml))[1]


def _build(cls, ds, graph="multiscale", seed=0):
    torch.manual_seed(seed)
    return cls(datastore=ds, graph_name=graph, hidden_dim=HIDDEN, processor_layers=2)


def _inputs(ds, seed=1):
    g = torch.Generator().manual_seed(seed)
    n, f = NX * NY, ds.get_num_data_vars("state")
    return (torch.randn(2, n, f, generator=g), torch.randn(2, n, f, generator=g),
            torch.randn(2, n, 3 * ds.get_num_data_vars("forcing"), generator=g))


def test_image_round_trips_a_pack_row(rea):
    root, out, source = rea
    ds = _datastore(root / "nlam_a.yaml")
    assert (ds.grid_shape_state.x, ds.grid_shape_state.y) == (NX, NY)
    names = ds.get_vars_names("state")
    row = torch.tensor(ds.get_dataarray("state", "train").values[0])[None]
    img = _image(row, NX, NY)
    assert img.shape == (1, len(names), NX, NY)
    torch.testing.assert_close(_nodes(img), row, rtol=0, atol=0)

    f = source[DAYS[0]]
    bottom = Grid.load(out / "grid.npz").static[0]  # depth (ny, nx)
    real = AXIS[3] < bottom  # 100 m is a real level here
    np.testing.assert_array_equal(img[0, names.index("temp_100m")].numpy().T[real], f["water_temp"][3][real])
    ocean = bottom > 0
    np.testing.assert_array_equal(img[0, names.index("vbaro")].numpy().T[ocean], f["v_barotropic_velocity"][ocean])


def test_conv_mask_is_the_pack_boundary(rea):
    root, out, _ = rea
    model = _build(ConvGraphLAM, _datastore(root / "nlam_a.yaml"))
    boundary = Grid.load(out / "grid.npz").boundary  # (ny, nx)
    np.testing.assert_array_equal(model.mask[0, 0].numpy().T.astype(bool), boundary)
    assert boundary[0, 5] and boundary[3, 5] and not boundary[4, 5]  # open south edge, band 2 (test_rea_pack)


def test_arch_writes_the_conv_section(rea, tmp_path):
    from hycom_emulator.datastore import HycomDatastore

    _, out, _ = rea
    nlam_yaml = _config(tmp_path, out, "r")
    assert apply_arch("unet_norm", tmp_path / "r.yaml") == ["--model", "conv_graph_lam", "--graph", "multiscale_s2"]
    cfg = HycomDatastore(tmp_path / "r.yaml").config
    assert cfg["conv"] == {"blocks": 3, "channel_attention": True, "stride": 2, "norm": True}
    assert cfg["splits"]["train"] and nlam_yaml.is_file()
    for spec in REA_ARCHS.values():
        assert spec["graph"].endswith(f"_s{spec['conv']['stride']}") and spec["conv"]["norm"]


def test_pre_norm_blocks_start_as_the_graph_lam_they_load(rea):
    """The conv blocks of every level start as the identity: at stride 1, where the mesh reads the same grid nodes as
    GraphLAM's, the loaded model is GraphLAM's forward step."""
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    root, out, _ = rea
    ds = _datastore(_config(root, out, "n", "conv:\n  norm: true\n"))
    parent = _build(GraphLAM, ds, seed=0)
    conv = _build(ConvGraphLAM, ds, seed=1)
    conv.load_state_dict(parent.state_dict(), strict=True)
    x = _inputs(ds)
    torch.testing.assert_close(conv(*x)[0], parent(*x)[0], rtol=0, atol=1e-5)


def test_unet_norm_takes_every_graph_lam_weight(rea):
    """At stride 2 the mesh reads 2x2 cell means and writes back through bilinear upsampling, so the step differs from
    GraphLAM's at init; every GraphLAM weight is carried and the new modules hold their identity init."""
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    from hycom_emulator.nlam import build_graph

    root, out, _ = rea
    nlam_yaml = _config(root, out, "u")
    args = apply_arch("unet_norm", root / "u.yaml")
    build_graph(str(nlam_yaml), args[3])
    ds = _datastore(nlam_yaml)
    parent = _build(GraphLAM, ds, seed=0)
    unet = _build(ConvGraphLAM, ds, graph=args[3], seed=1)
    unet.load_state_dict(parent.state_dict(), strict=True)
    own = unet.state_dict()
    for k, v in parent.state_dict().items():
        torch.testing.assert_close(own[k], v, rtol=0, atol=0)
    for level in ("encoder", "decoder", "coarse_encoder", "coarse_decoder"):
        assert all(not b.conv2.weight.any() and b.norm is not None for b in getattr(unet, level))
    torch.testing.assert_close(unet.down.weight.sum((-2, -1)), torch.eye(HIDDEN))
    x = _inputs(ds)
    y = unet(*x)[0]
    assert y.shape == x[0].shape and torch.isfinite(y).all()
