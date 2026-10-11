"""ensemble: noisy GraphLAM, afCRPS and the member rollout. Synthetic pack, no RCC data."""

import itertools

import numpy as np
import pytest
import torch
from test_stack import SPLITS, T, _run, write_pack

from hycom_emulator.ensemble import CRPSGraphLAM, crps_ensemble, module_factory, split_args

NX, NY, HIDDEN = 12, 10, 16


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    from neural_lam.config import load_config_and_datastore

    from hycom_emulator.nlam import main

    root = tmp_path_factory.mktemp("ens")
    rng = np.random.default_rng(0)
    pack = write_pack(root / "pack", *_run(rng, False, rng.normal(size=(T, NX * NY, 1))), shape=(NX, NY))
    (root / "a.yaml").write_text(f"zarr: {pack}\n{SPLITS}")
    (root / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: a.yaml\n")
    main(["nlam", "create_graph", "--config_path", str(root / "nlam.yaml"), "--name", "multiscale"])
    return load_config_and_datastore(config_path=str(root / "nlam.yaml"))


def build(cls, datastore, seed=0):
    torch.manual_seed(seed)
    return cls(datastore=datastore, graph_name="multiscale", hidden_dim=HIDDEN, processor_layers=2)


def inputs(datastore, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    n, f = NX * NY, datastore.get_num_data_vars("state")
    return (torch.randn(batch, n, f, generator=g), torch.randn(batch, n, f, generator=g),
            torch.randn(batch, n, 3 * datastore.get_num_data_vars("forcing"), generator=g))


def direct_afcrps(x, y, alpha):
    """x (M,) members, y truth: Lang et al. (2024) eq. for afCRPS, summed over ordered pairs j != k."""
    m = len(x)
    eps = (1 - alpha) / m
    pairs = [abs(x[j] - y) + abs(x[k] - y) - (1 - eps) * abs(x[j] - x[k]) for j, k in itertools.permutations(range(m), 2)]
    return sum(pairs) / (2 * m * (m - 1))


@pytest.mark.parametrize("alpha", [1.0, 0.95, 0.0])
@pytest.mark.parametrize("members", [2, 3, 5])
def test_afcrps_equals_the_direct_sum(alpha, members):
    g = torch.Generator().manual_seed(members)
    b, t, n, f = 2, 3, 7, 4
    pred = torch.randn(b, t, members, n, f, generator=g, dtype=torch.float64)
    target = torch.randn(b, t, n, f, generator=g, dtype=torch.float64)
    std = torch.rand(f, generator=g, dtype=torch.float64) + 0.5
    mask = torch.tensor([True, False, True, True, False, True, True])
    ours = crps_ensemble(pred, target, std, mask=mask, alpha=alpha)
    entry = torch.zeros(b, t, n, f, dtype=torch.float64)
    for i in np.ndindex(b, t, n, f):
        bi, ti, ni, fi = i
        entry[i] = direct_afcrps(pred[bi, ti, :, ni, fi].tolist(), target[i].item(), alpha) / std[fi]
    assert torch.allclose(ours, entry[..., mask, :].mean(-2).sum(-1), rtol=1e-12)


def test_alpha_zero_is_the_plain_ensemble_crps():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(4, 1, 1, generator=g, dtype=torch.float64), torch.randn(1, 1, generator=g, dtype=torch.float64)
    plain = (x - y).abs().mean() - (x[:, None] - x[None]).abs().mean() / 2
    assert torch.allclose(crps_ensemble(x, y, torch.ones(1, dtype=torch.float64), alpha=0.0), plain)


def test_crps_is_zero_for_members_on_the_truth():
    target = torch.randn(2, 3, 7, 4)
    pred = target.unsqueeze(2).expand(-1, -1, 2, -1, -1)
    for alpha in (1.0, 0.95):
        assert crps_ensemble(pred, target, torch.ones(4), alpha=alpha).abs().max() == 0


def test_one_member_is_wmae():
    from neural_lam.metrics import wmae

    pred, target, std = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4), torch.rand(4) + 0.5
    assert torch.allclose(crps_ensemble(pred, target, std), wmae(pred, target, std))


def test_noisy_graph_lam_starts_as_the_graph_lam_it_loads(ds):
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    _, datastore = ds
    parent, noisy = build(GraphLAM, datastore, seed=0), build(CRPSGraphLAM, datastore, seed=1)
    noisy.load_state_dict(parent.state_dict(), strict=True)
    x = inputs(datastore)
    assert torch.equal(noisy(*x)[0], parent(*x)[0]), "zero FiLM weights must reproduce GraphLAM exactly, noise on"


def test_zero_noise_is_the_deterministic_graph_lam(ds):
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    _, datastore = ds
    parent, noisy = build(GraphLAM, datastore, seed=0), build(CRPSGraphLAM, datastore, seed=1)
    noisy.load_state_dict(parent.state_dict(), strict=True)
    for layer in noisy.film:
        torch.nn.init.normal_(layer.weight)
    x = inputs(datastore)
    noisy.noise_scale = 0.0
    assert torch.equal(noisy(*x)[0], parent(*x)[0]), "z = 0 must give GraphLAM with the same weights"
    noisy.noise_scale = 1.0
    a, b = noisy(*x)[0], noisy(*x)[0]
    assert (a - b).abs().max() > 1e-3, "two draws with trained FiLM weights must differ"


def test_a_partial_film_state_dict_fails(ds):
    _, datastore = ds
    state = build(CRPSGraphLAM, datastore).state_dict()
    del state["film.0.weight"]
    with pytest.raises(RuntimeError, match="film.0.weight"):
        build(CRPSGraphLAM, datastore).load_state_dict(state, strict=True)


def ensemble_module(ds, tmp_path, members=2):
    """An EnsembleForecasterModule initialized from a saved GraphLAM ForecasterModule checkpoint."""
    from neural_lam.models import ARForecaster, ForecasterModule
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    config, datastore = ds
    parent = ForecasterModule(forecaster=ARForecaster(build(GraphLAM, datastore), datastore), config=config, datastore=datastore)
    torch.save({"state_dict": parent.state_dict()}, tmp_path / "parent.ckpt")
    forecaster = ARForecaster(build(CRPSGraphLAM, datastore, seed=7), datastore)
    module = module_factory(members, str(tmp_path / "parent.ckpt"))(forecaster=forecaster, config=config, datastore=datastore, loss="afcrps")
    return parent, module


def rollout_batch(datastore, steps=3, batch=2):
    g = torch.Generator().manual_seed(3)
    n, f = NX * NY, datastore.get_num_data_vars("state")
    return (torch.randn(batch, 2, n, f, generator=g), torch.randn(batch, steps, n, f, generator=g),
            torch.randn(batch, steps, n, 3 * datastore.get_num_data_vars("forcing"), generator=g), None)


def test_members_start_on_the_loaded_deterministic_rollout(ds, tmp_path):
    parent, module = ensemble_module(ds, tmp_path, members=3)
    init, target, forcing, _ = rollout_batch(ds[1])
    ens = module.forecast_members(init, forcing, target, 3)
    det, _ = parent.forecaster(init, forcing, target)
    assert ens.shape == (2, 3, *det.shape[1:])
    for j in range(3):
        assert torch.equal(ens[:, j], det), f"member {j} must start as the deterministic rollout"


def test_training_moves_the_film_and_splits_the_members(ds, tmp_path):
    _, module = ensemble_module(ds, tmp_path)
    module.train()
    batch = rollout_batch(ds[1])
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    pred, _, _, loss = module._compute_prediction_and_loss(batch)
    assert pred.shape == batch[1].shape and loss.shape == (batch[1].shape[1],)
    loss.mean().backward()
    assert all(layer.weight.grad.abs().sum() > 0 for layer in module.forecaster.predictor.film)
    opt.step()
    ens = module.forecast_members(batch[0], batch[2], batch[1], 2)
    assert (ens[:, 0] - ens[:, 1]).abs().max() > 0, "members must differ once the FiLM weights move"
    interior = module.interior_mask_bool
    assert torch.equal(ens[:, 0][..., ~interior, :], ens[:, 1][..., ~interior, :]), "the true band overwrites every member"


def test_ensemble_module_refuses_other_losses_and_models(ds):
    from neural_lam.models import ARForecaster
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    config, datastore = ds
    make = module_factory(2, None)
    with pytest.raises(ValueError, match="afcrps"):
        make(forecaster=ARForecaster(build(CRPSGraphLAM, datastore), datastore), config=config, datastore=datastore, loss="wmse")
    with pytest.raises(ValueError, match="crps_graph_lam"):
        make(forecaster=ARForecaster(build(GraphLAM, datastore), datastore), config=config, datastore=datastore, loss="afcrps")


def test_split_args():
    ours, rest = split_args(["--config_path", "x.yaml", "--model", "graph_lam", "--epochs", "3", "--model", "crps_graph_lam",
                             "--members", "4", "--init_from", "a.ckpt", "--checkpoint_steps", "--loss", "afcrps"])
    assert (ours.model, ours.members, ours.init_from, ours.checkpoint_steps) == ("crps_graph_lam", 4, "a.ckpt", True)
    assert rest == ["--config_path", "x.yaml", "--epochs", "3", "--loss", "afcrps", "--model", "crps_graph_lam"]
    assert split_args(["--model", "graph_lam"])[1] == ["--model", "graph_lam"]
    with pytest.raises(SystemExit, match="crps_graph_lam"):
        split_args(["--model", "graph_lam", "--members", "2"])


def test_checkpointed_steps_give_the_same_loss_and_gradients(ds, tmp_path):
    _, module = ensemble_module(ds, tmp_path)
    for layer in module.forecaster.predictor.film:
        torch.nn.init.normal_(layer.weight, std=0.1)
    module.train()
    batch = rollout_batch(ds[1])
    grads = []
    for flag in (False, True):
        module.forecaster.predictor.checkpoint_steps = flag
        module.zero_grad()
        torch.manual_seed(5)
        loss = module._compute_prediction_and_loss(batch)[3]
        loss.mean().backward()
        grads.append((loss.detach(), [p.grad.clone() for p in module.parameters() if p.grad is not None]))
    (l0, g0), (l1, g1) = grads
    assert torch.equal(l0, l1) and len(g0) == len(g1)
    assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(g0, g1)), "recomputed steps must redraw the same noise"


def test_validation_draws_the_same_noise_every_time_and_leaves_the_training_stream_alone(ds, tmp_path):
    _, module = ensemble_module(ds, tmp_path)
    for layer in module.forecaster.predictor.film:
        torch.nn.init.normal_(layer.weight, std=0.1)
    module.eval()
    init, target, forcing, _ = rollout_batch(ds[1])
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    runs = []
    for _ in range(2):
        module.on_validation_start()
        runs.append(module.forecast_members(init, forcing, target, 2))
        module.on_validation_end()
    assert torch.equal(*runs), "every validation must draw the same noise"
    assert torch.equal(torch.rand(3), expected), "the training RNG stream must continue where validation found it"


def test_noise_modulates_the_layer_update_not_the_mesh_state(ds, monkeypatch):
    """With one processor layer, scale c and no shift, the noisy output is h + (1 + c) (net(h) - h): the noise
    multiplies the layer's update. Multiplying the state itself, net((1 + c) h), compounded through the layers."""
    from hycom_emulator.ensemble import NOISE_DIM

    _, datastore = ds
    torch.manual_seed(0)
    noisy = CRPSGraphLAM(datastore=datastore, graph_name="multiscale", hidden_dim=HIDDEN, processor_layers=1)
    c = 3.0
    with torch.no_grad():
        noisy.film[0].weight.zero_()
        noisy.film[0].weight[:HIDDEN] = c / NOISE_DIM
    h = torch.randn(2, noisy.get_num_mesh()[0], HIDDEN, generator=torch.Generator().manual_seed(1))
    monkeypatch.setattr(torch, "randn", lambda *shape, **kw: torch.ones(*shape, **kw))
    noisy.noise_scale = 0.0
    plain = noisy.process_step(h)
    noisy.noise_scale = 1.0
    assert torch.allclose(noisy.process_step(h), h + (1 + c) * (plain - h), atol=1e-5)


def test_evaluate_rea_zero_noise_scores_the_deterministic_rollout_of_a_members_module(ds):
    from neural_lam.models import ARForecaster, ForecasterModule
    from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

    from hycom_emulator.evaluate_rea import noise_off

    config, datastore = ds
    parent = ARForecaster(build(GraphLAM, datastore, seed=0), datastore)
    noisy = build(CRPSGraphLAM, datastore, seed=1)
    noisy.load_state_dict(parent.predictor.state_dict(), strict=True)
    for layer in noisy.film:
        torch.nn.init.normal_(layer.weight)
    module = ForecasterModule(forecaster=ARForecaster(noisy, datastore), config=config, datastore=datastore, loss="afcrps")
    init, target, forcing, _ = rollout_batch(datastore)
    noise_off(module)
    a, _ = module.forecaster(init, forcing, target)
    b, _ = module.forecaster(init, forcing, target)
    assert torch.equal(a, b) and torch.equal(a, parent(init, forcing, target)[0]), "z = 0 must be the GraphLAM rollout"
    plain = ForecasterModule(forecaster=parent, config=config, datastore=datastore)
    with pytest.raises(ValueError, match="crps_graph_lam"):
        noise_off(plain)
