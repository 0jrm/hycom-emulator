"""isotach front distance and surrogate loss on synthetic grids."""

from __future__ import annotations

import numpy as np
import pytest

from hycom_emulator.isotach import (
    ISOTACH_MS,
    FrontStatus,
    front_distance,
    front_edges,
    isotach_loss,
    isotach_target,
    mercator_cell_km,
    surface_speed,
    to_flat,
    to_grid,
)

NY, NX = 60, 80
CELL = 4.0
REGION = np.zeros((NY, NX), bool)
REGION[3:-3, 3:-3] = True
FAST, SLOW = ISOTACH_MS + 0.4, ISOTACH_MS - 0.4


def _disk(cy=30, cx=40, r=10):
    y, x = np.mgrid[:NY, :NX]
    return np.where((y - cy) ** 2 + (x - cx) ** 2 <= r**2, FAST, SLOW)


def test_identical_fronts_are_zero_apart():
    d = front_distance(_disk(), _disk(), REGION, CELL)
    assert d.status is FrontStatus.OK
    assert d.hausdorff_km == d.mean_km == d.p95_km == 0


@pytest.mark.parametrize("k", [2, 5, 9])
def test_shifted_disk_is_k_cells_away(k):
    d = front_distance(_disk(cx=40 + k), _disk(), REGION, CELL)
    assert abs(d.hausdorff_km - k * CELL) <= CELL


def test_shifted_half_plane_is_exactly_k_cells_away():
    x = np.mgrid[:NY, :NX][1]
    d = front_distance(np.where(x < 46, FAST, SLOW), np.where(x < 40, FAST, SLOW), REGION, CELL)
    assert d.hausdorff_km == 6 * CELL


@pytest.mark.parametrize(
    "pred, true, status",
    [
        (_disk(), np.full((NY, NX), SLOW), FrontStatus.TRUE_EMPTY),
        (np.full((NY, NX), SLOW), _disk(), FrontStatus.PRED_EMPTY),
        (np.full((NY, NX), SLOW), np.full((NY, NX), SLOW), FrontStatus.BOTH_EMPTY),
    ],
)
def test_empty_fronts_give_nan_and_their_status(pred, true, status):
    d = front_distance(pred, true, REGION, CELL)
    assert d.status is status
    assert np.isnan([d.hausdorff_km, d.mean_km, d.p95_km]).all()


def test_region_boundary_is_not_a_front():
    speed = np.full((NY, NX), SLOW)
    speed[5:55, 5:75] = FAST
    region = np.zeros((NY, NX), bool)
    region[10:50, 10:70] = True
    assert not front_edges(speed, region).any()
    assert front_distance(speed, speed, region, CELL).status is FrontStatus.BOTH_EMPTY


def test_nan_land_and_zero_land_give_the_same_front():
    zero, nan = _disk(), _disk()
    zero[:, 52:] = 0.0
    nan[:, 52:] = np.nan
    zero[25:35, 48:] = nan[25:35, 48:] = FAST
    np.testing.assert_array_equal(front_edges(zero, REGION), front_edges(nan, REGION))


def test_to_grid_and_to_flat_use_x_outer_order():
    ny, nx = 3, 4
    g = np.arange(ny * nx).reshape(ny, nx)
    f = np.array([g[iy, ix] for ix in range(nx) for iy in range(ny)])
    np.testing.assert_array_equal(to_flat(g), f)
    np.testing.assert_array_equal(to_grid(f, (ny, nx)), g)
    lead = np.stack([f, f + 100])[None]
    np.testing.assert_array_equal(to_flat(to_grid(lead, (ny, nx))), lead)


def test_surface_speed_averages_c_grid_faces_to_p_points():
    u = np.array([[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]])
    v = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    up = np.array([[2.0, 4.0, 5.0], [2.0, 4.0, 5.0]])
    vp = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    got = surface_speed(u - 1.0, v + 0.5, np.ones_like(u), np.full_like(v, -0.5))
    np.testing.assert_allclose(got, np.hypot(up, vp), atol=1e-9)


def test_surface_speed_of_uniform_flow_is_the_total_velocity_magnitude():
    one = np.ones((2, 5, 7))
    got = surface_speed(0.3 * one, -0.4 * one, 0.2 * one, 0.1 * one)
    np.testing.assert_allclose(got, np.hypot(0.5, -0.3), atol=1e-9)


def test_mercator_cell_km_at_the_domain_edges():
    np.testing.assert_allclose(mercator_cell_km(np.array([18.0, 32.0])), [4.23, 3.77], atol=0.005)


def _loss(pred, target, tau=0.05):
    torch = pytest.importorskip("torch")
    q, d_q = isotach_target(target, REGION, CELL)
    return isotach_loss(
        torch.as_tensor(pred)[None], torch.as_tensor(q)[None], torch.as_tensor(d_q)[None], REGION, CELL, tau=tau
    )


def test_loss_is_zero_for_identical_hard_masks():
    assert _loss(_disk(), _disk(), tau=1e-3).item() == 0


def test_hard_mask_loss_is_symmetric_in_pred_and_target():
    small, large = _disk(r=5), _disk(r=15)
    assert _loss(small, large, tau=1e-3).item() == pytest.approx(_loss(large, small, tau=1e-3).item(), rel=1e-6)


def test_loss_falls_as_the_predicted_disk_approaches_the_target():
    losses = [_loss(_disk(cx=40 + dx), _disk()).item() for dx in (20, 15, 10, 5, 2, 0)]
    assert all(a > b for a, b in zip(losses, losses[1:])), losses


def test_loss_gradient_lives_on_the_predicted_front():
    torch = pytest.importorskip("torch")
    tau = 0.05
    y, x = np.mgrid[:NY, :NX]
    r = np.hypot(y - 30, x - 40)
    speed = ISOTACH_MS + 0.1 * (15 - r)
    q, d_q = isotach_target(ISOTACH_MS + 0.1 * (12 - r), REGION, CELL)
    pred = torch.tensor(speed[None], requires_grad=True)
    isotach_loss(pred, torch.as_tensor(q)[None], torch.as_tensor(d_q)[None], REGION, CELL, tau=tau).backward()
    g = pred.grad[0].abs().numpy()
    off = np.abs(speed - ISOTACH_MS)
    assert g[(off < tau) & REGION].min() > 0
    assert g[off > 10 * tau].max() < 1e-6 * g.max()


def test_flat_and_grid_inputs_give_the_same_loss():
    torch = pytest.importorskip("torch")
    preds = np.stack([_disk(cx=45), _disk(cy=34, r=8)])
    targets = [isotach_target(_disk(), REGION, CELL), isotach_target(_disk(cy=26), REGION, CELL)]
    q, d_q = (np.stack([t[i] for t in targets]) for i in (0, 1))
    cell = np.full((NY, NX), CELL) * np.linspace(0.9, 1.1, NY)[:, None]
    grid = isotach_loss(*(torch.as_tensor(a) for a in (preds, q, d_q)), REGION, cell)
    flat = isotach_loss(
        *(torch.as_tensor(to_flat(a)) for a in (preds, q, d_q)), to_flat(REGION), to_flat(cell), grid_shape=(NY, NX)
    )
    assert torch.isclose(grid, flat, rtol=1e-12)
    with pytest.raises(ValueError):
        isotach_loss(*(torch.as_tensor(to_flat(a)) for a in (preds, q, d_q)), to_flat(REGION), to_flat(cell))


def test_surface_speed_is_differentiable_at_rest_in_torch():
    torch = pytest.importorskip("torch")
    u1 = torch.zeros(2, 4, 5, dtype=torch.float64, requires_grad=True)
    z = torch.zeros(2, 4, 5, dtype=torch.float64)
    surface_speed(u1, z, z, z).sum().backward()
    assert torch.isfinite(u1.grad).all()
