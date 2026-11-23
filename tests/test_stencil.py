import numpy as np
import torch

from hycom_emulator.stencil import GramSet, Layout, centre, columns, features, heldout_mse, images, ridge, rows_by_class, window


def test_images_roundtrip_and_layout():
    nx, ny = 4, 3
    x = torch.arange(nx * ny * 2, dtype=torch.float32).reshape(1, nx * ny, 2)
    img = images(x, (nx, ny))
    assert img.shape == (1, 2, ny, nx)
    # grid_index = ix * ny + iy
    assert img[0, 1, 2, 3] == x[0, 3 * ny + 2, 1]
    assert torch.equal(columns(img), x)


def test_window_centre_is_the_cell():
    nx, ny, k = 6, 5, 3
    x = torch.randn(2, nx * ny, 3)
    w = window(x, (nx, ny), k)
    for c in range(3):
        assert torch.equal(w[..., centre(c, k)], x[..., c])
    # the east neighbour (ix + 1) of an interior cell
    i = 2 * ny + 2
    east = centre(1, k) + 1
    assert w[0, i, east] == x[0, i + ny, 1]


def test_ridge_recovers_a_shared_stencil():
    torch.manual_seed(0)
    nx, ny, k = 12, 10, 3
    lay = Layout.build(np.ones((nx * ny, 2), bool), ["a_0m", "a_10m"], [0.0, 10.0], (nx, ny), k)
    P = 2 * k * k + 1
    Wt = torch.randn(2, P, dtype=torch.float64) * 0.3
    rows = rows_by_class(lay)
    tr, ho = GramSet(P, 2, lay.nclass), GramSet(P, 2, lay.nclass)
    for gs in (tr, tr, tr, ho):
        for _ in range(20):
            x = torch.randn(1, nx * ny, 2, dtype=torch.float64)
            phi = features(x, None, (nx, ny), k)[0]
            gs.add(phi, phi @ Wt.T + 0.01 * torch.randn(nx * ny, 2, dtype=torch.float64), rows)
    W = ridge(tr, lay.levels, 1e-8)
    assert torch.allclose(W, Wt, atol=5e-3)
    mse = heldout_mse(ho, W, lay.levels)
    assert torch.all((mse > 0.5e-4) & (mse < 2e-4))  # the noise variance, 1e-4


def test_within_removes_cell_means():
    torch.manual_seed(1)
    nx, ny, k = 8, 8, 1
    lay = Layout.build(np.ones((nx * ny, 1), bool), ["a_0m"], [0.0], (nx, ny), k)
    rows = rows_by_class(lay)
    gs = GramSet(2, 1, lay.nclass, per_cell=True)
    bias = torch.randn(nx * ny, 1, dtype=torch.float64)
    for _ in range(50):
        x = torch.randn(1, nx * ny, 1, dtype=torch.float64) + 3 * bias[None]
        phi = features(x, None, (nx, ny), k)[0]
        gs.add(phi, 0.5 * x[0] + bias, rows)
    W = ridge(gs.within(rows), lay.levels, 1e-8, free_const=False)
    assert abs(float(W[0, 0]) - 0.5) < 1e-6
