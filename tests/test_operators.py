import numpy as np
import pytest
import torch

from hycom_emulator import operators as op
from hycom_emulator.physics import sigma2

NY, NX = 70, 90


def grid(land=None, levels=(100.0, 200.0)):
    lon, lat = np.meshgrid(-90 + 0.08 * np.arange(NX), 22 + 0.072 * np.arange(NY))
    ocean = np.ones((len(levels), NY, NX), bool)
    if land is not None:
        ocean[:, land] = False
    return op.op_grid(lon, lat, np.full((NY, NX), 1000.0), ocean, levels, ocean[0]), lon, lat


def metres(lon, lat):
    x = np.deg2rad(lon - lon.mean()) * op.EARTH_RADIUS_M * np.cos(np.deg2rad(lat))
    y = np.deg2rad(lat - lat.mean()) * op.EARTH_RADIUS_M
    return x, y


def t(a):
    return torch.as_tensor(np.asarray(a, float))[None]


def interior(g, ok, pad=15):
    m = torch.zeros_like(g.region)
    m[pad:-pad, pad:-pad] = True
    return ok & m


def test_geostrophic_eddy_balances():
    g, lon, lat = grid()
    x, y = metres(lon, lat)
    L, A = 80e3, 0.3
    eta = A * np.exp(-(x**2 + y**2) / L**2)
    f = g.f.numpy()
    u = -op.GRAVITY / f * eta * (-2 * y / L**2)
    v = op.GRAVITY / f * eta * (-2 * x / L**2)
    tie = op.geostrophic(t(u), t(v), t(eta), g.region, g, "s15")
    m = interior(g, tie.ok)
    assert op.rms(tie.residual, m, g.region) < 0.05 * op.rms(tie.lhs, m, g.region)


def test_geostrophic_flags_flow_against_the_pressure_gradient():
    g, lon, lat = grid()
    x, y = metres(lon, lat)
    eta = 0.3 * np.exp(-(x**2 + y**2) / 80e3**2)
    f = g.f.numpy()
    u = op.GRAVITY / f * eta * (-2 * y / 80e3**2)
    v = -op.GRAVITY / f * eta * (-2 * x / 80e3**2)
    tie = op.geostrophic(t(u), t(v), t(eta), g.region, g, "s15")
    m = interior(g, tie.ok)
    assert op.rms(tie.residual, m, g.region) > 1.9 * op.rms(tie.lhs, m, g.region)


def test_thermal_wind_sign():
    g, lon, lat = grid()
    _, y = metres(lon, lat)
    t1, t2 = 25 - 2e-5 * y, 15 - 2e-5 * y
    s = np.full((NY, NX), 36.0)
    rho = 0.5 * (sigma2(torch.tensor(t1), torch.tensor(s)) + sigma2(torch.tensor(t2), torch.tensor(s))).numpy()
    drho_dy = np.gradient(rho, axis=0) / np.gradient(y, axis=0)
    shear = op.GRAVITY / (g.f.numpy() * op.RHO0) * drho_dy  # du/dz, z up
    u1, u2 = shear * 100.0, np.zeros_like(shear)  # u1 at 100 m is above u2 at 200 m
    zero = np.zeros_like(shear)
    tie = op.thermal_wind(t(u1), t(zero), t(t1), t(s), t(u2), t(zero), t(t2), t(s), 100.0, 200.0, g.region, g, "s15")
    m = interior(g, tie.ok)
    assert op.rms(tie.lhs, m, g.region) > 0
    assert op.rms(tie.residual, m, g.region) < 0.05 * op.rms(tie.lhs, m, g.region)


def test_continuity_of_uniform_convergence():
    g, lon, lat = grid()
    x, _ = metres(lon, lat)
    a = 1e-6  # 1/s
    ub = a * x
    zero = np.zeros_like(x)
    deta = -1000.0 * a * op.DAY_S
    tie = op.continuity(t(zero + deta), t(zero), t(ub), t(zero), t(ub), t(zero), g.region, g, "s15")
    m = interior(g, tie.ok)
    assert op.rms(tie.residual, m, g.region) < 0.02 * op.rms(tie.lhs, m, g.region)


def test_solid_body_vorticity():
    g, lon, lat = grid()
    x, y = metres(lon, lat)
    w = 1e-5
    zeta, ok = op.vorticity(t(-w * y), t(w * x), g.region, g, "s15")
    m = interior(g, ok)
    assert torch.allclose(zeta[m], torch.full_like(zeta[m], 2 * w), rtol=0.02)
    div, okd = op.divergence(t(-w * y), t(w * x), g.region, g, "s15")
    assert op.rms(div, interior(g, okd), g.region) < 0.02 * 2 * w


def test_smoothing_keeps_constants_at_the_coast_and_skips_land():
    land = np.zeros((NY, NX), bool)
    land[30:40, 40:60] = True
    g, *_ = grid(land)
    s, ok = op.smooth(t(np.where(land, 1e6, 3.0)), g.region, g.sigma_px["s50"])
    assert not ok[0, 35, 50] and ok[0, 29, 50]
    assert torch.allclose(s[ok], torch.full_like(s[ok], 3.0))
    gx, okx = op.ddx(s, ok, g)
    assert not okx[0, 35, 39] and not okx[0, 35, 60]
    assert torch.isfinite(gx).all()


def test_inversion_fraction():
    g, *_ = grid()
    temp = torch.stack([torch.full((NY, NX), 25.0), torch.full((NY, NX), 15.0)])[None]
    salin = torch.full_like(temp, 36.0)
    assert op.inversion_fraction(temp, salin, g) == 0
    assert op.inversion_fraction(temp.flip(1), salin, g) == pytest.approx(1.0)


def test_ties_are_differentiable():
    g, lon, lat = grid()
    x, y = metres(lon, lat)
    eta = t(0.3 * np.exp(-(x**2 + y**2) / 80e3**2)).requires_grad_()
    zero = t(np.zeros_like(x))
    tie = op.geostrophic(zero, zero, eta, g.region, g, "s50")
    op.rms(tie.residual, tie.ok, g.region).sum().backward()
    assert torch.isfinite(eta.grad).all() and eta.grad.abs().sum() > 0
