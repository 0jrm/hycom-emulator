"""Scale-space operators and balance residuals on the reanalysis grid, after NeSPReSO v2's operator registry
(ISAS20_project NeSPReSO2_onTemplate/preproc/features/operators.py).

Fields are (B, ny, nx) images. Every derivative is a central difference of a mask-normalized Gaussian smoothing at a
named scale, so the operators see the mesoscale and not the aliased grid scale of a strided pack. A smoothed point is
valid where it is ocean and its smoothing weight is at least 0.5; a derivative is valid where the point and both of
its neighbours are. Everything is differentiable in the fields, so a later loss can use the same code.

The ties compare two ways of getting one quantity from the state. Their residual is not zero in the truth (Ekman and
ageostrophic flow, daily means, the strided grid), so a forecast is judged against the truth's own residual.

    geo      surface velocity against the geostrophic velocity of SSH              m/s
    tw_a_b   vertical shear between depths a and b against thermal wind of sigma2   1/s
    cont     SSH tendency against minus the divergence of H times barotropic flow   m/s
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from hycom_emulator.physics import sigma2

EARTH_RADIUS_M = 6_371_000.0
GRAVITY = 9.81
RHO0 = 1025.0
OMEGA = 7.2921e-5
DAY_S = 86400.0
SCALES_KM = {"s15": 15.0, "s50": 50.0}  # Gaussian sigma: Loop Current front, eddy
VORTICITY_DEPTHS = (0.0, 100.0)
SHEAR_PAIRS = ((100.0, 200.0), (200.0, 500.0))
INVERSION_TOL = 0.01  # kg/m3 of sigma2 lost downward between adjacent levels


@dataclass(frozen=True)
class OpGrid:
    sx: torch.Tensor  # (ny, nx) m between the two x neighbours of each point, nan on the edge columns
    sy: torch.Tensor  # (ny, nx) m between the two y neighbours, nan on the edge rows
    tan_r: torch.Tensor  # (ny, nx) tan(lat) / R, the Mercator metric term of divergence and vorticity
    f: torch.Tensor  # (ny, nx) 1/s
    depth: torch.Tensor  # (ny, nx) m
    levels: tuple[float, ...]  # m, positive down
    ocean: torch.Tensor  # (K, ny, nx) bool, real points per level
    region: torch.Tensor  # (ny, nx) bool, where statistics are taken
    sigma_px: dict[str, float]  # scale -> Gaussian sigma in cells


def op_grid(lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, ocean: np.ndarray, levels, region: np.ndarray,
            scales: dict[str, float] = SCALES_KM, device="cpu") -> OpGrid:
    """lon, lat, depth, region: (ny, nx); ocean: (K, ny, nx). Spacings come from the coordinates, so the image
    orientation does not matter."""
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    sx = np.full(lon.shape, np.nan)
    sy = np.full(lon.shape, np.nan)
    sx[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) * EARTH_RADIUS_M * np.cos(phi[:, 1:-1])
    sy[1:-1] = (phi[2:] - phi[:-2]) * EARTH_RADIUS_M
    side = float(np.nanmean(np.sqrt(np.abs(sx[1:-1, 1:-1] * sy[1:-1, 1:-1])))) / 2
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    return OpGrid(t(sx), t(sy), t(np.tan(phi) / EARTH_RADIUS_M), t(2 * OMEGA * np.sin(phi)), t(depth),
                  tuple(float(z) for z in levels), t(ocean, torch.bool), t(region, torch.bool),
                  {k: km * 1000.0 / side for k, km in scales.items()})


def _kernel(sigma: float, like: torch.Tensor) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=like.dtype, device=like.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(a: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = (len(k) - 1) // 2
    z = F.conv2d(a[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(z, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


def smooth(a: torch.Tensor, valid: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-normalized Gaussian smoothing of a (B, ny, nx) over valid (ny, nx): (value, ok), value 0 where not ok."""
    k = _kernel(sigma, a)
    m = valid.to(a.dtype).expand_as(a)
    num, den = _blur(torch.where(valid, a, 0.0), k), _blur(m, k)
    ok = valid & (den >= 0.5)
    return torch.where(ok, num / den.clamp_min(1e-6), 0.0), ok


def _centred(ok: torch.Tensor, dim: int) -> torch.Tensor:
    """ok at a point and both its neighbours along dim; False on the two edges."""
    n = ok.shape[dim]
    out = torch.zeros_like(ok)
    out.narrow(dim, 1, n - 2).copy_(ok.narrow(dim, 0, n - 2) & ok.narrow(dim, 1, n - 2) & ok.narrow(dim, 2, n - 2))
    return out


def ddx(s: torch.Tensor, ok: torch.Tensor, g: OpGrid) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.zeros_like(s)
    d[..., 1:-1] = (s[..., 2:] - s[..., :-2]) / g.sx[:, 1:-1]
    okd = _centred(ok, -1)
    return torch.where(okd, d, 0.0), okd


def ddy(s: torch.Tensor, ok: torch.Tensor, g: OpGrid) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.zeros_like(s)
    d[..., 1:-1, :] = (s[..., 2:, :] - s[..., :-2, :]) / g.sy[1:-1]
    okd = _centred(ok, -2)
    return torch.where(okd, d, 0.0), okd


def vorticity(u, v, valid, g: OpGrid, scale: str):
    """Relative vorticity of the smoothed flow, 1/s."""
    (su, oku), (sv, okv) = smooth(u, valid, g.sigma_px[scale]), smooth(v, valid, g.sigma_px[scale])
    (vx, a), (uy, b) = ddx(sv, okv, g), ddy(su, oku, g)
    return vx - uy + su * g.tan_r, a & b


def divergence(u, v, valid, g: OpGrid, scale: str):
    """Horizontal divergence of the smoothed flow, 1/s."""
    (su, oku), (sv, okv) = smooth(u, valid, g.sigma_px[scale]), smooth(v, valid, g.sigma_px[scale])
    (ux, a), (vy, b) = ddx(su, oku, g), ddy(sv, okv, g)
    return ux + vy - sv * g.tan_r, a & b


@dataclass(frozen=True)
class Tie:
    """lhs and rhs of one balance, (B, C, ny, nx) with C vector components, and where both are valid."""
    lhs: torch.Tensor
    rhs: torch.Tensor
    ok: torch.Tensor  # (B, ny, nx)

    @property
    def residual(self) -> torch.Tensor:
        return self.lhs - self.rhs


def geostrophic(u, v, ssh, valid, g: OpGrid, scale: str) -> Tie:
    s = g.sigma_px[scale]
    (su, oku), (sv, _), (se, oke) = smooth(u, valid, s), smooth(v, valid, s), smooth(ssh, valid, s)
    (ex, a), (ey, b) = ddx(se, oke, g), ddy(se, oke, g)
    return Tie(torch.stack([su, sv], 1), torch.stack([-GRAVITY / g.f * ey, GRAVITY / g.f * ex], 1), a & b & oku)


def thermal_wind(u1, v1, t1, s1, u2, v2, t2, s2, z1: float, z2: float, valid, g: OpGrid, scale: str) -> Tie:
    """Shear between depths z1 < z2 (positive down) of the smoothed flow, z up, against
    du/dz = g/(f rho0) drho/dy and dv/dz = -g/(f rho0) drho/dx, rho the mean sigma2 of the two levels."""
    s = g.sigma_px[scale]
    dz = z2 - z1
    (du, ok), (dv, _) = smooth((u1 - u2) / dz, valid, s), smooth((v1 - v2) / dz, valid, s)
    rho, okr = smooth(0.5 * (sigma2(t1, s1) + sigma2(t2, s2)), valid, s)
    (rx, a), (ry, b) = ddx(rho, okr, g), ddy(rho, okr, g)
    c = GRAVITY / (g.f * RHO0)
    return Tie(torch.stack([du, dv], 1), torch.stack([c * ry, -c * rx], 1), a & b & ok)


def continuity(ssh, ssh_prev, ub, vb, ub_prev, vb_prev, valid, g: OpGrid, scale: str) -> Tie:
    """One-day SSH change against minus the divergence of depth times the mean barotropic velocity of the day."""
    s = g.sigma_px[scale]
    de, ok = smooth((ssh - ssh_prev) / DAY_S, valid, s)
    div, okd = divergence(g.depth * 0.5 * (ub + ub_prev), g.depth * 0.5 * (vb + vb_prev), valid, g, scale)
    return Tie(de[:, None], -div[:, None], ok & okd)


def inversion_fraction(temp, salin, g: OpGrid) -> torch.Tensor:
    """(B,) share of region points and adjacent level pairs, both real, where sigma2 falls downward by more than
    INVERSION_TOL. temp, salin: (B, K, ny, nx)."""
    rho = sigma2(temp, salin)
    pair = g.ocean[1:] & g.ocean[:-1] & g.region
    inv = (rho[:, :-1] - rho[:, 1:] > INVERSION_TOL) & pair
    return inv.sum((1, 2, 3)).to(temp.dtype) / pair.sum()


def rms(x: torch.Tensor, ok: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """(B,) RMS over the region's valid points; x is (B, ny, nx) or (B, C, ny, nx) with C summed (vector norm)."""
    m = (ok & region).to(x.dtype)
    sq = x * x if x.dim() == 3 else (x * x).sum(1)
    return ((sq * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)).sqrt()
