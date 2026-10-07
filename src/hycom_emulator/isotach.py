"""Loop Current front skill: Hausdorff distance between predicted and true 1.5 kt surface-speed isotachs,
and a differentiable surrogate of it for training (Karimi & Salcudean 2019, "Reducing the Hausdorff
distance in medical image segmentation with CNNs").

Surface velocity. Layer u/v in the B00 state (u_k01, v_k01) are the archv `u-vel.`/`v-vel.` records
(build_store.py S00_LAYER). archv (artype 1) layer velocities are baroclinic: HYCOM-tools
archv2data3z.f adds ubaro only when `artype.eq.1 .and. .not.baclin`. On the abozec_053 pack,
2025-07-13, depth > 500 m, the thickness-weighted vertical mean of layer u has rms 0.006 m/s against
ubaro 0.121 m/s (v: 0.005 vs 0.169). Total surface velocity is u_k01 + ubaro, v_k01 + vbaro.

C-grid. The store keeps u/v where HYCOM puts them: u(j,i) on the west face of p-cell (j,i), v(j,i) on
the south face. surface_speed averages the two faces to the p-point and repeats the last column/row
at the far edge.

Distances. The 385 x 525 grid is Mercator: 0.04 deg in lon, 0.04*cos(lat) deg in lat, so cells are
square with side R*0.04deg*cos(lat) (4.23 km at 18N, 3.77 km at 32N). The EDT runs in cell units
and is scaled by the cell size at the query pixel; one mid-latitude constant errs by up to 6 percent
at the domain's edges.

Layout. neural-lam flat index = ix*ny + iy (x outer). Land is 0 in a pack and NaN in a store; both
count as below the threshold.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

ISOTACH_MS = 0.7717  # 1.5 kt
EARTH_RADIUS_KM = 6371.0
_CROSS = generate_binary_structure(2, 1)


class FrontStatus(enum.StrEnum):
    OK = "ok"
    TRUE_EMPTY = "true_empty"
    PRED_EMPTY = "pred_empty"
    BOTH_EMPTY = "both_empty"


_STATUS = {  # (pred has a front, true has a front)
    (True, True): FrontStatus.OK,
    (True, False): FrontStatus.TRUE_EMPTY,
    (False, True): FrontStatus.PRED_EMPTY,
    (False, False): FrontStatus.BOTH_EMPTY,
}


@dataclass(frozen=True)
class FrontDistance:
    """Distances (km) between edge pixels of the two fronts; nan unless status is OK."""

    hausdorff_km: float
    mean_km: float
    p95_km: float
    status: FrontStatus


def to_grid(a, grid_shape: tuple[int, int]):
    """(..., N) flat -> (..., ny, nx); numpy or torch."""
    ny, nx = grid_shape
    return a.reshape(*a.shape[:-1], nx, ny).swapaxes(-1, -2)


def to_flat(a):
    """(..., ny, nx) -> (..., N) flat; numpy or torch."""
    return a.swapaxes(-1, -2).reshape(*a.shape[:-2], -1)


def surface_speed(u1, v1, ubaro, vbaro):
    """p-point speed from (..., ny, nx) C-grid layer-1 and barotropic velocities; numpy or torch.

    The 1e-12 keeps the gradient finite at zero speed (land in a pack).
    """
    u, v = u1 + ubaro, v1 + vbaro
    ny, nx = u.shape[-2:]
    east = list(range(1, nx)) + [nx - 1]
    north = list(range(1, ny)) + [ny - 1]
    up = 0.5 * (u + u[..., east])
    vp = 0.5 * (v + v[..., north, :])
    return (up**2 + vp**2 + 1e-12) ** 0.5


def mercator_cell_km(lat, dlon_deg: float = 0.04):
    return EARTH_RADIUS_KM * np.deg2rad(dlon_deg) * np.cos(np.deg2rad(lat))


def gulf_region(lon, lat, depth, min_depth_m: float = 500.0, lon_max: float = -81.0, lat_min: float = 21.5):
    """Ocean deeper than min_depth_m west of the Florida Straits and north of the Yucatan Channel."""
    return (np.nan_to_num(depth) > min_depth_m) & (lon <= lon_max) & (lat >= lat_min)


def front_edges(speed, region, threshold: float = ISOTACH_MS):
    """Pixels of region on the edge of {speed >= threshold}, the set taken over the whole (ny, nx) grid.

    Eroding the whole-grid set before restricting to region keeps the region boundary from creating
    front pixels. Land touching fast water is an edge; the default region's depth floor excludes it.
    """
    fast = np.nan_to_num(speed, nan=-np.inf) >= threshold
    return fast & ~binary_erosion(fast, _CROSS, border_value=0) & region


def distance_to_front_km(edges, cell_km):
    """km from every pixel to the nearest edge pixel; all zeros without edges, so an empty front adds
    no weight to isotach_loss."""
    if not edges.any():
        return np.zeros(edges.shape)
    return distance_transform_edt(~edges) * cell_km


def front_distance(speed_pred, speed_true, region, cell_km, threshold: float = ISOTACH_MS) -> FrontDistance:
    """Symmetric front distance; directed distances pool pred->true and true->pred (medpy hd95)."""
    ep, et = front_edges(speed_pred, region, threshold), front_edges(speed_true, region, threshold)
    status = _STATUS[bool(ep.any()), bool(et.any())]
    if status is not FrontStatus.OK:
        return FrontDistance(np.nan, np.nan, np.nan, status)
    to_true = distance_to_front_km(et, cell_km)[ep]
    to_pred = distance_to_front_km(ep, cell_km)[et]
    pooled = np.concatenate([to_true, to_pred])
    return FrontDistance(
        float(max(to_true.max(), to_pred.max())), float(pooled.mean()), float(np.percentile(pooled, 95)), status
    )


def isotach_target(speed_true, region, cell_km, threshold: float = ISOTACH_MS):
    """Hard target mask q and its d_q for one (ny, nx) field, float32."""
    q = (np.nan_to_num(speed_true, nan=-np.inf) >= threshold).astype(np.float32)
    d_q = distance_to_front_km(front_edges(speed_true, region, threshold), cell_km).astype(np.float32)
    return q, d_q


def _numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def isotach_loss(
    speed_pred,
    q,
    d_q,
    region,
    cell_km,
    tau: float = 0.05,
    alpha: float = 2.0,
    threshold: float = ISOTACH_MS,
    grid_shape: tuple[int, int] | None = None,
):
    """Mean over batch and region pixels of (p - q)^2 * (d_q^alpha + d_p^alpha), p = sigmoid((s - threshold)/tau).

    speed_pred is (B, ny, nx), or flat (B, N) with grid_shape=(ny, nx); q, d_q match it; region and
    cell_km are (ny, nx) or (N,) in the same layout. d_p comes from the detached hard predicted mask.
    """
    import torch

    flat = speed_pred.ndim == 2
    if flat and grid_shape is None:
        raise ValueError("flat (B, N) speed_pred needs grid_shape=(ny, nx)")
    hard, reg, cell = _numpy(speed_pred), _numpy(region).astype(bool), _numpy(cell_km)
    if flat:
        hard, reg = to_grid(hard, grid_shape), to_grid(reg, grid_shape)
        cell = to_grid(cell, grid_shape) if cell.ndim else cell
    d_p = np.stack([distance_to_front_km(front_edges(s, reg, threshold), cell) for s in hard])
    if flat:
        d_p = to_flat(d_p)
    like = {"dtype": speed_pred.dtype, "device": speed_pred.device}
    mask = torch.as_tensor(_numpy(region).astype(bool), device=speed_pred.device)
    p = torch.sigmoid((speed_pred[:, mask] - threshold) / tau)
    weight = d_q.to(**like)[:, mask] ** alpha + torch.as_tensor(d_p, **like)[:, mask] ** alpha
    return ((p - q.to(**like)[:, mask]) ** 2 * weight).mean()
