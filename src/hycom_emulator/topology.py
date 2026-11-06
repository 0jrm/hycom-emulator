"""H0 superlevel persistence, its merge forest, the correlated-noise null and the Loop Current LC17 rules.

Vendored from persistence-ocean (Jose Miranda's local repository, branch feat/write-up-assets at 205e3c3):
boundary.py, cubical_h0.py, merge_tree.py, null_model.py and loop_current.py. Only what hycom_emulator.structure
uses is here, with the logic unchanged: cells enter by value descending (ties by flat index), components merge by
the elder rule (the older extremum survives), and an open boundary is a virtual vertex born at +inf that every cell
touching land or the grid edge joins. numba is optional: without it the union-find runs as plain Python, which is
fast enough on the 0.25 deg Gulf (about 2000 cells). Pandas tables keep the reference's column names, so results
compare row for row. Axis 0 is latitude, axis 1 longitude; NaN is land.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

try:
    from numba import njit
except ImportError:  # the union-find is the same Python either way
    def njit(**_):
        return lambda f: f

BOUNDARY_PARENT = -1
ESSENTIAL_PARENT = -2
LAND_OWNER = -3
NEVER = -1
SAME_EXTREMUM = -1
NEVER_CONNECTED = np.iinfo(np.int64).max
NEIGHBOUR_OFFSETS = {
    4: np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64),
    8: np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64),
}


# ------------------------------------------------------------------------------------------- boundary.py

def open_boundary_cells(valid_mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Valid cells with a neighbour that is invalid or off-grid."""
    padded = np.pad(np.asarray(valid_mask, bool), 1, constant_values=False)
    n_rows, n_cols = valid_mask.shape
    touches = np.zeros(valid_mask.shape, bool)
    for dr, dc in NEIGHBOUR_OFFSETS[connectivity]:
        touches |= ~padded[1 + dr:1 + dr + n_rows, 1 + dc:1 + dc + n_cols]
    return valid_mask & touches


# ----------------------------------------------------------------------------------------- cubical_h0.py

@njit(cache=True)
def _find(union_parent, node):
    while union_parent[node] != node:
        union_parent[node] = union_parent[union_parent[node]]
        node = union_parent[node]
    return node


@njit(cache=True)
def _elder_rule_union_find(cells_by_rank, is_boundary, n_rows, n_cols, offsets):
    n_cells = n_rows * n_cols
    boundary_node = n_cells
    max_extrema = cells_by_rank.size
    insertion_rank = np.full(n_cells, NEVER, dtype=np.int64)
    owner = np.full(n_cells, LAND_OWNER, dtype=np.int64)
    union_parent = np.arange(n_cells + 1, dtype=np.int64)
    union_size = np.ones(n_cells + 1, dtype=np.int64)
    root_extremum = np.full(n_cells + 1, BOUNDARY_PARENT, dtype=np.int64)
    extremum_cell = np.empty(max_extrema, dtype=np.int64)
    parent = np.full(max_extrema, ESSENTIAL_PARENT, dtype=np.int64)
    first_merge_rank = np.full(max_extrema, NEVER, dtype=np.int64)
    death_rank = np.full(max_extrema, NEVER, dtype=np.int64)
    death_cell = np.full(max_extrema, NEVER, dtype=np.int64)
    n_extrema = 0
    neighbour_roots = np.empty(offsets.shape[0] + 1, dtype=np.int64)
    for rank in range(cells_by_rank.size):
        cell = cells_by_rank[rank]
        row = cell // n_cols
        col = cell % n_cols
        n_roots = 0
        if is_boundary[cell]:
            neighbour_roots[0] = _find(union_parent, boundary_node)
            n_roots = 1
        for k in range(offsets.shape[0]):
            nr = row + offsets[k, 0]
            nc = col + offsets[k, 1]
            if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                continue
            neighbour = nr * n_cols + nc
            if insertion_rank[neighbour] == NEVER:
                continue
            root = _find(union_parent, neighbour)
            seen = False
            for m in range(n_roots):
                if neighbour_roots[m] == root:
                    seen = True
                    break
            if not seen:
                neighbour_roots[n_roots] = root
                n_roots += 1
        insertion_rank[cell] = rank
        if n_roots == 0:
            extremum_cell[n_extrema] = cell
            root_extremum[cell] = n_extrema
            owner[cell] = n_extrema
            n_extrema += 1
            continue
        survivor_root = neighbour_roots[0]
        for m in range(1, n_roots):
            if root_extremum[neighbour_roots[m]] < root_extremum[survivor_root]:
                survivor_root = neighbour_roots[m]
        survivor = root_extremum[survivor_root]
        if n_roots >= 2:
            for m in range(n_roots):
                extremum = root_extremum[neighbour_roots[m]]
                if extremum >= 0 and first_merge_rank[extremum] == NEVER:
                    first_merge_rank[extremum] = rank
        merged_root = survivor_root
        for m in range(n_roots):
            root = neighbour_roots[m]
            if root == survivor_root:
                continue
            dying = root_extremum[root]
            parent[dying] = survivor
            death_rank[dying] = rank
            death_cell[dying] = cell
            if union_size[root] > union_size[merged_root]:
                union_parent[merged_root] = root
                union_size[root] += union_size[merged_root]
                merged_root = root
            else:
                union_parent[root] = merged_root
                union_size[merged_root] += union_size[root]
        root_extremum[merged_root] = survivor
        union_parent[cell] = merged_root
        union_size[merged_root] += 1
        owner[cell] = survivor
    return (owner, insertion_rank, extremum_cell[:n_extrema], parent[:n_extrema], first_merge_rank[:n_extrema],
            death_rank[:n_extrema], death_cell[:n_extrema])


@dataclass(frozen=True)
class MergeForest:
    owner: np.ndarray  # (ny, nx) extremum id owning each cell; LAND_OWNER on land
    insertion_rank: np.ndarray  # (ny, nx); NEVER on land
    parent: np.ndarray  # per extremum: the extremum it died into, BOUNDARY_PARENT or ESSENTIAL_PARENT
    first_merge_rank: np.ndarray
    death_rank: np.ndarray


@dataclass(frozen=True)
class Pairs:
    """Pairs of positive persistence: birth and death in field units (death -inf for an essential class); sign -1
    for cyclones, whose birth (the minimum) is below their death (the saddle)."""
    birth: np.ndarray
    death: np.ndarray
    extremum_i: np.ndarray
    extremum_j: np.ndarray
    sign: float = 1.0

    @property
    def persistence(self) -> np.ndarray:
        return self.sign * (self.birth - self.death)


def compute_superlevel_persistence(values: np.ndarray, connectivity: int = 4, open_boundary: bool = False
                                   ) -> tuple[Pairs, MergeForest]:
    values = np.asarray(values, dtype=np.float64)
    n_rows, n_cols = values.shape
    valid = ~np.isnan(values)
    flat = values.ravel()
    cells = np.flatnonzero(valid)
    ranked = cells[np.argsort(-flat[cells], kind="stable")]
    is_boundary = (open_boundary_cells(valid, connectivity) if open_boundary else np.zeros(valid.shape, bool)).ravel()
    owner, rank, ext_cell, parent, first_merge, death_rank, death_cell = _elder_rule_union_find(
        ranked, is_boundary, n_rows, n_cols, NEIGHBOUR_OFFSETS[connectivity])
    forest = MergeForest(owner.reshape(values.shape), rank.reshape(values.shape), parent, first_merge, death_rank)
    birth = flat[ext_cell]
    death = np.where(death_cell == NEVER, -np.inf, flat[np.maximum(death_cell, 0)])
    keep = birth - death > 0
    return Pairs(birth[keep], death[keep], ext_cell[keep] // n_cols, ext_cell[keep] % n_cols), forest


def compute_eddy_persistence(h: np.ndarray, **kw) -> dict[str, Pairs]:
    """Anticyclones from the superlevel filtration of h, cyclones from that of -h, every value in h units."""
    anti, _ = compute_superlevel_persistence(h, **kw)
    cyc, _ = compute_superlevel_persistence(-np.asarray(h, dtype=np.float64), **kw)
    return {"anticyclone": anti, "cyclone": Pairs(-cyc.birth, -cyc.death, cyc.extremum_i, cyc.extremum_j, -1.0)}


def diagram(p: Pairs, polarity: str, floor: float) -> np.ndarray:
    """(k, 2) pairs above floor oriented so birth > death (cyclones negated back)."""
    d = np.stack([p.birth, p.death], 1)
    d = d if polarity == "anticyclone" else -d
    return d[d[:, 0] - d[:, 1] >= floor]


# ---------------------------------------------------------------------------------------- merge_tree.py

def _ancestor_chain(forest: MergeForest, extremum_id: int) -> list[int]:
    chain = [int(extremum_id)]
    while chain[-1] >= 0 and forest.parent[chain[-1]] != ESSENTIAL_PARENT:
        chain.append(int(forest.parent[chain[-1]]))
    return chain


def connection_rank(forest: MergeForest, a: int, b: int) -> int:
    """Insertion rank at which extrema a and b first lie in one superlevel component (a later rank is a lower
    level): the later death rank of the two chain nodes just below their lowest common ancestor."""
    chain_a = _ancestor_chain(forest, a)
    position_in_a = {node: k for k, node in enumerate(chain_a)}
    chain_b = _ancestor_chain(forest, b)
    for k_b, node in enumerate(chain_b):
        k_a = position_in_a.get(node)
        if k_a is None:
            continue
        below = [chain[k - 1] for chain, k in ((chain_a, k_a), (chain_b, k_b)) if k > 0]
        return max((int(forest.death_rank[n]) for n in below), default=SAME_EXTREMUM)
    return NEVER_CONNECTED


def cell_connection_rank(forest: MergeForest, cell_ij, extremum_id: int) -> int:
    return max(int(forest.insertion_rank[cell_ij]), connection_rank(forest, forest.owner[cell_ij], extremum_id))


def cells_by_rank(forest: MergeForest) -> np.ndarray:
    flat_rank = forest.insertion_rank.ravel()
    inserted = np.flatnonzero(flat_rank != NEVER)
    cells = np.empty(inserted.size, dtype=np.int64)
    cells[flat_rank[inserted]] = inserted
    return cells


def level_at_rank(values: np.ndarray, ranked_cells: np.ndarray, rank: int) -> float:
    if rank == SAME_EXTREMUM:
        return np.inf
    if rank == NEVER_CONNECTED:
        return -np.inf
    return float(np.ravel(values)[ranked_cells[rank]])


# ---------------------------------------------------------------------------------------- null_model.py

EARTH_RADIUS_KM = 6371.0


def grid_spacing_km(latitude: np.ndarray, longitude: np.ndarray) -> tuple[float, float]:
    dy = np.deg2rad(np.mean(np.diff(latitude))) * EARTH_RADIUS_KM
    dx = np.deg2rad(np.mean(np.diff(longitude))) * EARTH_RADIUS_KM * np.cos(np.deg2rad(np.mean(latitude)))
    return dy, dx


def spatial_surrogates(err: np.ndarray, latitude, longitude, length_km: float, n: int, seed: int) -> np.ndarray:
    """(n, ny, nx) independent fields of std err (map or scalar) with covariance exp(-r^2 / (2 L^2)); the domain is
    padded by 3 L so the FFT does not wrap opposite edges together (null_model.generate_surrogates without the
    AR(1) time step)."""
    rng = np.random.default_rng(seed)
    ny, nx = len(latitude), len(longitude)
    dy, dx = grid_spacing_km(latitude, longitude)
    py, px = int(np.ceil(3 * length_km / dy)), int(np.ceil(3 * length_km / dx))
    shape = (ny + 2 * py, nx + 2 * px)
    ky = np.fft.fftfreq(shape[0], d=dy) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=dx) * 2 * np.pi
    spectrum = np.exp(-0.5 * (ky[:, None] ** 2 + kx[None, :] ** 2) * (length_km / np.sqrt(2.0)) ** 2)
    spectrum /= np.sqrt(np.mean(spectrum**2))
    out = np.empty((n, ny, nx))
    for k in range(n):
        out[k] = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * spectrum).real[py:py + ny, px:px + nx]
    return out * np.asarray(err)


# -------------------------------------------------------------------------------------- loop_current.py

SEED_BOX = {"west": -86.6, "east": -85.2, "south": 21.5, "north": 22.0}
LC_CONTOUR = 0.17
CANDIDATE_NORTH_OF = 24.0
CANDIDATE_WEST_OF = -87.0
TRACK_RADIUS_DEG = 1.0
REATTACH_WINDOW_DAYS = 30


def gulf_region(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Gulf of Mexico by three cuts: Caribbean south of the Yucatan Channel, Gulf of Batabano south of Cuba, and
    everything east of 81 W."""
    lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
    return ~(((lat < 21.5) & (lon > -87.5)) | ((lat < 22.75) & (lon > -84.0)) | (lon > -81.0))


@dataclass(frozen=True)
class GulfGrid:
    latitude: np.ndarray
    longitude: np.ndarray
    gulf: np.ndarray
    seed: np.ndarray

    def lat_lon(self, flat_cell):
        i, j = np.unravel_index(flat_cell, self.gulf.shape)
        return float(self.latitude[i]), float(self.longitude[j])


def gulf_grid(ocean: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> GulfGrid:
    """The Gulf is the 4-connected piece of ocean inside gulf_region that holds the seed box; the seed is every such
    cell inside SEED_BOX."""
    lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
    in_box = (lat >= SEED_BOX["south"]) & (lat <= SEED_BOX["north"]) & (lon >= SEED_BOX["west"]) & (lon <= SEED_BOX["east"])
    labels, _ = ndimage.label(ocean & gulf_region(latitude, longitude))
    seed_labels = np.unique(labels[in_box & (labels > 0)])
    if seed_labels.size != 1:
        raise ValueError(f"seed box must lie in one ocean piece, got labels {seed_labels}")
    gulf = labels == seed_labels[0]
    return GulfGrid(latitude, longitude, gulf, gulf & in_box)


@dataclass(frozen=True)
class GulfAnomaly:
    dates: pd.DatetimeIndex
    values: np.ndarray  # (T, ny, nx), NaN outside the Gulf
    gulf_mean: np.ndarray


def gulf_anomaly(adt: np.ndarray, dates, grid: GulfGrid) -> GulfAnomaly:
    adt = np.asarray(adt, dtype=np.float64)
    mean = adt[:, grid.gulf].mean(axis=1)
    return GulfAnomaly(pd.DatetimeIndex(dates), np.where(grid.gulf, adt - mean[:, None, None], np.nan), mean)


def highest_seed_cell(field, grid: GulfGrid) -> int:
    seed_cells = np.flatnonzero(grid.seed.ravel())
    return int(seed_cells[np.argmax(field.ravel()[seed_cells])])


def in_lce_region(latitude, longitude):
    return (np.asarray(latitude) > CANDIDATE_NORTH_OF) | (np.asarray(longitude) < CANDIDATE_WEST_OF)


def seed_side_peak(field, seed_cell: int, level: float) -> float:
    labels, _ = ndimage.label(np.nan_to_num(field, nan=-np.inf) > level)
    seed_label = labels.ravel()[seed_cell]
    return float(level) if seed_label == 0 else float(field[labels == seed_label].max())


def neck_saddles(field, grid: GulfGrid) -> pd.DataFrame:
    """Every maximum of the Gulf field with its connection level to the highest seed cell and its neck depth."""
    _, forest = compute_superlevel_persistence(field)
    ranked = cells_by_rank(forest)
    flat = field.ravel()
    seed_cell = highest_seed_cell(field, grid)
    seed_ij = np.unravel_index(seed_cell, field.shape)
    ids, birth_ranks = np.unique(forest.owner.ravel()[ranked], return_index=True)
    rows = []
    for extremum_id, birth_rank in zip(ids, birth_ranks):
        max_cell = int(ranked[birth_rank])
        saddle_cell = int(ranked[cell_connection_rank(forest, seed_ij, extremum_id)])
        max_lat, max_lon = grid.lat_lon(max_cell)
        rows.append({"max_i": max_cell // field.shape[1], "max_j": max_cell % field.shape[1], "max_lat": max_lat,
                     "max_lon": max_lon, "h_max": flat[max_cell], "h_saddle": flat[saddle_cell],
                     "saddle_cell": saddle_cell, "saddle_at_seed": saddle_cell == seed_cell})
    s = pd.DataFrame(rows)
    s["p"] = s["h_max"] - s["h_saddle"]
    peaks = {c: seed_side_peak(field, seed_cell, flat[c]) for c in np.unique(s["saddle_cell"])}
    s["seed_side_peak"] = s["saddle_cell"].map(peaks)
    s["neck_depth"] = (np.minimum(s["h_max"], s["seed_side_peak"]) - s["h_saddle"]).clip(lower=0.0)
    return s


def lc17_labels(field, grid: GulfGrid):
    labels, _ = ndimage.label(field >= LC_CONTOUR)
    return labels, int(labels.ravel()[highest_seed_cell(field, grid)])


@dataclass(frozen=True)
class TouchRule:
    """A rejoin counts only when the touch depth is at least min_depth_m on more than longer_than_days days in a row."""
    min_depth_m: float
    longer_than_days: int


@dataclass(frozen=True)
class PieceFate:
    fate: str
    day: int
    ignored_touches: dict


def _component_holding(mask, cell_ij):
    labels, _ = ndimage.label(mask)
    if labels[cell_ij] == 0:
        only = np.zeros_like(mask)
        only[cell_ij] = True
        return only
    return labels == labels[cell_ij]


def connection_to_seed(field, grid: GulfGrid, cell_ij) -> float:
    _, forest = compute_superlevel_persistence(field)
    seed_ij = np.unravel_index(highest_seed_cell(field, grid), field.shape)
    rank = max(int(forest.insertion_rank[cell_ij]), cell_connection_rank(forest, seed_ij, forest.owner[cell_ij]))
    return float(field.ravel()[cells_by_rank(forest)[rank]])


def _piece_fate(piece, start_day, all_labels, fields, grid, rule: TouchRule) -> PieceFate:
    ignored, run = {}, []
    for day in range(start_day + 1, len(all_labels)):
        labels, lc_label = all_labels[day]
        overlapping = labels[piece & (labels > 0)]
        if overlapping.size == 0:
            return PieceFate("dissipated", day, ignored)
        if not (lc_label and np.any(overlapping == lc_label)):
            run = []
            piece = labels == np.bincount(overlapping).argmax()
            continue
        field = fields[day]
        touch = piece & (labels == lc_label)
        max_cell = int(np.flatnonzero(touch.ravel())[np.argmax(field[touch])])
        max_ij = np.unravel_index(max_cell, field.shape)
        level = connection_to_seed(field, grid, max_ij)
        piece = _component_holding(np.nan_to_num(field, nan=-np.inf) > level, max_ij)
        depth = level - LC_CONTOUR
        ignored[day] = (depth, piece)
        run = run + [day] if depth >= rule.min_depth_m else []
        if len(run) > rule.longer_than_days:
            for d in run:
                del ignored[d]
            return PieceFate("reattached", run[0], ignored)
    return PieceFate("record_end", len(all_labels) - 1, ignored)


def classify_detachment(fate: str, days_off_lc17: int) -> str:
    if days_off_lc17 > REATTACH_WINDOW_DAYS:
        return "separated"
    return {"reattached": "reattached", "dissipated": "dissipated", "record_end": "censored"}[fate]


def lc17_detachments(anomaly: GulfAnomaly, grid: GulfGrid, rule: TouchRule) -> pd.DataFrame:
    """Every day a >= 17 cm component that overlapped yesterday's LC17 no longer belongs to today's, with the
    piece's fate and classification."""
    all_labels = [lc17_labels(f, grid) for f in anomaly.values]
    ignored_masks: dict[int, list] = {}
    rows = []
    for day in range(1, len(anomaly.dates)):
        prev_labels, prev_lc = all_labels[day - 1]
        labels, lc_label = all_labels[day]
        if not prev_lc:
            continue
        for piece_label in np.unique(labels[(prev_labels == prev_lc) & (labels > 0)]):
            if piece_label == lc_label:
                continue
            piece = labels == piece_label
            if any(np.any(piece & m) for m in ignored_masks.get(day - 1, [])):
                continue
            field = anomaly.values[day]
            max_cell = int(np.flatnonzero(piece.ravel())[np.argmax(field[piece])])
            max_i, max_j = np.unravel_index(max_cell, field.shape)
            max_lat, max_lon = grid.lat_lon(max_cell)
            fate = _piece_fate(piece, day, all_labels, anomaly.values, grid, rule)
            for touch_day, (_, mask) in fate.ignored_touches.items():
                ignored_masks.setdefault(touch_day, []).append(mask)
            depths = [d for d, _ in fate.ignored_touches.values()]
            outcome = classify_detachment(fate.fate, fate.day - day)
            rows.append({"date": anomaly.dates[day], "day": day, "piece_max_i": int(max_i), "piece_max_j": int(max_j),
                         "piece_max_lat": max_lat, "piece_max_lon": max_lon, "piece_h_max": field.ravel()[max_cell],
                         "piece_pixels": int(piece.sum()), "lc_present": bool(lc_label), "outcome": outcome,
                         "fate": fate.fate, "fate_date": anomaly.dates[fate.day],
                         "final_separation": outcome == "separated" and fate.fate != "reattached",
                         "in_lce_region": bool(in_lce_region(max_lat, max_lon)), "n_ignored_touch_days": len(depths),
                         "max_ignored_touch_depth_m": max(depths, default=np.nan)})
    return pd.DataFrame(rows)


def separation_events(detachments: pd.DataFrame) -> pd.DataFrame:
    """Each final separation in the LCE region (or one censored at the record end), with the LCE-region detachments
    that reattached since the previous final separation."""
    if detachments.empty:
        return pd.DataFrame()
    events, pending = [], []
    for row in detachments[detachments["in_lce_region"]].sort_values("date").itertuples():
        if row.fate == "reattached":
            pending.append(row)
        elif row.final_separation or row.outcome == "censored":
            events.append({"separation_date": row.date, "day": row.day, "censored": not row.final_separation,
                           "piece_max_lat": row.piece_max_lat, "piece_max_lon": row.piece_max_lon,
                           "piece_pixels": row.piece_pixels, "piece_fate": row.fate, "piece_fate_date": row.fate_date,
                           "first_detachment_date": pending[0].date if pending else row.date,
                           "n_reattachments": len(pending),
                           "n_late_reattachments": sum(p.outcome == "separated" for p in pending)})
            pending = []
    return pd.DataFrame(events)


def gulf_maxima(anomaly: GulfAnomaly, grid: GulfGrid) -> pd.DataFrame:
    return pd.concat([neck_saddles(f, grid).assign(date=d) for d, f in zip(anomaly.dates, anomaly.values)], ignore_index=True)


def follow_lce_back(maxima: pd.DataFrame, event, earliest_date) -> pd.DataFrame:
    """The separating eddy's maximum followed from the separation date back to earliest_date and forward
    REATTACH_WINDOW_DAYS days, each day the LCE-region maximum nearest the previous day's within TRACK_RADIUS_DEG."""
    in_region = maxima[in_lce_region(maxima["max_lat"], maxima["max_lon"])]
    by_date = dict(tuple(in_region.groupby("date")))
    steps = []
    for direction, n_days in ((-1, (event["separation_date"] - earliest_date).days), (1, REATTACH_WINDOW_DAYS)):
        lat, lon = event["piece_max_lat"], event["piece_max_lon"]
        for k in range(0 if direction < 0 else 1, n_days + 1):
            day = by_date.get(event["separation_date"] + pd.Timedelta(days=direction * k))
            if day is None:
                break
            distance = np.hypot(day["max_lat"] - lat, day["max_lon"] - lon)
            if distance.min() > TRACK_RADIUS_DEG:
                break
            nearest = day.loc[distance.idxmin()]
            steps.append(nearest)
            lat, lon = nearest["max_lat"], nearest["max_lon"]
    if not steps:
        return pd.DataFrame(columns=["date", "days_from_separation", "neck_depth", "h_max", "h_saddle"])
    track = pd.DataFrame(steps).sort_values("date").reset_index(drop=True)
    return track.assign(days_from_separation=(track["date"] - event["separation_date"]).dt.days)
