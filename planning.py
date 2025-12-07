# Planning utilities for Firebots
# RBE 550, Firebots (course project)
#
# Updated corridor parameters for smoother paths

import math
import numpy as np


def _dilate8(mask: np.ndarray, radius: int) -> np.ndarray:
    """8-neighborhood dilation by `radius` cells (pure NumPy)."""
    out = mask.astype(bool).copy()
    for _ in range(max(0, radius)):
        d = out
        up = np.pad(d[:-1, :], ((1, 0), (0, 0)))
        down = np.pad(d[1:, :], ((0, 1), (0, 0)))
        left = np.pad(d[:, :-1], ((0, 0), (1, 0)))
        right = np.pad(d[:, 1:], ((0, 0), (0, 1)))
        upleft = np.pad(d[:-1, :-1], ((1, 0), (1, 0)))
        upright = np.pad(d[:-1, 1:], ((1, 0), (0, 1)))
        downleft = np.pad(d[1:, :-1], ((0, 1), (1, 0)))
        downright = np.pad(d[1:, 1:], ((0, 1), (0, 1)))
        out = d | up | down | left | right | upleft | upright | downleft | downright
    return out


def _ring_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    """Cells at exact Chebyshev distance `radius` from mask (radius>=1)."""
    return _dilate8(mask, radius) & ~_dilate8(mask, radius - 1)


def _chamfer_distance8(seeds: np.ndarray) -> np.ndarray:
    """≈Euclidean distance transform (8-neighbor, weights 1 / √2)."""
    H, W = seeds.shape
    INF = 1e9
    SQ2 = 2 ** 0.5
    d = np.where(seeds, 0.0, INF).astype(np.float32)

    # forward pass
    for r in range(H):
        for c in range(W):
            v = d[r, c]
            if r:
                v = min(v, d[r - 1, c] + 1)
                if c:       v = min(v, d[r - 1, c - 1] + SQ2)
                if c < W - 1: v = min(v, d[r - 1, c + 1] + SQ2)
            if c: v = min(v, d[r, c - 1] + 1)
            d[r, c] = v

    # backward pass
    for r in range(H - 1, -1, -1):
        for c in range(W - 1, -1, -1):
            v = d[r, c]
            if r < H - 1:
                v = min(v, d[r + 1, c] + 1)
                if c:       v = min(v, d[r + 1, c - 1] + SQ2)
                if c < W - 1: v = min(v, d[r + 1, c + 1] + SQ2)
            if c < W - 1: v = min(v, d[r, c + 1] + 1)
            d[r, c] = v
    return d


def build_weight_grid(
        fire_grid: np.ndarray,
        tree_grid: np.ndarray,
        *,
        base_cost: float,
        fire_1: float,
        fire_2: float,
        trunk_scale: float,
        trunk_tau: float,
        trunk_max_radius: int,
) -> np.ndarray:
    """
    Returns a float cost grid (rows×cols). Use np.inf for impassable.
    """
    rows, cols = fire_grid.shape
    cost = np.full((rows, cols), float(base_cost), dtype=np.float32)

    # Fire: impassable + two rings of penalty (Chebyshev)
    cost[fire_grid] = np.inf
    ring1 = _ring_chebyshev(fire_grid, 1)
    ring2 = _ring_chebyshev(fire_grid, 2)
    cost[ring1 & np.isfinite(cost)] += fire_1
    cost[ring2 & np.isfinite(cost)] += fire_2

    # Trees: impassable + gradient nearby
    cost[tree_grid] = np.inf

    dist_tree = _chamfer_distance8(tree_grid.astype(bool))
    mask = (dist_tree >= 1) & (dist_tree <= trunk_max_radius) & np.isfinite(cost)
    penalty = trunk_scale * np.exp(-(dist_tree - 1) / max(1e-6, trunk_tau))
    cost[mask] += penalty[mask].astype(cost.dtype)

    return cost


def _distance_point_to_cell_edge(px: float, py: float, cell_row: int, cell_col: int) -> float:
    """Compute the distance from a point to the nearest edge of a cell."""
    left = cell_col
    right = cell_col + 1
    top = cell_row
    bottom = cell_row + 1

    nearest_x = max(left, min(px, right))
    nearest_y = max(top, min(py, bottom))

    dx = px - nearest_x
    dy = py - nearest_y
    return math.sqrt(dx * dx + dy * dy)


def compute_fire_forbidden_zone(
        fire_grid: np.ndarray,
        robot_size: int = 3,
        margin: float = 0.5,
        resolution: int = 4,
) -> tuple[np.ndarray, int]:
    """Compute the forbidden zone for the robot center using Minkowski sum inflation."""
    inflation_dist = robot_size / 2.0 + margin

    rows, cols = fire_grid.shape
    sub_rows, sub_cols = rows * resolution, cols * resolution
    forbidden = np.zeros((sub_rows, sub_cols), dtype=bool)

    fire_positions = np.argwhere(fire_grid)
    search_range = int(math.ceil(inflation_dist)) + 1

    for fire_row, fire_col in fire_positions:
        r_min = max(0, (fire_row - search_range) * resolution)
        r_max = min(sub_rows, (fire_row + search_range + 1) * resolution)
        c_min = max(0, (fire_col - search_range) * resolution)
        c_max = min(sub_cols, (fire_col + search_range + 1) * resolution)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                sub_center_x = (c + 0.5) / resolution
                sub_center_y = (r + 0.5) / resolution
                dist = _distance_point_to_cell_edge(sub_center_x, sub_center_y, fire_row, fire_col)
                if dist <= inflation_dist:
                    forbidden[r, c] = True

    return forbidden, resolution


def find_nearest_fire_approach_point(
        fire_grid: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_size: int = 3,
        margin: float = 0.5,
        resolution: int = 4,
) -> tuple[float, float] | None:
    """Find the nearest point where the robot center can be to approach the fire."""
    rows, cols = fire_grid.shape

    if not fire_grid.any():
        return None

    forbidden, res = compute_fire_forbidden_zone(fire_grid, robot_size, margin, resolution)
    sub_rows, sub_cols = rows * res, cols * res

    boundary_zone = _dilate8(forbidden, 1) & ~forbidden

    half_sub = (robot_size // 2) * res
    valid_mask = np.zeros((sub_rows, sub_cols), dtype=bool)
    valid_mask[half_sub:sub_rows - half_sub, half_sub:sub_cols - half_sub] = True

    candidates = boundary_zone & valid_mask

    if not candidates.any():
        candidates = ~forbidden & valid_mask
        if not candidates.any():
            return None

    candidate_positions = np.argwhere(candidates)

    robot_sub_x = robot_x * res
    robot_sub_y = robot_y * res

    best_dist = float('inf')
    best_pos = None

    for sub_row, sub_col in candidate_positions:
        sub_center_x = sub_col + 0.5
        sub_center_y = sub_row + 0.5
        dist = math.sqrt((sub_center_x - robot_sub_x) ** 2 + (sub_center_y - robot_sub_y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_pos = (sub_center_x / res, sub_center_y / res)

    return best_pos


# ============================================================================
# UPDATED CORRIDOR FUNCTIONS - Key changes for smoother paths
# ============================================================================

def compute_fire_distance_field(fire_grid: np.ndarray) -> np.ndarray:
    """Compute the distance from each cell to the nearest fire cell."""
    return _chamfer_distance8(fire_grid)


def build_fire_corridor_cost(
        fire_grid: np.ndarray,
        fire_distance: np.ndarray,
        robot_size: int = 3,
        min_margin: float = 1.0,
        ideal_distance: float = 6.0,
        corridor_width: float = 8.0,
        falloff_rate: float = 0.2,
) -> np.ndarray:
    """
    Build cost field showing the corridor around fire.

    UPDATED: Wider corridor, gentler costs for smoother paths.

    Args:
        fire_grid: Boolean grid of fire cells
        fire_distance: Distance field from fire
        robot_size: Robot footprint size (default 3 for 3x3)
        min_margin: Extra margin beyond robot radius (cells)
        ideal_distance: Center of the ideal corridor
        corridor_width: Width of the low-cost corridor band
        falloff_rate: How fast cost increases outside corridor (lower = gentler)
    """
    cost = np.zeros_like(fire_distance, dtype=np.float32)

    # Minimum safe distance = robot radius + margin
    min_distance = (robot_size / 2.0) + min_margin

    # Too close = impassable
    cost[fire_distance < min_distance] = np.inf

    # Define corridor bounds
    corridor_inner = max(min_distance, ideal_distance - corridor_width / 2)
    corridor_outer = ideal_distance + corridor_width / 2

    # Transition zone: min_distance to corridor_inner
    # Gentle ramp - not too expensive, just slightly prefer corridor
    transition = (fire_distance >= min_distance) & (fire_distance < corridor_inner)
    if transition.any():
        # Linear ramp from 1.5 at min_distance to 1.0 at corridor_inner
        t = (fire_distance[transition] - min_distance) / max(corridor_inner - min_distance, 0.1)
        cost[transition] = 1.5 - 0.5 * t

    # Inside corridor = base cost (lowest)
    in_corridor = (fire_distance >= corridor_inner) & (fire_distance <= corridor_outer)
    cost[in_corridor] = 1.0

    # Beyond corridor = gentle increase
    # Key: falloff_rate is LOW (0.2) so paths don't zigzag to stay in corridor
    beyond = fire_distance > corridor_outer
    if beyond.any():
        cost[beyond] = 1.0 + (fire_distance[beyond] - corridor_outer) * falloff_rate

    return cost


def rebuild_weight_grid(fire_grid, fire_distance, known_trees, robot_size: int = 3):
    """
    Rebuild weight grid using currently known obstacles.

    UPDATED: Gentler parameters for smoother paths.
    """
    corridor_cost = build_fire_corridor_cost(
        fire_grid,
        fire_distance,
        robot_size=robot_size,
        min_margin=1.0,  # Safe distance from fire
        ideal_distance=5.0,  # Where robot ideally operates
        corridor_width=6.0,  # Wide corridor (6 +/- 4 cells)
        falloff_rate=1,  # GENTLE slope outside corridor
    )

    tree_cost = build_weight_grid(
        fire_grid,
        known_trees,
        base_cost=0.0,  # No base cost (corridor provides it)
        fire_1=0.0,  # Fire penalty handled by corridor
        fire_2=0.0,
        trunk_scale=10.0,  # REDUCED from 10.0 - gentler tree avoidance
        trunk_tau=.5,  # INCREASED from 0.5 - slower falloff
        trunk_max_radius=6,  # Slightly larger radius
    )

    return corridor_cost + tree_cost