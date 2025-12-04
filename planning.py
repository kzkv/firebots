# ChatGPT to draft cells weights

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
        trunk_tau: float,  # smaller = faster fade
        trunk_max_radius: int,  # apply gradient up to this many cells
) -> np.ndarray:
    """
    Returns a float cost grid (rows×cols). Use np.inf for impassable.
    - Fire cells: impassable
    - Fire ring 1: +fire_1, Fire ring 2: +fire_2 (Chebyshev distance)
    - Tree cells: impassable (if trunk_impassable)
    - Around trunks: + trunk_scale * exp(-(d-1)/trunk_tau), for d=1..trunk_max_radius
    """
    rows, cols = fire_grid.shape
    cost = np.full((rows, cols), float(base_cost), dtype=np.float32)

    # Fire: impassable + two rings of penalty (Chebyshev)
    cost[fire_grid] = np.inf
    ring1 = _ring_chebyshev(fire_grid, 1)
    ring2 = _ring_chebyshev(fire_grid, 2)
    cost[ring1 & np.isfinite(cost)] += fire_1
    cost[ring2 & np.isfinite(cost)] += fire_2

    # Trees: impassable + fast-fading gradient nearby
    cost[tree_grid] = np.inf

    # distance from tree trunks (0 at trunks)
    dist_tree = _chamfer_distance8(tree_grid.astype(bool))
    # apply only for cells not impassable and within radius
    mask = (dist_tree >= 1) & (dist_tree <= trunk_max_radius) & np.isfinite(cost)
    # exponential falloff from d=1 outward
    penalty = trunk_scale * np.exp(-(dist_tree - 1) / max(1e-6, trunk_tau))
    cost[mask] += penalty[mask].astype(cost.dtype)

    return cost


def _distance_point_to_cell_edge(px: float, py: float, cell_row: int, cell_col: int) -> float:
    """
    Compute the distance from a point to the nearest edge of a cell.

    The cell occupies the area [cell_col, cell_col+1] x [cell_row, cell_row+1].
    If the point is inside the cell, distance is 0.

    Args:
        px, py: Point coordinates (x=col, y=row)
        cell_row, cell_col: Cell indices

    Returns:
        Distance from point to nearest cell edge (0 if inside)
    """
    # Cell boundaries
    left = cell_col
    right = cell_col + 1
    top = cell_row
    bottom = cell_row + 1

    # Clamp point to cell boundaries to find nearest point on cell
    nearest_x = max(left, min(px, right))
    nearest_y = max(top, min(py, bottom))

    # Distance from point to nearest point on cell
    dx = px - nearest_x
    dy = py - nearest_y
    return math.sqrt(dx * dx + dy * dy)


def compute_fire_forbidden_zone(
    fire_grid: np.ndarray,
    robot_size: int = 3,
    margin: float = 0.5,
    resolution: int = 4,
) -> tuple[np.ndarray, int]:
    """
    Compute the forbidden zone for the robot center using Minkowski sum inflation.

    The robot is robot_size x robot_size cells. To prevent any part of the robot
    from entering fire, we inflate fire cells by (robot_size / 2 + margin) from
    the cell edges (not centers).

    Uses a finer subgrid for more precise boundary representation.

    Args:
        fire_grid: Boolean grid where True = fire cell
        robot_size: Robot footprint size (default 3 for 3x3)
        margin: Additional margin in fractional cells
        resolution: Subdivisions per cell (default 4, so each cell becomes 4x4 subcells)

    Returns:
        Tuple of (forbidden zone at subgrid resolution, resolution factor)
    """
    # Inflation distance from fire cell EDGE
    # For a 3x3 robot, center is 1.5 cells from edge, so we need 1.5 + margin
    inflation_dist = robot_size / 2.0 + margin

    rows, cols = fire_grid.shape
    sub_rows, sub_cols = rows * resolution, cols * resolution
    forbidden = np.zeros((sub_rows, sub_cols), dtype=bool)

    fire_positions = np.argwhere(fire_grid)

    # For efficiency, compute search range based on inflation distance
    search_range = int(math.ceil(inflation_dist)) + 1

    for fire_row, fire_col in fire_positions:
        # Check all subcells that could be within range
        r_min = max(0, (fire_row - search_range) * resolution)
        r_max = min(sub_rows, (fire_row + search_range + 1) * resolution)
        c_min = max(0, (fire_col - search_range) * resolution)
        c_max = min(sub_cols, (fire_col + search_range + 1) * resolution)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                # Subcell center in cell coordinates
                sub_center_x = (c + 0.5) / resolution
                sub_center_y = (r + 0.5) / resolution

                # Distance from subcell center to fire cell edge
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
    """
    Find the nearest point where the robot center can be to approach the fire.

    This finds subcells on the boundary of the forbidden zone (just outside it)
    that are closest to the robot's current position.

    Args:
        fire_grid: Boolean grid where True = fire cell
        robot_x: Robot center x position (in cells)
        robot_y: Robot center y position (in cells, corresponds to row)
        robot_size: Robot footprint size (default 3 for 3x3)
        margin: Additional margin in fractional cells
        resolution: Subdivisions per cell for precision

    Returns:
        (x, y) of the nearest approach point, or None if no valid point exists
    """
    rows, cols = fire_grid.shape

    # If no fire, return None
    if not fire_grid.any():
        return None

    # Compute forbidden zone at subgrid resolution
    forbidden, res = compute_fire_forbidden_zone(fire_grid, robot_size, margin, resolution)
    sub_rows, sub_cols = rows * res, cols * res

    # Find subcells that are:
    # 1. Not forbidden (robot can be here)
    # 2. Adjacent to forbidden zone (as close as possible to fire)

    # Dilate forbidden zone by 1 subcell to find the boundary
    boundary_zone = _dilate8(forbidden, 1) & ~forbidden

    # Also need to ensure we're within grid bounds with enough room for robot
    # Convert robot half-size to subcell units
    half_sub = (robot_size // 2) * res
    valid_mask = np.zeros((sub_rows, sub_cols), dtype=bool)
    valid_mask[half_sub:sub_rows-half_sub, half_sub:sub_cols-half_sub] = True

    # Candidate subcells: on boundary and valid for robot placement
    candidates = boundary_zone & valid_mask

    if not candidates.any():
        # No boundary found, try just finding closest non-forbidden subcell
        candidates = ~forbidden & valid_mask
        if not candidates.any():
            return None

    # Find the candidate closest to robot position
    candidate_positions = np.argwhere(candidates)

    # Convert robot position to subcell coordinates
    robot_sub_x = robot_x * res
    robot_sub_y = robot_y * res

    best_dist = float('inf')
    best_pos = None

    for sub_row, sub_col in candidate_positions:
        # Subcell center coordinates
        sub_center_x = sub_col + 0.5
        sub_center_y = sub_row + 0.5
        # Distance from robot to this subcell center
        dist = math.sqrt((sub_center_x - robot_sub_x) ** 2 + (sub_center_y - robot_sub_y) ** 2)
        if dist < best_dist:
            best_dist = dist
            # Convert back to cell coordinates
            best_pos = (sub_center_x / res, sub_center_y / res)

    return best_pos
