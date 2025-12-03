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


def compute_fire_forbidden_zone(
    fire_grid: np.ndarray,
    robot_size: int = 3,
    margin: float = 0.5,
) -> np.ndarray:
    """
    Compute the forbidden zone for the robot center using Minkowski sum inflation.

    The robot is robot_size x robot_size cells. To prevent any part of the robot
    from entering fire, we inflate the fire cells by (robot_size // 2 + margin).

    Args:
        fire_grid: Boolean grid where True = fire cell
        robot_size: Robot footprint size (default 3 for 3x3)
        margin: Additional margin in fractional cells

    Returns:
        Boolean grid where True = robot center cannot be here
    """
    # Minkowski radius: half the robot size (rounded down) + margin
    # For a 3x3 robot, the center is 1 cell from edge, so inflate by 1 + margin
    inflation_radius = robot_size // 2 + margin

    rows, cols = fire_grid.shape
    forbidden = np.zeros((rows, cols), dtype=bool)

    # For each fire cell, mark all cells within inflation_radius as forbidden
    fire_positions = np.argwhere(fire_grid)

    for fire_row, fire_col in fire_positions:
        # Check all cells that could be within range
        r_min = max(0, int(fire_row - inflation_radius - 1))
        r_max = min(rows, int(fire_row + inflation_radius + 2))
        c_min = max(0, int(fire_col - inflation_radius - 1))
        c_max = min(cols, int(fire_col + inflation_radius + 2))

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                # Distance from cell center to fire cell center
                dist = math.sqrt((r - fire_row) ** 2 + (c - fire_col) ** 2)
                if dist <= inflation_radius:
                    forbidden[r, c] = True

    return forbidden


def find_nearest_fire_approach_point(
    fire_grid: np.ndarray,
    robot_x: float,
    robot_y: float,
    robot_size: int = 3,
    margin: float = 0.5,
) -> tuple[float, float] | None:
    """
    Find the nearest point where the robot center can be to approach the fire.

    This finds cells on the boundary of the forbidden zone (just outside it)
    that are closest to the robot's current position.

    Args:
        fire_grid: Boolean grid where True = fire cell
        robot_x: Robot center x position (in cells)
        robot_y: Robot center y position (in cells, corresponds to row)
        robot_size: Robot footprint size (default 3 for 3x3)
        margin: Additional margin in fractional cells

    Returns:
        (x, y) of the nearest approach point, or None if no valid point exists
    """
    rows, cols = fire_grid.shape

    # Compute forbidden zone
    forbidden = compute_fire_forbidden_zone(fire_grid, robot_size, margin)

    # If no fire, return None
    if not fire_grid.any():
        return None

    # Find cells that are:
    # 1. Not forbidden (robot can be here)
    # 2. Adjacent to forbidden zone (as close as possible to fire)

    # Dilate forbidden zone by 1 to find the boundary
    boundary_zone = _dilate8(forbidden, 1) & ~forbidden

    # Also need to ensure we're within grid bounds with enough room for robot
    half = robot_size // 2
    valid_mask = np.zeros((rows, cols), dtype=bool)
    valid_mask[half:rows-half, half:cols-half] = True

    # Candidate cells: on boundary and valid for robot placement
    candidates = boundary_zone & valid_mask

    if not candidates.any():
        # No boundary found, try just finding closest non-forbidden cell
        candidates = ~forbidden & valid_mask
        if not candidates.any():
            return None

    # Find the candidate closest to robot position
    candidate_positions = np.argwhere(candidates)

    best_dist = float('inf')
    best_pos = None

    for row, col in candidate_positions:
        # Distance from robot to this cell center
        dist = math.sqrt((col - robot_x) ** 2 + (row - robot_y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_pos = (float(col), float(row))

    return best_pos
