# Planning utilities with tunable cost functions
# RBE 550, Firebots (course project)

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
                if c:
                    v = min(v, d[r - 1, c - 1] + SQ2)
                if c < W - 1:
                    v = min(v, d[r - 1, c + 1] + SQ2)
            if c:
                v = min(v, d[r, c - 1] + 1)
            d[r, c] = v

    # backward pass
    for r in range(H - 1, -1, -1):
        for c in range(W - 1, -1, -1):
            v = d[r, c]
            if r < H - 1:
                v = min(v, d[r + 1, c] + 1)
                if c:
                    v = min(v, d[r + 1, c - 1] + SQ2)
                if c < W - 1:
                    v = min(v, d[r + 1, c + 1] + SQ2)
            if c < W - 1:
                v = min(v, d[r, c + 1] + 1)
            d[r, c] = v
    return d


def compute_fire_distance_field(fire_grid: np.ndarray) -> np.ndarray:
    """
    Compute the distance from each cell to the nearest fire cell.
    Uses chamfer distance for a good approximation of Euclidean distance.
    """
    return _chamfer_distance8(fire_grid)


def build_fire_corridor_cost(
        fire_grid: np.ndarray,
        fire_distance: np.ndarray,
        min_distance: float = 2.0,
        corridor_width: float = 2.0,
        falloff_rate: float = 1.0,
) -> np.ndarray:
    """
    Build cost field for the corridor around fire.

    Creates a flat optimal band around fire (cost = 0) that forms a
    contiguous ring the robot can follow to encircle the fire.

    Args:
        fire_grid: Boolean grid where True = fire cell
        fire_distance: Distance field from fire (from compute_fire_distance_field)
        min_distance: Cells closer than this are impassable (inf cost)
        corridor_width: Width of optimal corridor (cost = 0)
                       Corridor spans from min_distance to min_distance + corridor_width
        falloff_rate: How fast cost increases beyond corridor

    Returns:
        Cost grid with fire corridor penalties
    """
    cost = np.zeros_like(fire_distance, dtype=np.float32)

    corridor_outer = min_distance + corridor_width

    # Too close = impassable
    cost[fire_distance < min_distance] = np.inf

    # In corridor = cost 0 (optimal ring for encircling)
    # Already 0 from np.zeros

    # Beyond corridor = increasing cost (pushes robot toward corridor)
    beyond = fire_distance > corridor_outer
    cost[beyond] = (fire_distance[beyond] - corridor_outer) * falloff_rate

    return cost


def build_tree_cost(
        tree_grid: np.ndarray,
        min_distance: float = 0.0,
        trunk_scale: float = 10.0,
        trunk_tau: float = 0.5,
        trunk_max_radius: int = 3,
) -> np.ndarray:
    """
    Build cost field for tree avoidance.

    Args:
        tree_grid: Boolean grid where True = tree cell
        min_distance: Cells within this distance of trees = IMPASSABLE (inf)
        trunk_scale: Peak penalty magnitude beyond min_distance
        trunk_tau: Decay rate (smaller = faster falloff, larger = wider berth)
        trunk_max_radius: Maximum radius of influence around trees (cells)

    Returns:
        Cost grid with tree penalties (inf at tree cells and within min_distance)
    """
    rows, cols = tree_grid.shape
    cost = np.zeros((rows, cols), dtype=np.float32)

    # Distance from tree trunks (0 at trunks)
    dist_tree = _chamfer_distance8(tree_grid.astype(bool))

    # Trees and cells within min_distance are impassable
    cost[dist_tree < min_distance] = np.inf

    # Apply exponential falloff for cells beyond min_distance, within radius
    falloff_start = max(min_distance, 1.0)  # Start falloff at least 1 cell away
    mask = (dist_tree >= falloff_start) & (dist_tree <= trunk_max_radius) & np.isfinite(cost)

    # Exponential falloff: penalty = scale * exp(-(d - falloff_start) / tau)
    penalty = trunk_scale * np.exp(-(dist_tree - falloff_start) / max(1e-6, trunk_tau))
    cost[mask] += penalty[mask].astype(cost.dtype)

    return cost


def rebuild_weight_grid(
        fire_grid: np.ndarray,
        fire_distance: np.ndarray,
        known_trees: np.ndarray,
        # Base cost
        base_cost: float = 1.0,
        # Fire corridor params
        fire_min_distance: float = 2.0,
        fire_corridor_width: float = 2.0,
        fire_falloff_rate: float = 2.0,
        # Tree avoidance params
        tree_min_distance: float = 0.0,
        tree_trunk_scale: float = 10.0,
        tree_trunk_tau: float = 0.5,
        tree_trunk_max_radius: int = 3,
) -> np.ndarray:
    """
    Build complete weight grid combining fire corridor and tree costs.

    Args:
        fire_grid: Boolean grid where True = fire cell
        fire_distance: Distance field from fire
        known_trees: Boolean grid of known tree positions

        base_cost: Baseline traversal cost for all cells

        fire_min_distance: Minimum safe distance from fire (closer = inf)
        fire_corridor_width: Width of optimal corridor (cost = 0)
        fire_falloff_rate: Cost increase rate beyond corridor

        tree_min_distance: Minimum safe distance from trees (closer = inf)
        tree_trunk_scale: Peak avoidance penalty beyond min distance
        tree_trunk_tau: Tree penalty decay rate (larger = wider berth)
        tree_trunk_max_radius: Maximum tree influence radius

    Returns:
        Combined weight grid for path planning
    """
    rows, cols = fire_grid.shape

    # Start with base cost
    weight = np.full((rows, cols), base_cost, dtype=np.float32)

    # Add fire corridor cost
    corridor_cost = build_fire_corridor_cost(
        fire_grid,
        fire_distance,
        min_distance=fire_min_distance,
        corridor_width=fire_corridor_width,
        falloff_rate=fire_falloff_rate,
    )
    weight += corridor_cost

    # Add tree cost
    tree_cost = build_tree_cost(
        known_trees,
        min_distance=tree_min_distance,
        trunk_scale=tree_trunk_scale,
        trunk_tau=tree_trunk_tau,
        trunk_max_radius=tree_trunk_max_radius,
    )

    # Combine: if either is inf, result is inf
    inf_mask = ~np.isfinite(weight) | ~np.isfinite(tree_cost)
    weight += np.where(np.isfinite(tree_cost), tree_cost, 0)
    weight[inf_mask] = np.inf

    return weight


# === Fire Approach Point Finding ===

def _distance_point_to_cell_edge(px: float, py: float, cell_row: int, cell_col: int) -> float:
    """
    Compute the distance from a point to the nearest edge of a cell.
    """
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
    """
    Compute the forbidden zone for the robot center using Minkowski sum inflation.
    """
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
    """
    Find the nearest point where the robot center can be to approach the fire.
    """
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