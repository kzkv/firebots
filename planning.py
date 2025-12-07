# Planning utilities with potential field cost functions
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
    SQ2 = 2**0.5
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


def build_fire_potential_field(
    fire_distance: np.ndarray,
    ideal_distance: float = 4.0,
    min_distance: float = 2.0,
    inner_repulsion: float = 10.0,
    outer_repulsion: float = 2.0,
) -> np.ndarray:
    """
    Build a potential field cost for fire that creates a "valley" at ideal distance.

    This creates:
    - Infinite cost if closer than min_distance (hard barrier)
    - Repulsion (increasing cost) as you get closer to fire
    - Repulsion (increasing cost) as you get farther from fire
    - Minimum cost at ideal_distance

    Args:
        fire_distance: Distance field from fire
        ideal_distance: The optimal distance from fire (cost = 0 here)
        min_distance: Closer than this = impassable (inf cost)
        inner_repulsion: Strength of repulsion for being too close
        outer_repulsion: Strength of repulsion for being too far

    Returns:
        Cost grid with potential field around fire
    """
    cost = np.zeros_like(fire_distance, dtype=np.float32)

    # Hard barrier - too close is impassable
    too_close = fire_distance < min_distance
    cost[too_close] = np.inf

    # Potential field for passable cells
    passable = ~too_close & np.isfinite(fire_distance)

    # Deviation from ideal distance
    deviation = fire_distance[passable] - ideal_distance

    # Quadratic potential - different strengths for inner vs outer
    # Inner: quadratic (strong push away from fire)
    # Outer: LINEAR (gentle pull toward fire, doesn't overwhelm tree costs)
    inner_mask = fire_distance[passable] < ideal_distance
    outer_mask = ~inner_mask

    # Create temporary array for passable costs
    passable_costs = np.zeros(passable.sum(), dtype=np.float32)
    passable_costs[inner_mask] = inner_repulsion * (deviation[inner_mask] ** 2)
    passable_costs[outer_mask] = (
        outer_repulsion * deviation[outer_mask]
    )  # LINEAR not squared

    cost[passable] = passable_costs

    return cost


def build_fire_corridor_cost(
    fire_grid: np.ndarray,
    fire_distance: np.ndarray,
    min_distance: float = 2.0,
    corridor_width: float = 2.0,
    falloff_rate: float = 1.0,
) -> np.ndarray:
    """
    Build cost field for the corridor around fire. (Legacy function)

    Creates a flat optimal band around fire (cost = 0) that forms a
    contiguous ring the robot can follow to encircle the fire.
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


def build_tree_potential_field(
    tree_grid: np.ndarray,
    min_distance: float = 1.5,
    repulsion_strength: float = 20.0,
    repulsion_decay: float = 2.0,
    max_radius: int = 5,
) -> np.ndarray:
    """
    Build a potential field for tree avoidance.

    Creates repulsion that pushes the robot away from trees.

    Args:
        tree_grid: Boolean grid where True = tree cell
        min_distance: Cells within this distance = IMPASSABLE (inf)
        repulsion_strength: Peak repulsion magnitude beyond min_distance
        repulsion_decay: Decay rate (larger = wider influence)
        max_radius: Maximum radius of influence (cells)

    Returns:
        Cost grid with tree repulsion field
    """
    rows, cols = tree_grid.shape
    cost = np.zeros((rows, cols), dtype=np.float32)

    # Distance from tree trunks
    dist_tree = _chamfer_distance8(tree_grid.astype(bool))

    # Trees and cells within min_distance are impassable
    cost[dist_tree < min_distance] = np.inf

    # Exponential repulsion falloff beyond min_distance
    falloff_start = max(min_distance, 0.1)
    mask = (dist_tree >= falloff_start) & (dist_tree <= max_radius) & np.isfinite(cost)

    # Repulsion: strength * exp(-(d - falloff_start) / decay)
    repulsion = repulsion_strength * np.exp(
        -(dist_tree - falloff_start) / max(1e-6, repulsion_decay)
    )
    cost[mask] += repulsion[mask].astype(cost.dtype)

    return cost


def build_tree_cost(
    tree_grid: np.ndarray,
    min_distance: float = 0.0,
    trunk_scale: float = 10.0,
    trunk_tau: float = 0.5,
    trunk_max_radius: int = 3,
) -> np.ndarray:
    """
    Build cost field for tree avoidance. (Legacy function - use build_tree_potential_field)
    """
    return build_tree_potential_field(
        tree_grid,
        min_distance=min_distance,
        repulsion_strength=trunk_scale,
        repulsion_decay=trunk_tau,
        max_radius=trunk_max_radius,
    )


def rebuild_weight_grid(
    fire_grid: np.ndarray,
    fire_distance: np.ndarray,
    known_trees: np.ndarray,
    base_cost: float = 1.0,
    # Fire potential field params
    fire_min_distance: float = 2.0,
    fire_ideal_distance: float = 4.0,
    fire_inner_repulsion: float = 10.0,
    fire_outer_repulsion: float = 2.0,
    # Legacy fire corridor params (for backwards compatibility)
    fire_corridor_width: float = None,
    fire_falloff_rate: float = None,
    # Tree potential field params
    tree_min_distance: float = 1.5,
    tree_repulsion_strength: float = 20.0,
    tree_repulsion_decay: float = 2.0,
    tree_max_radius: int = 5,
    # Legacy tree params (for backwards compatibility)
    tree_trunk_scale: float = None,
    tree_trunk_tau: float = None,
    tree_trunk_max_radius: int = None,
    # Use potential field or legacy
    use_potential_field: bool = True,
) -> np.ndarray:
    """
    Build complete weight grid combining fire and tree potential fields.

    The potential field approach creates smooth gradients that guide the robot
    to the optimal distance from fire while avoiding trees.
    """
    rows, cols = fire_grid.shape

    # Start with base cost
    weight = np.full((rows, cols), base_cost, dtype=np.float32)

    if use_potential_field:
        # New potential field approach
        fire_cost = build_fire_potential_field(
            fire_distance,
            ideal_distance=fire_ideal_distance,
            min_distance=fire_min_distance,
            inner_repulsion=fire_inner_repulsion,
            outer_repulsion=fire_outer_repulsion,
        )

        tree_cost = build_tree_potential_field(
            known_trees,
            min_distance=tree_min_distance,
            repulsion_strength=tree_repulsion_strength,
            repulsion_decay=tree_repulsion_decay,
            max_radius=tree_max_radius,
        )

        # Debug: show cost ranges
        fire_finite = fire_cost[np.isfinite(fire_cost)]
        tree_finite = tree_cost[np.isfinite(tree_cost)]
        if len(fire_finite) > 0 and len(tree_finite) > 0:
            print(
                f"  Fire cost range: {fire_finite.min():.1f} - {fire_finite.max():.1f}"
            )
            print(
                f"  Tree cost range: {tree_finite.min():.1f} - {tree_finite.max():.1f} (trees found: {known_trees.sum()})"
            )
    else:
        # Legacy corridor approach
        corridor_width = fire_corridor_width if fire_corridor_width is not None else 2.0
        falloff = fire_falloff_rate if fire_falloff_rate is not None else 1.0

        fire_cost = build_fire_corridor_cost(
            fire_grid,
            fire_distance,
            min_distance=fire_min_distance,
            corridor_width=corridor_width,
            falloff_rate=falloff,
        )

        t_scale = (
            tree_trunk_scale
            if tree_trunk_scale is not None
            else tree_repulsion_strength
        )
        t_tau = tree_trunk_tau if tree_trunk_tau is not None else tree_repulsion_decay
        t_radius = (
            tree_trunk_max_radius
            if tree_trunk_max_radius is not None
            else tree_max_radius
        )

        tree_cost = build_tree_cost(
            known_trees,
            min_distance=tree_min_distance,
            trunk_scale=t_scale,
            trunk_tau=t_tau,
            trunk_max_radius=t_radius,
        )

    # Combine: if either is inf, result is inf
    inf_mask = ~np.isfinite(weight) | ~np.isfinite(fire_cost) | ~np.isfinite(tree_cost)
    weight += np.where(np.isfinite(fire_cost), fire_cost, 0)
    weight += np.where(np.isfinite(tree_cost), tree_cost, 0)
    weight[inf_mask] = np.inf

    return weight


# === Fire Approach Point Finding ===


def _distance_point_to_cell_edge(
    px: float, py: float, cell_row: int, cell_col: int
) -> float:
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
                dist = _distance_point_to_cell_edge(
                    sub_center_x, sub_center_y, fire_row, fire_col
                )
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

    forbidden, res = compute_fire_forbidden_zone(
        fire_grid, robot_size, margin, resolution
    )
    sub_rows, sub_cols = rows * res, cols * res

    boundary_zone = _dilate8(forbidden, 1) & ~forbidden

    half_sub = (robot_size // 2) * res
    valid_mask = np.zeros((sub_rows, sub_cols), dtype=bool)
    valid_mask[half_sub : sub_rows - half_sub, half_sub : sub_cols - half_sub] = True

    candidates = boundary_zone & valid_mask

    if not candidates.any():
        candidates = ~forbidden & valid_mask
        if not candidates.any():
            return None

    candidate_positions = np.argwhere(candidates)

    robot_sub_x = robot_x * res
    robot_sub_y = robot_y * res

    best_dist = float("inf")
    best_pos = None

    for sub_row, sub_col in candidate_positions:
        sub_center_x = sub_col + 0.5
        sub_center_y = sub_row + 0.5
        dist = math.sqrt(
            (sub_center_x - robot_sub_x) ** 2 + (sub_center_y - robot_sub_y) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_pos = (sub_center_x / res, sub_center_y / res)

    return best_pos
