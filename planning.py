# ChatGPT to draft cells weights

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
