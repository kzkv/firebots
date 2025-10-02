import numpy as np

def _dilate8(mask: np.ndarray, radius: int) -> np.ndarray:
    """8-neighborhood dilation by `radius` cells (pure NumPy, no SciPy)."""
    out = mask.copy()
    for _ in range(max(0, radius)):
        d = out
        up        = np.pad(d[:-1, :], ((1,0),(0,0)))
        down      = np.pad(d[1:,  :], ((0,1),(0,0)))
        left      = np.pad(d[:, :-1], ((0,0),(1,0)))
        right     = np.pad(d[:, 1: ], ((0,0),(0,1)))
        upleft    = np.pad(d[:-1, :-1], ((1,0),(1,0)))
        upright   = np.pad(d[:-1, 1: ], ((1,0),(0,1)))
        downleft  = np.pad(d[1:,  :-1], ((0,1),(1,0)))
        downright = np.pad(d[1:,   1: ], ((0,1),(0,1)))
        out = d | up | down | left | right | upleft | upright | downleft | downright
    return out

def fireline_cells(fire_grid: np.ndarray, gap: int, obstacles: np.ndarray | None = None) -> np.ndarray:
    """
    Return a boolean grid for a 1-cell thick band hugging the fire with `gap` empty cells.
    gap=0 => immediately adjacent; gap=2 => two empty cells between fire and band.
    """
    inner = _dilate8(fire_grid, gap)
    outer = _dilate8(fire_grid, gap + 1)
    ring = outer & ~inner                     # cells at exact Chebyshev distance gap+1
    if obstacles is not None:
        ring &= ~obstacles                    # avoid trees if desired
    return ring

