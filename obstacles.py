import numpy as np

TREE_SHAPES = [
    ((0, 0),) * 5,
    ((0, 0), (1, 0)) * 2,
    ((0, 0), (0, 1)) * 2,
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
]


def place_trees(cols, rows, count, rng=None):
    tree_grid = np.zeros((rows, cols), dtype=bool)

    for _ in range(count):
        tree = TREE_SHAPES[rng.integers(len(TREE_SHAPES))]
        max_dx = max(dx for dx, _ in tree)
        max_dy = max(dy for _, dy in tree)

        anchor_col = int(rng.integers(0, cols - max_dx))
        anchor_row = int(rng.integers(0, rows - max_dy))

        for dx, dy in tree:
            r, c = anchor_row + dy, anchor_col + dx
            tree_grid[r, c] = True  # overlap is fine (remains True)

    return tree_grid


def clear_robot_start(tree_grid, robot_x, robot_y, robot_size=3, margin=1):
    """
    Clear trees from robot starting position.

    Args:
        tree_grid: Boolean grid of trees (modified in place)
        robot_x, robot_y: Robot center position
        robot_size: Robot footprint size (default 3 for 3x3)
        margin: Extra clearance around robot (default 1 cell)
    """
    offset = robot_size // 2 + margin
    cx, cy = int(round(robot_x)), int(round(robot_y))

    rows, cols = tree_grid.shape
    r_min = max(0, cy - offset)
    r_max = min(rows, cy + offset + 1)
    c_min = max(0, cx - offset)
    c_max = min(cols, cx + offset + 1)

    tree_grid[r_min:r_max, c_min:c_max] = False