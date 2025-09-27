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
