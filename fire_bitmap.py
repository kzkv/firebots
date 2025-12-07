"""
Bitmap workflow:
1. https://www.pixilart.com/draw
2. Canvas size: 720x360
3. Tool: spray, size 50, speed 50, color (255, 0, 0)
4. Export as PNG

Ingesting fire bitmap into the game.

Assume that the source bitmap conforms to the proportions of the cell grid.
Consider a cell "on fire" if more pixels in it are lit than not.
Pixel-level opacity, if present in the bitmap, should be ignored.
"""

import pygame as pg
import numpy as np


def load_fire_bitmap(path, cols, rows, scale=1.0):
    """
    Load fire bitmap and convert to grid.

    Args:
        path: Path to bitmap file
        cols: Grid width in cells
        rows: Grid height in cells
        scale: Scale factor for fire (0.0-1.0).
               1.0 = fire fills grid, 0.5 = fire is half size, centered

    Returns:
        (fire_surface, fire_grid)
    """
    fire_surface = pg.image.load(path).convert_alpha()
    surface_width, surface_height = fire_surface.get_width(), fire_surface.get_height()

    # Scale down the effective grid area for fire
    scaled_cols = max(1, int(cols * scale))
    scaled_rows = max(1, int(rows * scale))

    # Offset to center the scaled fire
    col_offset = (cols - scaled_cols) // 2
    row_offset = (rows - scaled_rows) // 2

    cell_width = surface_width // scaled_cols
    cell_height = surface_height // scaled_rows
    pixel_array = pg.surfarray.array3d(fire_surface)

    # Start with empty grid
    fire_grid = np.zeros((rows, cols), dtype=bool)

    # Fill only the scaled/centered region
    for row in range(scaled_rows):
        y_start, y_end = row * cell_height, (row + 1) * cell_height
        for col in range(scaled_cols):
            x_start, x_end = col * cell_width, (col + 1) * cell_width
            block = pixel_array[x_start:x_end, y_start:y_end]
            non_black_pixels = np.count_nonzero(block.any(axis=2))
            if non_black_pixels > (block.shape[0] * block.shape[1] // 2):
                fire_grid[row + row_offset, col + col_offset] = True

    return fire_surface, fire_grid