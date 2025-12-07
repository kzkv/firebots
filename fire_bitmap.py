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

import numpy as np
import pygame as pg


def load_fire_bitmap(path, cols, rows, fire_cols=None, fire_rows=None):
    """
    Load fire bitmap and convert to grid.

    Args:
        path: Path to bitmap file
        cols: Grid width in cells (full field)
        rows: Grid height in cells (full field)
        fire_cols: Width of fire region in cells (default: cols)
        fire_rows: Height of fire region in cells (default: rows)

    Returns:
        (fire_surface, fire_grid) - surface scaled to fire region, grid centered on field
    """
    raw_surface = pg.image.load(path).convert_alpha()
    surface_width, surface_height = raw_surface.get_width(), raw_surface.get_height()

    # Default to full field if not specified
    if fire_cols is None:
        fire_cols = cols
    if fire_rows is None:
        fire_rows = rows

    # Offset to center the fire region on the field
    col_offset = (cols - fire_cols) // 2
    row_offset = (rows - fire_rows) // 2

    cell_width = surface_width // fire_cols
    cell_height = surface_height // fire_rows
    pixel_array = pg.surfarray.array3d(raw_surface)

    # Start with empty grid
    fire_grid = np.zeros((rows, cols), dtype=bool)

    # Fill only the fire region (centered)
    for row in range(fire_rows):
        y_start, y_end = row * cell_height, (row + 1) * cell_height
        for col in range(fire_cols):
            x_start, x_end = col * cell_width, (col + 1) * cell_width
            block = pixel_array[x_start:x_end, y_start:y_end]
            non_black_pixels = np.count_nonzero(block.any(axis=2))
            if non_black_pixels > (block.shape[0] * block.shape[1] // 2):
                fire_grid[row + row_offset, col + col_offset] = True

    # Scale surface to match fire region size (for rendering)
    # This will be blitted at the correct offset in render
    return raw_surface, fire_grid, (col_offset, row_offset, fire_cols, fire_rows)
