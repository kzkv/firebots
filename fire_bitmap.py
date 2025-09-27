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


def load_fire_bitmap(path, cols, rows):
    fire_surface = pg.image.load(path).convert_alpha()
    surface_width, surface_height = fire_surface.get_width(), fire_surface.get_height()
    cell_width, cell_height = surface_width // cols, surface_height // rows
    pixel_array = pg.surfarray.array3d(fire_surface)
    fire_grid = np.zeros((rows, cols), dtype=bool)

    # majority rule per cell: count pixels
    for row in range(rows):
        y_start, y_end = row * cell_height, (row + 1) * cell_height
        for col in range(cols):
            x_start, x_end = col * cell_width, (col + 1) * cell_width
            block = pixel_array[x_start:x_end, y_start:y_end]
            non_black_pixels = np.count_nonzero(block.any(axis=2))
            fire_grid[row, col] = non_black_pixels > (
                block.shape[0] * block.shape[1] // 2
            )
    return fire_surface, fire_grid
