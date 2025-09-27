# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro

import pygame
import numpy as np
from fire_bitmap import load_fire_bitmap
from obstacles import place_trees
from render import World

# 6x3' field, 72x36", scale is 1" -> 1'
COLS = 72
ROWS = 36
CELL_SIZE = 30
hud_height = CELL_SIZE

TREE_COUNT = 50

rng = np.random.default_rng()
pygame.init()
world = World(ROWS, COLS, CELL_SIZE)

running = True

# Ingest fire bitmap
fire_surface, fire_grid = load_fire_bitmap("fire1.png", world.cols, world.rows)

# Generate obstacles
tree_grid = place_trees(world.cols, world.rows, count=TREE_COUNT, rng=rng)

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)
    world.render_trees(tree_grid)
    world.render_tree_sprites(tree_grid, rng)

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()
