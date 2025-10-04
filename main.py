# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro

import pygame
import numpy as np
from fire_bitmap import load_fire_bitmap
from obstacles import place_trees
from planning import build_weight_grid
from render import World

# 6x3' field, 72x36", scale is 1" -> 1'
COLS = 72
ROWS = 36
CELL_SIZE = 30
hud_height = CELL_SIZE
FIRELINE_FIRE_GAP = 2  # A marging between the fireline and the fire cells
FIRELINE_OBSTACLE_GAP = 1  # Hugging the obstacles

TREE_COUNT = 50

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots")
world = World(ROWS, COLS, CELL_SIZE)

running = True

# Ingest fire bitmap
fire_surface, fire_grid = load_fire_bitmap("fire1.png", world.cols, world.rows)
# fire_surface, fire_grid = load_fire_bitmap("fire2.png", world.cols, world.rows)

# Generate obstacles
tree_grid = place_trees(world.cols, world.rows, count=TREE_COUNT, rng=rng)

# Establish cell weights
weight_grid = build_weight_grid(
    fire_grid,
    tree_grid,
    base_cost=1.0,
    fire_1=50.0,
    fire_2=20.0,
    trunk_scale=10.0,
    trunk_tau=0.5,  # quicker fade
    trunk_max_radius=3,
)

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        else:
            world.handle_event(e)

    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)
    world.render_trees(tree_grid)
    world.render_tree_sprites(tree_grid, rng)

    if world.show_weights:
        world.render_weight_heatmap(weight_grid)
        world.render_weight_on_hover(weight_grid, decimals=1)
    world.draw_hud()

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()
