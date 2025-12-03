# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro, Claude

import pygame
import numpy as np
from fire_bitmap import load_fire_bitmap
from obstacles import place_trees
from planning import build_weight_grid, find_nearest_fire_approach_point
from render import World
from firebot import Firebot

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
# fire_surface, fire_grid = load_fire_bitmap("fire1.png", world.cols, world.rows)
fire_surface, fire_grid = load_fire_bitmap("fire2.png", world.cols, world.rows)

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

# Create firebot at initial position (center-left of field)
# Robot is 3x3 cells, so place center at (5, ROWS//2) to be safely inside
firebot = Firebot(x=5.0, y=ROWS / 2.0, theta=0.0)

# Preload firebot sprite
world.load_firebot_sprite()

# Target position for click-to-drive
target_pos = None

while running:
    dt = world.clock.tick(60) / 1000.0  # Delta time in seconds

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            # Left click - check if it's on the field
            cell_pos = world.screen_to_cell(e.pos[0], e.pos[1])
            if cell_pos is not None:
                # Check if clicking on HUD toggle
                if hasattr(world, "toggle_rect") and world.toggle_rect.collidepoint(
                    e.pos
                ):
                    world.show_weights = not world.show_weights
                else:
                    # Set new target for firebot
                    target_pos = cell_pos
                    firebot.set_target(cell_pos[0], cell_pos[1])
            else:
                # Check HUD click
                if hasattr(world, "toggle_rect") and world.toggle_rect.collidepoint(
                    e.pos
                ):
                    world.show_weights = not world.show_weights
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_w:
                world.show_weights = not world.show_weights
            elif e.key == pygame.K_ESCAPE:
                # Cancel current movement
                firebot.target = None
                firebot.state = "idle"
                firebot.v = 0.0
                firebot.omega = 0.0
                target_pos = None
            elif e.key == pygame.K_f:
                # Find and drive to nearest fire approach point
                approach_point = find_nearest_fire_approach_point(
                    fire_grid,
                    firebot.x,
                    firebot.y,
                    robot_size=firebot.size,
                    margin=firebot.fire_approach_margin,
                )
                if approach_point is not None:
                    target_pos = approach_point
                    firebot.set_target(approach_point[0], approach_point[1])

    # Update firebot motion controller
    firebot.control_step(dt)

    # Clear target marker when arrived
    if not firebot.is_moving():
        target_pos = None

    # Render
    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)
    world.render_trees(tree_grid, firebot)
    world.render_tree_sprites(tree_grid, rng, firebot)
    world.render_fog_of_war(firebot)

    if world.show_weights:
        world.render_weight_heatmap(weight_grid)
        world.render_weight_on_hover(weight_grid, decimals=1)

    # Render target marker if we have one
    if target_pos is not None:
        world.render_target_marker(target_pos[0], target_pos[1])

    # Render firebot
    world.render_firebot(firebot)

    world.draw_hud()

    pygame.display.flip()

pygame.quit()
