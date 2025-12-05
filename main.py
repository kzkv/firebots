# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro, Claude

import pygame
import numpy as np

from d_star_lite import DStarLite
from fire_bitmap import load_fire_bitmap
from obstacles import place_trees
from planning import (
    find_nearest_fire_approach_point,
    compute_fire_forbidden_zone,
    compute_fire_distance_field,
    rebuild_weight_grid,
)
from render import World
from firebot import Firebot
from exploration import ExplorationMap


# Field dimensions: 1 cell = 3 ft
COLS = 100
ROWS = 60
CELL_SIZE = 18
TREE_COUNT = 50

# Initialize pygame and world
rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots")
world = World(ROWS, COLS, CELL_SIZE)

# Load fire bitmap
fire_surface, fire_grid = load_fire_bitmap("fire2.png", world.cols, world.rows)

# Generate obstacles
tree_grid = place_trees(world.cols, world.rows, count=TREE_COUNT, rng=rng)

# Create exploration map
exploration = ExplorationMap(world.rows, world.cols)

# Compute fire distance field
fire_distance = compute_fire_distance_field(fire_grid)

# Create firebot at initial position
firebot = Firebot(x=5.0, y=ROWS / 2.0, theta=0.0)

# Initial exploration from starting position
exploration.update(firebot.x, firebot.y, firebot.sensor_radius, tree_grid)

# Build weight grid with known obstacles
weight_grid = rebuild_weight_grid(fire_grid, fire_distance, exploration.get_known_obstacles())

# Debug output
finite = weight_grid[np.isfinite(weight_grid)]
print(f"Weight range: min={finite.min():.1f}, max={finite.max():.1f}")

# Create D* Lite planner
planner = DStarLite(ROWS, COLS)
planned_path = []
path_index = 0

# Compute forbidden zone for visualization
FORBIDDEN_ZONE_RESOLUTION = 4
forbidden_zone, forbidden_zone_res = compute_fire_forbidden_zone(
    fire_grid,
    robot_size=firebot.size,
    margin=firebot.fire_approach_margin,
    resolution=FORBIDDEN_ZONE_RESOLUTION,
)

# Preload sprites
world.load_firebot_sprite()

# UI state
target_pos = None
path_following_enabled = True  # Toggle for auto-following path
show_path = True  # Toggle for rendering path

# Main loop
running = True
while running:
    dt = world.clock.tick(60) / 1000.0

    # Event handling
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            # Check HUD toggles first
            if hasattr(world, "toggle_rect") and world.toggle_rect.collidepoint(e.pos):
                world.show_weights = not world.show_weights
            elif hasattr(world, "arrows_toggle_rect") and world.arrows_toggle_rect.collidepoint(e.pos):
                world.show_arrows = not world.show_arrows
            elif hasattr(world, "forbidden_zone_toggle_rect") and world.forbidden_zone_toggle_rect.collidepoint(e.pos):
                world.show_forbidden_zone = not world.show_forbidden_zone
            else:
                # Left click on field - plan path to target
                cell_pos = world.screen_to_cell(e.pos[0], e.pos[1])
                if cell_pos is not None:
                    target_pos = cell_pos
                    start = (int(firebot.y), int(firebot.x))
                    goal = (int(cell_pos[1]), int(cell_pos[0]))

                    print(f"Planning from {start} to {goal}")
                    planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
                    planner.compute_shortest_path()
                    planned_path = planner.extract_path()
                    path_index = 0
                    print(f"Path has {len(planned_path)} cells")

        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_w:
                world.show_weights = not world.show_weights
            elif e.key == pygame.K_a:
                world.show_arrows = not world.show_arrows
            elif e.key == pygame.K_p:
                show_path = not show_path
                print(f"Show path: {show_path}")
            elif e.key == pygame.K_m:
                path_following_enabled = not path_following_enabled
                print(f"Path following: {path_following_enabled}")
            elif e.key == pygame.K_ESCAPE:
                # Cancel current movement and path
                firebot.target = None
                firebot.state = "idle"
                firebot.v = 0.0
                firebot.omega = 0.0
                planned_path = []
                path_index = 0
                target_pos = None
                print("Movement cancelled")
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
                    start = (int(firebot.y), int(firebot.x))
                    goal = (int(approach_point[1]), int(approach_point[0]))
                    planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
                    planner.compute_shortest_path()
                    planned_path = planner.extract_path()
                    path_index = 0
                    print(f"Path to fire: {len(planned_path)} cells")

    # Path following (before control_step)
    if path_following_enabled and len(planned_path) > 0:
        # Update which waypoint we're targeting
        path_index = planner.get_next_waypoint(planned_path, path_index, firebot.x, firebot.y, lookahead=4.0)

        if path_index < len(planned_path):
            waypoint = planned_path[path_index]
            wx, wy = waypoint[1], waypoint[0]  # col, row -> x, y

            # Only update target if it's different enough
            if firebot.target is None:
                firebot.set_target(wx, wy)
            else:
                dx = wx - firebot.target[0]
                dy = wy - firebot.target[1]
                if dx * dx + dy * dy > 1.0:  # Target moved more than 1 cell
                    firebot.set_target(wx, wy)
        else:
            # Reached end of path
            print("Path complete!")
            planned_path = []
            path_index = 0
            target_pos = None

    # Update robot controller
    firebot.control_step(dt)

    # Clear target when stopped (only if not following a path)
    if not firebot.is_moving() and len(planned_path) == 0:
        target_pos = None

    # Update exploration
    new_obstacles_found = exploration.update(
        firebot.x, firebot.y, firebot.sensor_radius, tree_grid
    )

    # Replan if obstacles discovered
    if new_obstacles_found:
        weight_grid = rebuild_weight_grid(fire_grid, fire_distance, exploration.get_known_obstacles())

        if len(planned_path) > 0 and target_pos is not None:
            start = (int(firebot.y), int(firebot.x))
            goal = (int(target_pos[1]), int(target_pos[0]))
            planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
            planner.compute_shortest_path()
            planned_path = planner.extract_path()
            path_index = 0
            print(f"Replanned: {len(planned_path)} cells")

    # === Rendering ===
    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)
    world.render_corridor(weight_grid)
    world.render_trees(exploration.get_known_obstacles())
    world.render_tree_sprites(exploration.get_known_obstacles(), rng)

    # Render fireline
    if firebot.cutting_fireline and len(firebot.fireline_path) > 0:
        fireline = list(firebot.fireline_path)
        if firebot.state in ("driving", "rotating"):
            front = firebot.front_position
            fireline.append((front[0], front[1], firebot.theta))
        world.render_fireline_path(fireline)

    # Render planned path
    if show_path and len(planned_path) > 0:
        world.render_path(planned_path)

    world.render_fog_of_war(firebot)

    if world.show_weights:
        world.render_weight_heatmap(weight_grid)
        world.render_weight_on_hover(weight_grid, decimals=1)

    if world.show_arrows:
        world.render_gradient_arrows(weight_grid, spacing=3)

    if world.show_forbidden_zone:
        world.render_forbidden_zone(forbidden_zone, forbidden_zone_res)

    if target_pos is not None:
        world.render_target_marker(target_pos[0], target_pos[1])

    world.render_firebot(firebot)
    world.draw_hud()

    pygame.display.flip()

pygame.quit()