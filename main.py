# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# FIXED VERSION v2 - with dt capping and better replan handling

import pygame
import numpy as np
import math

from d_star_lite import DStarLite
from fire_bitmap import load_fire_bitmap
from obstacles import place_trees, clear_robot_start
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

# Robot Constants
ROBOT_X = 5
ROBOT_Y = ROWS / 2

# Initialize pygame and world
rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots")
world = World(ROWS, COLS, CELL_SIZE)

# Load fire bitmap
fire_surface, fire_grid = load_fire_bitmap("fire2.png", world.cols, world.rows)

# Generate obstacles
tree_grid = place_trees(world.cols, world.rows, count=TREE_COUNT, rng=rng)

# Clear robots start position
clear_robot_start(tree_grid, robot_x=ROBOT_X, robot_y=ROBOT_Y)

# Create exploration map
exploration = ExplorationMap(world.rows, world.cols)

# Compute fire distance field
fire_distance = compute_fire_distance_field(fire_grid)

# Create firebot at initial position
firebot = Firebot(x=ROBOT_X, y=ROBOT_Y, theta=0.0)

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
path_following_enabled = True
show_path = True


def get_smooth_waypoint(path, robot_x, robot_y, robot_theta, lookahead=3.0):
    """
    Get a waypoint along the path using pure pursuit style lookahead.
    Skips points that are behind the robot.

    Returns (x, y) in cell coordinates and index, or (None, -1) if path is empty.
    """
    if not path:
        return None, -1

    # First, find the first path point that's NOT behind the robot
    # A point is "behind" if the angle to it differs from robot heading by > 90°
    start_idx = 0
    for i, (row, col) in enumerate(path):
        dx = col - robot_x
        dy = row - robot_y
        dist_sq = dx * dx + dy * dy

        # Skip points we're already very close to
        if dist_sq < 0.5:
            start_idx = i + 1
            continue

        # Check if point is in front of robot (within 90° of heading)
        angle_to_point = math.atan2(dy, dx)
        angle_diff = abs(math.atan2(math.sin(angle_to_point - robot_theta),
                                    math.cos(angle_to_point - robot_theta)))

        if angle_diff < math.pi / 2:  # Point is in front
            start_idx = i
            break
        else:
            start_idx = i + 1  # Skip this point, it's behind us

    # If all points are behind us, just use the last one
    if start_idx >= len(path):
        start_idx = len(path) - 1

    # Now find closest point from start_idx onward
    min_dist = float('inf')
    closest_idx = start_idx

    for i in range(start_idx, len(path)):
        row, col = path[i]
        dx = col - robot_x
        dy = row - robot_y
        dist = dx * dx + dy * dy
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Look ahead from the closest point
    target_idx = closest_idx
    accumulated_dist = 0.0

    for i in range(closest_idx, len(path) - 1):
        row1, col1 = path[i]
        row2, col2 = path[i + 1]
        segment_dist = ((col2 - col1) ** 2 + (row2 - row1) ** 2) ** 0.5
        accumulated_dist += segment_dist

        if accumulated_dist >= lookahead:
            target_idx = i + 1
            break
        target_idx = i + 1

    # Clamp to path bounds
    target_idx = min(target_idx, len(path) - 1)

    row, col = path[target_idx]
    return (col, row), target_idx  # Return (x, y) and index


# Track the last commanded waypoint
last_waypoint = None

# Main loop
running = True
while running:
    # FIX: Cap dt to prevent huge jumps during expensive computations (like D* Lite)
    raw_dt = world.clock.tick(60) / 1000.0
    dt = min(raw_dt, 0.05)  # Cap at 50ms (20 fps minimum)

    # Event handling
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if hasattr(world, "toggle_rect") and world.toggle_rect.collidepoint(e.pos):
                world.show_weights = not world.show_weights
            elif hasattr(world, "arrows_toggle_rect") and world.arrows_toggle_rect.collidepoint(e.pos):
                world.show_arrows = not world.show_arrows
            elif hasattr(world, "forbidden_zone_toggle_rect") and world.forbidden_zone_toggle_rect.collidepoint(e.pos):
                world.show_forbidden_zone = not world.show_forbidden_zone
            else:
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
                    last_waypoint = None
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
                firebot.stop()
                planned_path = []
                path_index = 0
                target_pos = None
                last_waypoint = None
                print("Movement cancelled")
            elif e.key == pygame.K_f:
                approach_point = find_nearest_fire_approach_point(
                    fire_grid, firebot.x, firebot.y,
                    robot_size=firebot.size, margin=firebot.fire_approach_margin,
                )
                if approach_point is not None:
                    target_pos = approach_point
                    start = (int(firebot.y), int(firebot.x))
                    goal = (int(approach_point[1]), int(approach_point[0]))
                    planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
                    planner.compute_shortest_path()
                    planned_path = planner.extract_path()
                    path_index = 0
                    last_waypoint = None
                    print(f"Path to fire: {len(planned_path)} cells")

    # Path following
    if path_following_enabled and len(planned_path) > 0:
        # Pass robot theta so we can skip points behind us
        waypoint, new_index = get_smooth_waypoint(
            planned_path, firebot.x, firebot.y, firebot.theta, lookahead=3.0
        )

        if waypoint is not None:
            wx, wy = waypoint

            if new_index >= len(planned_path) - 1:
                final_row, final_col = planned_path[-1]
                dx = final_col - firebot.x
                dy = final_row - firebot.y
                dist_to_goal = (dx * dx + dy * dy) ** 0.5

                if dist_to_goal < 1.0:
                    print("Path complete!")
                    planned_path = []
                    path_index = 0
                    target_pos = None
                    last_waypoint = None
                else:
                    if last_waypoint != (final_col, final_row):
                        firebot.set_target(final_col, final_row)
                        last_waypoint = (final_col, final_row)
            else:
                if last_waypoint is None:
                    firebot.set_target(wx, wy)
                    last_waypoint = (wx, wy)
                else:
                    dx = wx - last_waypoint[0]
                    dy = wy - last_waypoint[1]
                    if dx * dx + dy * dy > 4.0:
                        firebot.set_target(wx, wy)
                        last_waypoint = (wx, wy)

            path_index = new_index

    # Update robot controller
    firebot.control_step(dt)

    # Clear target when stopped
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
            new_path = planner.extract_path()

            if new_path:
                planned_path = new_path
                path_index = 0
                # DON'T reset last_waypoint unless the path changed dramatically
                # This prevents the robot from suddenly turning around
                # The next frame's get_smooth_waypoint will skip behind-points anyway
                print(f"Replanned: {len(planned_path)} cells")
            else:
                # No path found - stop the robot
                print("Replan failed - no path!")
                firebot.stop()
                planned_path = []
                path_index = 0

    # === Rendering ===
    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)
    world.render_corridor(weight_grid)
    world.render_trees(exploration.get_known_obstacles())
    world.render_tree_sprites(exploration.get_known_obstacles(), rng)

    if firebot.cutting_fireline and len(firebot.fireline_path) > 0:
        if firebot.is_moving():
            front = firebot.front_position
            fireline = firebot.fireline_path + [(front[0], front[1], firebot.theta)]
            world.render_fireline_path(fireline)
        else:
            world.render_fireline_path(firebot.fireline_path)

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