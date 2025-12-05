# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# Field D* version with smooth pursuit controller

import pygame
import numpy as np
import math

from field_d_star import FieldDStar
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
CELL_SIZE = 20
TREE_COUNT = 50

# Robot Constants
ROBOT_X = 5
ROBOT_Y = ROWS / 2

# Initialize pygame and world
rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots - Field D*")
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

# Create Field D* planner
planner = FieldDStar(ROWS, COLS)
planned_path = []  # Will contain (x, y) tuples from Field D*
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

# Stuck detection
stuck_check_interval = 0.5  # Check every 0.5 seconds
stuck_timer = 0.0
last_stuck_check_pos = None
stuck_threshold = 0.3  # Must move at least this far

# Main loop
running = True
use_smooth_control = True  # Toggle between smooth pursuit and rotate-then-drive

while running:
    raw_dt = world.clock.tick(60) / 1000.0
    dt = min(raw_dt, 0.05)  # Cap dt

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
                    if planner.compute_shortest_path():
                        planned_path = planner.extract_path()  # Returns (x, y) tuples
                        path_index = 0
                        last_stuck_check_pos = (firebot.x, firebot.y)  # Reset stuck detection
                        print(f"Smooth path has {len(planned_path)} waypoints")
                    else:
                        planned_path = []
                        print("No path found!")

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
            elif e.key == pygame.K_c:
                use_smooth_control = not use_smooth_control
                print(f"Smooth control: {use_smooth_control}")
            elif e.key == pygame.K_ESCAPE:
                firebot.stop()
                planned_path = []
                path_index = 0
                target_pos = None
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
                    if planner.compute_shortest_path():
                        planned_path = planner.extract_path()
                        path_index = 0
                        print(f"Path to fire: {len(planned_path)} waypoints")

    # Path following
    if path_following_enabled and len(planned_path) > 0:
        if use_smooth_control:
            # Pure pursuit - smooth motion without stopping to turn
            still_going = firebot.pure_pursuit_step(planned_path, dt, lookahead=2.5)

            if not still_going:
                print("Path complete!")
                planned_path = []
                target_pos = None
        else:
            # Original rotate-then-drive behavior using set_target
            # Find a target point on the path
            target = None
            for i, (px, py) in enumerate(planned_path):
                dx = px - firebot.x
                dy = py - firebot.y
                if dx * dx + dy * dy > 4.0:  # 2 cells away
                    target = (px, py)
                    break

            if target is None and planned_path:
                target = planned_path[-1]

            if target:
                firebot.set_target(target[0], target[1])

            firebot.control_step(dt)

            # Check if done
            if planned_path:
                goal_x, goal_y = planned_path[-1]
                dx = goal_x - firebot.x
                dy = goal_y - firebot.y
                if dx * dx + dy * dy < 1.0:
                    print("Path complete!")
                    planned_path = []
                    target_pos = None
                    firebot.stop()
    else:
        # No path - just run control step (handles stopping)
        firebot.control_step(dt)

    # Update exploration
    new_obstacles_found = exploration.update(
        firebot.x, firebot.y, firebot.sensor_radius, tree_grid
    )

    # Stuck detection - check if robot is making progress
    stuck_timer += dt
    needs_replan = new_obstacles_found

    if stuck_timer >= stuck_check_interval and len(planned_path) > 0:
        stuck_timer = 0.0

        if last_stuck_check_pos is not None:
            dx = firebot.x - last_stuck_check_pos[0]
            dy = firebot.y - last_stuck_check_pos[1]
            dist_moved = math.sqrt(dx * dx + dy * dy)

            if dist_moved < stuck_threshold:
                print(f"Robot stuck! Only moved {dist_moved:.2f} cells. Forcing replan...")
                needs_replan = True

        last_stuck_check_pos = (firebot.x, firebot.y)

    # Replan if obstacles discovered or robot stuck
    if needs_replan:
        weight_grid = rebuild_weight_grid(fire_grid, fire_distance, exploration.get_known_obstacles())

        if len(planned_path) > 0 and target_pos is not None:
            start = (int(firebot.y), int(firebot.x))
            goal = (int(target_pos[1]), int(target_pos[0]))
            planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
            if planner.compute_shortest_path():
                new_path = planner.extract_path()
                if new_path:
                    planned_path = new_path
                    path_index = 0
                    last_stuck_check_pos = (firebot.x, firebot.y)  # Reset stuck check
                    print(f"Replanned: {len(planned_path)} waypoints")
                else:
                    print("Replan returned empty path!")
                    firebot.stop()
                    planned_path = []
            else:
                print("Replan failed - no path!")
                firebot.stop()
                planned_path = []

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

    # Render planned path (smooth path is list of (x, y))
    if show_path and len(planned_path) > 0:
        # Convert (x, y) to (row, col) for render_path
        path_cells = [(int(round(y)), int(round(x))) for x, y in planned_path]
        world.render_path(path_cells)

        # Also draw the smooth path as a line
        if len(planned_path) >= 2:
            points = []
            for x, y in planned_path:
                sx = int(world.field_rect.left + x * world.cell_size)
                sy = int(world.field_rect.top + y * world.cell_size)
                points.append((sx, sy))
            if len(points) >= 2:
                pygame.draw.lines(world.screen, (255, 100, 0), False, points, 2)

    world.render_fog_of_war(firebot)

    if world.show_weights:
        world.render_weight_heatmap(weight_grid)
        world.render_weight_on_hover(weight_grid, decimals=1)

    if world.show_arrows:
        world.render_gradient_arrows(weight_grid, spacing=1)

    if world.show_forbidden_zone:
        world.render_forbidden_zone(forbidden_zone, forbidden_zone_res)

    if target_pos is not None:
        world.render_target_marker(target_pos[0], target_pos[1])

    world.render_firebot(firebot)
    world.draw_hud()

    pygame.display.flip()

pygame.quit()