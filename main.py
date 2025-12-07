# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# D* Lite with subcell resolution for tight gap navigation

import math

import numpy as np
import pygame

from exploration import ExplorationMap
from d_star_lite import DStarLite
from fire_bitmap import load_fire_bitmap
from fire_spread import FireSpread
from firebot import Firebot
from obstacles import place_trees, clear_robot_start
from planning import (
    find_nearest_fire_approach_point,
    compute_fire_forbidden_zone,
    compute_fire_distance_field,
    rebuild_weight_grid,
)
from render import World

# =============================================================================
# TUNABLE PARAMETERS - Adjust these to change behavior
# =============================================================================

# --- Field Dimensions ---
COLS = 100  # Grid width in cells
ROWS = 60  # Grid height in cells
CELL_SIZE = 18  # Pixels per cell for rendering
TREE_COUNT = 75  # Number of random trees to place

# --- Robot Initial Position ---
ROBOT_X = 5  # Starting X position (cells)
ROBOT_Y = ROWS / 2  # Starting Y position (cells)

# --- Fire Simulation ---
SPREAD_PACE = 0.5  # Seconds per new cell catching fire

# --- Cost Grid: Base ---
BASE_COST = 1.0  # Baseline traversal cost for all cells

# --- Cost Grid: Fire Corridor ---
FIRE_MIN_DISTANCE = 1.5    # Cells closer than this are impassable (inf cost)
FIRE_CORRIDOR_WIDTH = 2.0  # Width of optimal corridor (cost = 0)
                           # Corridor spans: MIN_DISTANCE to MIN_DISTANCE + WIDTH
FIRE_FALLOFF_RATE = 10.0   # Cost increase per cell BEYOND corridor

# --- Cost Grid: Tree Avoidance ---
# NOTE: With subcell planning, we can use smaller min_distance.
#       The planner can find paths through tight gaps by positioning
#       the robot center between cell centers.
TREE_MIN_DISTANCE = 1.5    # Cells within this distance = impassable
                           # Reduced to allow tight gap navigation
TREE_TRUNK_SCALE = 50.0    # Peak avoidance penalty beyond min_distance
TREE_TRUNK_TAU = 2.0       # Decay rate (larger = wider influence)
TREE_TRUNK_MAX_RADIUS = 5  # Max influence radius

# --- Path Extraction ---
PATH_STEP_SIZE = 1.0       # Distance per gradient step (cells)
PATH_MOMENTUM = 0.6        # Blend with previous direction (0.0 - 1.0)
PATH_GOAL_TOLERANCE = 1.0  # Distance to goal to consider "arrived" (cells)

# --- Path Smoothing ---
PATH_SMOOTH_ITERATIONS = 2 # Number of smoothing passes
PATH_SMOOTH_WEIGHT = 0.5    # Smoothing aggressiveness (0.0 - 1.0)
                            # Lower = less corner cutting in tight spots

# --- Pure Pursuit Controller ---
PURSUIT_LOOKAHEAD = 1  # How far ahead to look on path (cells)
                         # REDUCED to follow path more closely in tight gaps

# --- D* Lite Planner ---
HEURISTIC_WEIGHT = 3.0  # Weight on heuristic (goal-directedness)
                        # 1.0 = optimal paths (may explore wrong direction)
                        # 2.0+ = more greedy, faster

# --- Planning Resolution (KEY FOR TIGHT GAPS) ---
PLANNING_RESOLUTION = 1  # Subcells per world cell
                         # 1 = original cell-based planning
                         # 2 = half-cell precision (can find paths through 3-cell gaps)
                         # 4 = quarter-cell precision (expensive but very precise)

# --- Stuck Detection ---
STUCK_CHECK_INTERVAL = 0.5  # How often to check if stuck (seconds)
STUCK_THRESHOLD = 0.1       # Minimum distance to move per interval (cells)
MAX_STUCK_REPLANS = 3       # Max consecutive replan attempts when stuck

# --- Forbidden Zone Visualization ---
FORBIDDEN_ZONE_RESOLUTION = 4  # Subcells per cell for forbidden zone rendering


# =============================================================================
# END TUNABLE PARAMETERS
# =============================================================================


def extract_path_with_params(planner):
    """Helper to extract path using current parameter settings."""
    return planner.extract_path(
        step_size=PATH_STEP_SIZE,
        momentum=PATH_MOMENTUM,
        goal_tolerance=PATH_GOAL_TOLERANCE,
        smooth_iterations=PATH_SMOOTH_ITERATIONS,
        smooth_weight=PATH_SMOOTH_WEIGHT,
    )


def build_weights(fire_grid, fire_distance, known_trees):
    """Helper to build weight grid using current parameter settings."""
    return rebuild_weight_grid(
        fire_grid,
        fire_distance,
        known_trees,
        base_cost=BASE_COST,
        # Fire params
        fire_min_distance=FIRE_MIN_DISTANCE,
        fire_corridor_width=FIRE_CORRIDOR_WIDTH,
        fire_falloff_rate=FIRE_FALLOFF_RATE,
        # Tree params
        tree_min_distance=TREE_MIN_DISTANCE,
        tree_trunk_scale=TREE_TRUNK_SCALE,
        tree_trunk_tau=TREE_TRUNK_TAU,
        tree_trunk_max_radius=TREE_TRUNK_MAX_RADIUS,
    )


# Initialize pygame and world
rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots - D* Lite")
world = World(ROWS, COLS, CELL_SIZE)

# Load fire bitmap (scale=0.5 makes fire half the size, centered)
fire_surface, fire_grid = load_fire_bitmap("fire2.png", world.cols, world.rows, scale=0.75)

# Generate obstacles
tree_grid = place_trees(world.cols, world.rows, count=TREE_COUNT, rng=rng)

# Clear robots start position
clear_robot_start(tree_grid, robot_x=ROBOT_X, robot_y=ROBOT_Y)

# Create exploration map
exploration = ExplorationMap(world.rows, world.cols)

# Create fire spread simulation
fire_spread = FireSpread(fire_grid, SPREAD_PACE)

# === Known fire grid - robot knows original fire, discovers spread in sensor range ===
known_fire_grid = fire_grid.copy()

# Create firebot at initial position
firebot = Firebot(x=ROBOT_X, y=ROBOT_Y, theta=0.0)

# Initial exploration from starting position
exploration.update(firebot.x, firebot.y, firebot.sensor_radius, tree_grid)

# Build weight grid with known fire and known obstacles
fire_distance = compute_fire_distance_field(known_fire_grid)
weight_grid = build_weights(known_fire_grid, fire_distance, exploration.get_known_obstacles())

# Debug output
finite = weight_grid[np.isfinite(weight_grid)]
print(f"Weight range: min={finite.min():.1f}, max={finite.max():.1f}")
print(f"\n=== Current Parameters ===")
print(f"Fire: min_dist={FIRE_MIN_DISTANCE}, corridor={FIRE_CORRIDOR_WIDTH}, falloff={FIRE_FALLOFF_RATE}")
print(f"Tree: min_dist={TREE_MIN_DISTANCE}, scale={TREE_TRUNK_SCALE}, tau={TREE_TRUNK_TAU}")
print(f"Planning resolution: {PLANNING_RESOLUTION}x (grid: {ROWS*PLANNING_RESOLUTION}x{COLS*PLANNING_RESOLUTION})")
print(f"Pure pursuit lookahead: {PURSUIT_LOOKAHEAD}")
print(f"D* Lite heuristic_weight: {HEURISTIC_WEIGHT}")
print(f"==========================\n")

# Create D* Lite planner with subcell resolution
planner = DStarLite(
    ROWS, COLS,
    heuristic_weight=HEURISTIC_WEIGHT,
    planning_resolution=PLANNING_RESOLUTION,
)
planned_path = []  # Will contain (x, y) tuples from D* Lite
path_index = 0

# Compute forbidden zone for visualization
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
stuck_timer = 0.0
last_stuck_check_pos = None
stuck_replan_count = 0  # Track consecutive stuck replans

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
            elif hasattr(
                    world, "arrows_toggle_rect"
            ) and world.arrows_toggle_rect.collidepoint(e.pos):
                world.show_arrows = not world.show_arrows
            elif hasattr(
                    world, "forbidden_zone_toggle_rect"
            ) and world.forbidden_zone_toggle_rect.collidepoint(e.pos):
                world.show_forbidden_zone = not world.show_forbidden_zone
            elif hasattr(
                    world, "fireline_grid_toggle_rect"
            ) and world.fireline_grid_toggle_rect.collidepoint(e.pos):
                world.show_fireline_grid = not world.show_fireline_grid
            else:
                cell_pos = world.screen_to_cell(e.pos[0], e.pos[1])
                if cell_pos is not None:
                    cx, cy = cell_pos
                    if 0 <= cx < COLS and 0 <= cy < ROWS:
                        target_pos = cell_pos
                        start = (int(firebot.y), int(firebot.x))
                        goal = (int(cell_pos[1]), int(cell_pos[0]))

                        print(f"Click at ({cx:.1f}, {cy:.1f}) -> Planning from {start} to {goal}")
                        planner.initialize(
                            start, goal, weight_grid, exploration.get_known_obstacles()
                        )
                        if planner.compute_shortest_path():
                            planned_path = extract_path_with_params(planner)
                            path_index = 0
                            last_stuck_check_pos = (firebot.x, firebot.y)
                            stuck_replan_count = 0
                            print(f"Path has {len(planned_path)} waypoints")
                            if planned_path:
                                end = planned_path[-1]
                                print(f"Path ends at ({end[0]:.1f}, {end[1]:.1f})")
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
                    planner.initialize(
                        start, goal, weight_grid, exploration.get_known_obstacles()
                    )
                    if planner.compute_shortest_path():
                        planned_path = extract_path_with_params(planner)
                        path_index = 0
                        stuck_replan_count = 0
                        print(f"Path to fire: {len(planned_path)} waypoints")

    # Path following
    if path_following_enabled and len(planned_path) > 0:
        if use_smooth_control:
            # Pure pursuit - smooth motion
            still_going = firebot.pure_pursuit_step(
                planned_path, dt, lookahead=PURSUIT_LOOKAHEAD
            )

            if not still_going:
                print("Path complete!")
                planned_path = []
                target_pos = None
        else:
            # Original rotate-then-drive behavior using set_target
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

    # Mark fireline cells as robot moves
    if firebot.cutting_fireline and firebot.is_moving():
        fire_spread.mark_fireline(
            firebot.x,
            firebot.y,
            firebot.theta,
            blade_width=1.5,
            robot_size=firebot.size,
        )

    # Update fire spread simulation
    fire_spread.update(dt)

    # === Detect spread fire within sensor range ===
    new_fire_detected = False
    unknown_fire = fire_grid & ~known_fire_grid
    if unknown_fire.any():
        fire_rows, fire_cols = np.where(unknown_fire)
        dx = (fire_cols + 0.5) - firebot.x
        dy = (fire_rows + 0.5) - firebot.y
        distances = np.sqrt(dx * dx + dy * dy)
        visible = distances <= firebot.sensor_radius
        if visible.any():
            known_fire_grid[fire_rows[visible], fire_cols[visible]] = True
            new_fire_detected = True
            fire_distance = compute_fire_distance_field(known_fire_grid)
            weight_grid = build_weights(known_fire_grid, fire_distance, exploration.get_known_obstacles())
            print(f"New fire detected! {visible.sum()} cells discovered")

    # Update exploration
    new_obstacles_found = exploration.update(
        firebot.x, firebot.y, firebot.sensor_radius, tree_grid
    )

    # Stuck detection - check if robot is making progress
    stuck_timer += dt
    needs_replan = new_obstacles_found or new_fire_detected
    stuck_triggered = False

    if stuck_timer >= STUCK_CHECK_INTERVAL and len(planned_path) > 0:
        stuck_timer = 0.0

        if last_stuck_check_pos is not None:
            dx = firebot.x - last_stuck_check_pos[0]
            dy = firebot.y - last_stuck_check_pos[1]
            dist_moved = math.sqrt(dx * dx + dy * dy)

            if dist_moved < STUCK_THRESHOLD:
                stuck_replan_count += 1
                print(
                    f"Robot stuck! Only moved {dist_moved:.2f} cells. "
                    f"Replan attempt {stuck_replan_count}/{MAX_STUCK_REPLANS}"
                )

                if stuck_replan_count >= MAX_STUCK_REPLANS:
                    print("Max stuck replans reached - stopping robot")
                    firebot.stop()
                    planned_path = []
                    target_pos = None
                    stuck_replan_count = 0
                else:
                    needs_replan = True
                    stuck_triggered = True
            else:
                # Robot is making progress, reset counter
                stuck_replan_count = 0

        last_stuck_check_pos = (firebot.x, firebot.y)

    # Replan if obstacles discovered, new fire detected, or robot stuck
    if needs_replan and len(planned_path) > 0 and target_pos is not None:
        weight_grid = build_weights(
            known_fire_grid, fire_distance, exploration.get_known_obstacles()
        )

        start = (int(firebot.y), int(firebot.x))
        goal = (int(target_pos[1]), int(target_pos[0]))
        planner.initialize(
            start, goal, weight_grid, exploration.get_known_obstacles()
        )
        if planner.compute_shortest_path():
            new_path = extract_path_with_params(planner)
            if new_path and planner.is_path_valid(new_path):
                planned_path = new_path
                path_index = 0
                last_stuck_check_pos = (firebot.x, firebot.y)
                if not stuck_triggered:
                    stuck_replan_count = 0
                print(f"Replanned: {len(planned_path)} waypoints")
            else:
                print("Replan produced invalid path - keeping current path")
                if stuck_triggered:
                    stuck_replan_count += 1
        else:
            print("Replan failed - no path!")
            if stuck_triggered:
                stuck_replan_count += 1
            if stuck_replan_count >= MAX_STUCK_REPLANS:
                firebot.stop()
                planned_path = []
                stuck_replan_count = 0

    # === Rendering ===
    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_grid)  # Show actual fire (user can see it)
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

    if world.show_fireline_grid:
        world.render_fireline_cells(fire_spread.fireline_grid)

    if target_pos is not None:
        world.render_target_marker(target_pos[0], target_pos[1])

    world.render_firebot(firebot)
    world.draw_hud()

    pygame.display.flip()

pygame.quit()