# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# D* Lite with subcell resolution for tight gap navigation
# + Sequential Encirclement (simplified)

import math

import numpy as np
import pygame

from d_star_lite import DStarLite
from encirclement import EncirclementPlanner, EncirclementState
from exploration import ExplorationMap
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

# --- Cost Grid: Fire Potential Field ---
FIRE_MIN_DISTANCE = 2.0  # Cells closer than this are impassable (inf cost)
FIRE_IDEAL_DISTANCE = 5.0  # Optimal distance from fire (minimum cost here)
FIRE_INNER_REPULSION = 10.0  # Repulsion strength for being too close to fire
FIRE_OUTER_REPULSION = 1.2  # Repulsion strength for being too far from fire

# --- Cost Grid: Tree Potential Field ---
TREE_MIN_DISTANCE = 1  # Cells within this distance = impassable
TREE_REPULSION_STRENGTH = 50.0  # Peak repulsion at min_distance
TREE_REPULSION_DECAY = 2.0  # Decay rate (larger = wider influence)
TREE_MAX_RADIUS = 5  # Max influence radius

# --- Path Extraction ---
PATH_STEP_SIZE = 1.0  # Distance per gradient step (cells)
PATH_MOMENTUM = 0.6  # Blend with previous direction (0.0 - 1.0)
PATH_GOAL_TOLERANCE = 1.0  # Distance to goal to consider "arrived" (cells)

# --- Path Smoothing ---
PATH_SMOOTH_ITERATIONS = 2  # Number of smoothing passes
PATH_SMOOTH_WEIGHT = 0.5  # Smoothing aggressiveness (0.0 - 1.0)

# --- Pure Pursuit Controller ---
PURSUIT_LOOKAHEAD = 1  # How far ahead to look on path (cells)

# --- D* Lite Planner ---
HEURISTIC_WEIGHT = 3.0  # Weight on heuristic (goal-directedness)

# --- Planning Resolution (KEY FOR TIGHT GAPS) ---
PLANNING_RESOLUTION = 1  # Subcells per world cell

# --- Stuck Detection ---
STUCK_CHECK_INTERVAL = 0.5  # How often to check if stuck (seconds)
STUCK_THRESHOLD = 0.1  # Minimum distance to move per interval (cells)
MAX_STUCK_REPLANS = 3  # Max consecutive replan attempts when stuck

# --- Forbidden Zone Visualization ---
FORBIDDEN_ZONE_RESOLUTION = 4  # Subcells per cell for forbidden zone rendering

# --- Encirclement Parameters ---
ENCIRCLEMENT_WAYPOINT_SPACING = 20.0  # Target distance between waypoints (cells)
ENCIRCLEMENT_WAYPOINT_THRESHOLD = 2  # Distance to consider waypoint "reached"
ENCIRCLEMENT_RELOCATION_RADIUS = 15.0  # Search radius for optimizing waypoint positions
ENCIRCLEMENT_MIN_SPACING = 10.0  # Minimum allowed distance between waypoints
ENCIRCLEMENT_MIN_FIRE_DIST = 3.0  # Minimum distance waypoints must be from fire
ENCIRCLEMENT_MAX_WAYPOINT_COST = (
    5.0  # Max cost for valid waypoint (rejects positions near obstacles)
)


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


def plan_to_current_waypoint(
    encirclement, planner, weight_grid, known_obstacles, robot_x, robot_y
):
    """
    Plan a path to the current encirclement waypoint.

    Returns:
        (target_pos, path, success)
    """
    wp = encirclement.get_current_waypoint()
    if wp is None:
        return None, [], False

    wx, wy = wp
    start = (int(robot_y), int(robot_x))
    goal = (int(wy), int(wx))

    planner.initialize(start, goal, weight_grid, known_obstacles)

    if planner.compute_shortest_path():
        path = extract_path_with_params(planner)
        if path and planner.is_path_valid(path):
            return (wx, wy), path, True

    return None, [], False


def build_weights(fire_grid, fire_distance, known_trees):
    """Helper to build weight grid using current parameter settings."""
    return rebuild_weight_grid(
        fire_grid,
        fire_distance,
        known_trees,
        base_cost=BASE_COST,
        # Fire potential field params
        fire_min_distance=FIRE_MIN_DISTANCE,
        fire_ideal_distance=FIRE_IDEAL_DISTANCE,
        fire_inner_repulsion=FIRE_INNER_REPULSION,
        fire_outer_repulsion=FIRE_OUTER_REPULSION,
        # Tree potential field params
        tree_min_distance=TREE_MIN_DISTANCE,
        tree_repulsion_strength=TREE_REPULSION_STRENGTH,
        tree_repulsion_decay=TREE_REPULSION_DECAY,
        tree_max_radius=TREE_MAX_RADIUS,
    )


# Initialize pygame and world
rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Firebots - D* Lite + Encirclement")
world = World(ROWS, COLS, CELL_SIZE)

# Load fire bitmap - 72x36 cells centered on the field
FIRE_COLS = 72
FIRE_ROWS = 36
fire_surface, fire_grid, fire_bounds = load_fire_bitmap(
    "fire1.png", world.cols, world.rows, fire_cols=FIRE_COLS, fire_rows=FIRE_ROWS
)

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
weight_grid = build_weights(
    known_fire_grid, fire_distance, exploration.get_known_obstacles()
)

# Debug output
finite = weight_grid[np.isfinite(weight_grid)]
print(f"Weight range: min={finite.min():.1f}, max={finite.max():.1f}")
print(f"\n=== Current Parameters ===")
print(f"Fire potential: min_dist={FIRE_MIN_DISTANCE}, ideal_dist={FIRE_IDEAL_DISTANCE}")
print(f"Fire repulsion: inner={FIRE_INNER_REPULSION}, outer={FIRE_OUTER_REPULSION}")
print(
    f"Tree potential: min_dist={TREE_MIN_DISTANCE}, strength={TREE_REPULSION_STRENGTH}, decay={TREE_REPULSION_DECAY}"
)
print(
    f"Planning resolution: {PLANNING_RESOLUTION}x (grid: {ROWS * PLANNING_RESOLUTION}x{COLS * PLANNING_RESOLUTION})"
)
print(f"Pure pursuit lookahead: {PURSUIT_LOOKAHEAD}")
print(f"D* Lite heuristic_weight: {HEURISTIC_WEIGHT}")
print(f"Encirclement waypoint spacing: {ENCIRCLEMENT_WAYPOINT_SPACING}")
print(f"==========================\n")

# Create D* Lite planner with subcell resolution
planner = DStarLite(
    ROWS,
    COLS,
    heuristic_weight=HEURISTIC_WEIGHT,
    planning_resolution=PLANNING_RESOLUTION,
)
planned_path = []  # Will contain (x, y) tuples from D* Lite
path_index = 0

# Create encirclement planner
encirclement = EncirclementPlanner(
    ROWS,
    COLS,
    robot_size=3,  # 3x3 robot footprint
    waypoint_spacing=ENCIRCLEMENT_WAYPOINT_SPACING,
    min_waypoint_spacing=ENCIRCLEMENT_MIN_SPACING,
    waypoint_reached_threshold=ENCIRCLEMENT_WAYPOINT_THRESHOLD,
    min_fire_distance=ENCIRCLEMENT_MIN_FIRE_DIST,
    relocation_search_radius=ENCIRCLEMENT_RELOCATION_RADIUS,
    max_waypoint_cost=ENCIRCLEMENT_MAX_WAYPOINT_COST,
)

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

# Encirclement state
encirclement_mode = False  # True when actively encircling

# Verification mode - test if fireline is closed after encirclement
verification_mode = False
verification_start_fire_count = 0

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
            elif hasattr(
                world, "waypoints_toggle_rect"
            ) and world.waypoints_toggle_rect.collidepoint(e.pos):
                world.show_waypoints = not world.show_waypoints
            else:
                # Manual click navigation (disabled during encirclement)
                if not encirclement_mode:
                    cell_pos = world.screen_to_cell(e.pos[0], e.pos[1])
                    if cell_pos is not None:
                        cx, cy = cell_pos
                        if 0 <= cx < COLS and 0 <= cy < ROWS:
                            target_pos = cell_pos
                            start = (int(firebot.y), int(firebot.x))
                            goal = (int(cell_pos[1]), int(cell_pos[0]))

                            print(
                                f"Click at ({cx:.1f}, {cy:.1f}) -> Planning from {start} to {goal}"
                            )
                            planner.initialize(
                                start,
                                goal,
                                weight_grid,
                                exploration.get_known_obstacles(),
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
                encirclement_mode = False
                verification_mode = False
                encirclement.reset()
                print("Movement cancelled, encirclement/verification reset")
            elif e.key == pygame.K_f:
                # Find nearest fire approach point
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

            # === ENCIRCLEMENT START ===
            elif e.key == pygame.K_e:
                if encirclement.is_active():
                    # Already encircling - cancel
                    encirclement.reset()
                    encirclement_mode = False
                    planned_path = []
                    target_pos = None
                    firebot.stop()
                    print("Encirclement cancelled")
                else:
                    # Start encirclement
                    print("\n=== Starting Encirclement ===")
                    success = encirclement.generate_waypoints(
                        known_fire_grid,
                        fire_distance,
                        weight_grid,
                        firebot.x,
                        firebot.y,
                    )
                    if success:
                        encirclement_mode = True

                        # Plan to first waypoint (index 0)
                        target_pos, planned_path, plan_success = (
                            plan_to_current_waypoint(
                                encirclement,
                                planner,
                                weight_grid,
                                exploration.get_known_obstacles(),
                                firebot.x,
                                firebot.y,
                            )
                        )

                        if plan_success:
                            last_stuck_check_pos = (firebot.x, firebot.y)
                            stuck_replan_count = 0
                            print(
                                f"Planning to waypoint {encirclement.get_current_waypoint_index()}: {len(planned_path)} path points"
                            )
                        else:
                            print("No path to first waypoint!")
                            encirclement_mode = False
                            encirclement.reset()
                    else:
                        print("Failed to generate encirclement waypoints!")

            # === MANUAL VERIFICATION ===
            elif e.key == pygame.K_v:
                if not verification_mode and not encirclement_mode:
                    print("\n=== Starting Manual Verification ===")
                    verification_mode = True
                    verification_start_fire_count = fire_grid.sum()
                    world.show_fog_of_war = False
                elif verification_mode:
                    print("Verification cancelled")
                    verification_mode = False

    # === Encirclement waypoint management (SIMPLIFIED - SEQUENTIAL) ===
    if encirclement_mode and encirclement.is_active():
        current_wp = encirclement.get_current_waypoint()

        if current_wp is not None:
            # Check if we reached the current waypoint
            if encirclement.waypoint_reached(firebot.x, firebot.y):
                wp_idx = encirclement.get_current_waypoint_index()
                print(f"Reached waypoint {wp_idx}")

                # Mark as visited and move to next
                has_more = encirclement.advance_waypoint()

                if has_more:
                    # Plan to next waypoint
                    target_pos, planned_path, plan_success = plan_to_current_waypoint(
                        encirclement,
                        planner,
                        weight_grid,
                        exploration.get_known_obstacles(),
                        firebot.x,
                        firebot.y,
                    )

                    if plan_success:
                        last_stuck_check_pos = (firebot.x, firebot.y)
                        stuck_replan_count = 0
                        print(
                            f"Going to waypoint {encirclement.get_current_waypoint_index()}"
                        )
                    else:
                        # Can't reach this waypoint - skip it
                        print(
                            f"Can't reach waypoint {encirclement.get_current_waypoint_index()}, skipping"
                        )
                        encirclement.skip_current_waypoint()
                        # Try next one
                        target_pos, planned_path, _ = plan_to_current_waypoint(
                            encirclement,
                            planner,
                            weight_grid,
                            exploration.get_known_obstacles(),
                            firebot.x,
                            firebot.y,
                        )
                else:
                    # Encirclement complete
                    encirclement_mode = False
                    planned_path = []
                    target_pos = None
                    firebot.stop()

            # If we have no path but encirclement is active, try to plan
            elif len(planned_path) == 0:
                target_pos, planned_path, plan_success = plan_to_current_waypoint(
                    encirclement,
                    planner,
                    weight_grid,
                    exploration.get_known_obstacles(),
                    firebot.x,
                    firebot.y,
                )

                if plan_success:
                    last_stuck_check_pos = (firebot.x, firebot.y)
                    stuck_replan_count = 0
                    print(
                        f"Re-planning to waypoint {encirclement.get_current_waypoint_index()}"
                    )

        # Handle encirclement completion
        if encirclement.is_complete():
            print("\n=== ENCIRCLEMENT COMPLETE! ===")
            print("Starting fire spread verification...")
            encirclement_mode = False
            planned_path = []
            target_pos = None
            firebot.stop()

            # Start verification mode
            verification_mode = True
            verification_start_fire_count = fire_grid.sum()
            world.show_fog_of_war = False

        elif encirclement.is_failed():
            print("\n=== ENCIRCLEMENT FAILED ===")
            encirclement_mode = False
            planned_path = []
            target_pos = None
            firebot.stop()

    # Path following
    if path_following_enabled and len(planned_path) > 0:
        if use_smooth_control:
            # Pure pursuit - smooth motion
            still_going = firebot.pure_pursuit_step(
                planned_path, dt, lookahead=PURSUIT_LOOKAHEAD
            )

            if not still_going:
                if not encirclement_mode:
                    print("Path complete!")
                    planned_path = []
                    target_pos = None
                # In encirclement mode, waypoint check handles completion
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
                    if not encirclement_mode:
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
    if not verification_mode:
        fire_spread.update(dt)
    else:
        # Verification mode: spread fire rapidly to test containment
        # Spread multiple cells per frame for faster verification
        cells_per_frame = 30

        for _ in range(cells_per_frame):
            # Try to spread fire
            spread_happened = fire_spread._ignite_one_neighbor()

            if not spread_happened:
                # No more cells to spread to - fire is CONTAINED!
                final_fire_count = fire_grid.sum()
                print(f"\n{'=' * 50}")
                print(f"=== FIRELINE VERIFICATION: SUCCESS! ===")
                print(
                    f"Fire contained! Started with {verification_start_fire_count} cells, ended with {final_fire_count} cells"
                )
                print(
                    f"Fireline held - fire filled {final_fire_count - verification_start_fire_count} cells inside the ring"
                )
                print(f"{'=' * 50}\n")
                verification_mode = False
                break

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
            weight_grid = build_weights(
                known_fire_grid, fire_distance, exploration.get_known_obstacles()
            )
            print(f"New fire detected! {visible.sum()} cells discovered")

            # Update encirclement waypoints for new fire
            if encirclement_mode and encirclement.is_active():
                _, waypoint_changed = encirclement.update_waypoints_for_new_obstacles(
                    weight_grid, fire_distance
                )
                if waypoint_changed:
                    # Current waypoint moved - replan
                    new_wp = encirclement.get_current_waypoint()
                    if new_wp:
                        target_pos = new_wp
                        print(
                            f"Current waypoint relocated to ({new_wp[0]:.1f}, {new_wp[1]:.1f})"
                        )

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
                    if encirclement_mode:
                        # Skip this waypoint and try next
                        print(
                            f"Skipping waypoint {encirclement.get_current_waypoint_index()}"
                        )
                        encirclement.skip_current_waypoint()
                        stuck_replan_count = 0

                        # Plan to next waypoint
                        target_pos, planned_path, plan_success = (
                            plan_to_current_waypoint(
                                encirclement,
                                planner,
                                weight_grid,
                                exploration.get_known_obstacles(),
                                firebot.x,
                                firebot.y,
                            )
                        )

                        if plan_success:
                            last_stuck_check_pos = (firebot.x, firebot.y)
                        elif not encirclement.is_active():
                            encirclement_mode = False
                            planned_path = []
                            target_pos = None
                            firebot.stop()
                    else:
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

        # Update encirclement waypoints for new obstacles
        if encirclement_mode and encirclement.is_active() and new_obstacles_found:
            _, waypoint_changed = encirclement.update_waypoints_for_new_obstacles(
                weight_grid, fire_distance
            )
            if waypoint_changed:
                new_wp = encirclement.get_current_waypoint()
                if new_wp:
                    target_pos = new_wp

        # Replan to current target
        start = (int(firebot.y), int(firebot.x))
        goal = (int(target_pos[1]), int(target_pos[0]))
        planner.initialize(start, goal, weight_grid, exploration.get_known_obstacles())
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
                if not encirclement_mode:
                    firebot.stop()
                    planned_path = []
                    stuck_replan_count = 0

    # === Rendering ===
    world.clear()
    world.fire_bitmap_overlay(fire_surface, fire_bounds)
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

    # Render encirclement corridor cells (for debugging)
    corridor_cells = encirclement.get_corridor_cells()
    if corridor_cells and world.show_waypoints:
        world.render_corridor_cells(corridor_cells, color=(0, 255, 255), alpha=60)

    # Render encirclement waypoints
    if encirclement.is_active() or encirclement.is_complete():
        waypoints = encirclement.get_waypoints()
        states = encirclement.get_waypoint_states()
        world.render_encirclement_waypoints(
            waypoints, states, encirclement.get_current_waypoint_index()
        )

    if world.show_fog_of_war:
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

    # Draw HUD with encirclement/verification state
    if verification_mode:
        current_fire = fire_grid.sum()
        spread = current_fire - verification_start_fire_count
        hud_state = f"VERIFYING (+{spread} cells)"
    elif encirclement.state != EncirclementState.IDLE:
        hud_state = encirclement.state.value
    else:
        hud_state = None
    world.draw_hud(hud_state)

    pygame.display.flip()

pygame.quit()
