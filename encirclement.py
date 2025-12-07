# Fire Encirclement Planner
# RBE 550, Firebots (course project)
#
# Generates waypoints around the fire perimeter for the robot to follow,
# creating a complete fireline encirclement.

import math
import numpy as np
from planning import compute_fire_distance_field


class FireEncirclement:
    """
    Plans and tracks progress of fire encirclement mission.

    The robot follows waypoints around the fire perimeter, staying in
    the optimal corridor zone while cutting a fireline.
    """

    def __init__(
            self,
            fire_grid: np.ndarray,
            min_distance: float = 2.0,
            corridor_width: float = 3.0,
            waypoint_spacing: float = 3.0,
            obstacle_grid: np.ndarray = None,
    ):
        """
        Initialize encirclement planner.

        Args:
            fire_grid: Boolean grid where True = fire cell
            min_distance: Minimum safe distance from fire
            corridor_width: Width of optimal corridor
            waypoint_spacing: Approximate spacing between waypoints (cells)
            obstacle_grid: Boolean grid where True = obstacle (optional)
        """
        self.min_distance = min_distance
        self.corridor_width = corridor_width
        self.waypoint_spacing = waypoint_spacing
        self.obstacle_grid = obstacle_grid

        # Mission state
        self.waypoints = []  # List of (x, y) waypoints around fire
        self.current_waypoint_idx = 0
        self.mission_active = False
        self.mission_complete = False
        self.waypoints_visited = 0
        self.start_idx = 0

        # Compute initial waypoints
        if fire_grid.any():
            self._compute_waypoints(fire_grid)

    def _compute_waypoints(self, fire_grid: np.ndarray):
        """
        Compute waypoints around the fire perimeter.

        Finds cells in the optimal corridor zone and sorts them by angle
        around the fire centroid to create a traversal order.
        Filters out waypoints that are on or near obstacles.
        """
        rows, cols = fire_grid.shape

        # Compute distance field
        fire_distance = compute_fire_distance_field(fire_grid)

        # Find optimal corridor zone
        corridor_inner = self.min_distance
        corridor_outer = self.min_distance + self.corridor_width

        # Get cells in the corridor
        in_corridor = (fire_distance >= corridor_inner) & (fire_distance <= corridor_outer)

        # Filter out obstacle cells if obstacle grid provided
        if self.obstacle_grid is not None:
            # Also avoid cells adjacent to obstacles (robot needs clearance)
            from planning import _dilate8
            obstacle_buffer = _dilate8(self.obstacle_grid, 2)  # 2-cell buffer
            in_corridor = in_corridor & ~obstacle_buffer

        if not in_corridor.any():
            print("Warning: No cells in corridor zone!")
            self.waypoints = []
            return

        # Find fire centroid (for angle sorting)
        fire_positions = np.argwhere(fire_grid)
        if len(fire_positions) == 0:
            self.waypoints = []
            return

        centroid_row = fire_positions[:, 0].mean()
        centroid_col = fire_positions[:, 1].mean()

        # Get corridor cells and compute angles from centroid
        corridor_positions = np.argwhere(in_corridor)

        angles = []
        for row, col in corridor_positions:
            dy = row - centroid_row
            dx = col - centroid_col
            angle = math.atan2(dy, dx)
            angles.append(angle)

        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_positions = corridor_positions[sorted_indices]

        # Subsample to get evenly-spaced waypoints
        self.waypoints = self._subsample_waypoints(sorted_positions)

        print(f"Encirclement: {len(self.waypoints)} waypoints around fire")

    def _subsample_waypoints(self, positions: np.ndarray) -> list[tuple[float, float]]:
        """
        Subsample positions to get evenly-spaced waypoints.
        """
        if len(positions) == 0:
            return []

        waypoints = []
        last_pos = None
        spacing_sq = self.waypoint_spacing ** 2

        for row, col in positions:
            # Convert to (x, y) - cell center
            x = col + 0.5
            y = row + 0.5

            if last_pos is None:
                waypoints.append((x, y))
                last_pos = (x, y)
            else:
                dx = x - last_pos[0]
                dy = y - last_pos[1]
                dist_sq = dx * dx + dy * dy

                if dist_sq >= spacing_sq:
                    waypoints.append((x, y))
                    last_pos = (x, y)

        # Ensure we close the loop - check if last waypoint is far from first
        if len(waypoints) >= 2:
            first = waypoints[0]
            last = waypoints[-1]
            dx = first[0] - last[0]
            dy = first[1] - last[1]
            if dx * dx + dy * dy > spacing_sq:
                # Add intermediate point to close loop
                mid_x = (first[0] + last[0]) / 2
                mid_y = (first[1] + last[1]) / 2
                waypoints.append((mid_x, mid_y))

        return waypoints

    def update_fire(self, fire_grid: np.ndarray):
        """
        Update waypoints when fire spreads.

        Recomputes waypoints but tries to maintain progress.
        """
        if not self.mission_active:
            self._compute_waypoints(fire_grid)
            return

        # Store current target
        old_target = None
        if self.current_waypoint_idx < len(self.waypoints):
            old_target = self.waypoints[self.current_waypoint_idx]

        # Recompute waypoints
        self._compute_waypoints(fire_grid)

        # Try to find closest waypoint to old target
        if old_target and self.waypoints:
            best_idx = 0
            best_dist = float('inf')
            for i, (wx, wy) in enumerate(self.waypoints):
                dx = wx - old_target[0]
                dy = wy - old_target[1]
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            self.current_waypoint_idx = best_idx

    def update_obstacles(self, obstacle_grid: np.ndarray):
        """Update known obstacles (call when robot discovers new obstacles)."""
        self.obstacle_grid = obstacle_grid

    def skip_current_waypoint(self) -> bool:
        """
        Skip the current waypoint (e.g., if unreachable due to obstacles).

        Returns:
            True if mission complete after skipping, False otherwise
        """
        if not self.mission_active or not self.waypoints:
            return False

        print(f"Skipping blocked waypoint {self.current_waypoint_idx}")

        # Count as visited (skipped)
        self.waypoints_visited += 1

        # Move to next waypoint
        self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

        # Check if mission complete
        if self.waypoints_visited >= len(self.waypoints):
            self.mission_complete = True
            self.mission_active = False
            print("Encirclement mission COMPLETE (some waypoints skipped)")
            return True

        return False

    def start_mission(self, robot_x: float, robot_y: float):
        """
        Start encirclement mission from robot's current position.

        Finds the nearest waypoint to start from.
        """
        if not self.waypoints:
            print("Cannot start mission: no waypoints!")
            return False

        # Find nearest waypoint to robot
        best_idx = 0
        best_dist = float('inf')

        for i, (wx, wy) in enumerate(self.waypoints):
            dx = wx - robot_x
            dy = wy - robot_y
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        self.current_waypoint_idx = best_idx
        self.mission_active = True
        self.mission_complete = False
        self.start_idx = best_idx  # Remember where we started
        self.waypoints_visited = 0  # Track how many we've visited

        print(f"Encirclement mission started at waypoint {best_idx}/{len(self.waypoints)}")
        return True

    def get_current_target(self) -> tuple[float, float] | None:
        """Get the current waypoint target."""
        if not self.mission_active or not self.waypoints:
            return None

        if self.current_waypoint_idx >= len(self.waypoints):
            return None

        return self.waypoints[self.current_waypoint_idx]

    def advance_waypoint(self, robot_x: float, robot_y: float, threshold: float = 2.0) -> bool:
        """
        Check if robot reached current waypoint and advance to next.

        Args:
            robot_x, robot_y: Robot position
            threshold: Distance to consider waypoint "reached"

        Returns:
            True if mission is complete (full loop), False otherwise
        """
        if not self.mission_active or not self.waypoints:
            return False

        target = self.get_current_target()
        if target is None:
            return True

        # Check if reached current waypoint
        dx = target[0] - robot_x
        dy = target[1] - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < threshold:
            # Count this waypoint as visited
            self.waypoints_visited += 1

            # Advance to next waypoint (circular)
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

            # Check if we've completed the loop (visited all waypoints)
            if self.waypoints_visited >= len(self.waypoints):
                self.mission_complete = True
                self.mission_active = False
                print("Encirclement mission COMPLETE!")
                return True

            print(f"Waypoint reached! Now heading to {self.current_waypoint_idx}/{len(self.waypoints)}")

        return False

        return False

    def stop_mission(self):
        """Stop the encirclement mission."""
        self.mission_active = False
        print("Encirclement mission stopped")

    def get_progress(self) -> float:
        """Get mission progress as a fraction (0.0 to 1.0)."""
        if not self.waypoints:
            return 0.0

        if self.mission_complete:
            return 1.0

        if not self.mission_active:
            return 0.0

        return self.waypoints_visited / len(self.waypoints)

    def get_all_waypoints(self) -> list[tuple[float, float]]:
        """Get all waypoints for visualization."""
        return self.waypoints.copy()

    def get_remaining_waypoints(self) -> list[tuple[float, float]]:
        """Get waypoints not yet visited."""
        if not self.mission_active or not self.waypoints:
            return []

        remaining = []
        remaining_count = len(self.waypoints) - self.waypoints_visited
        idx = self.current_waypoint_idx

        for _ in range(remaining_count):
            remaining.append(self.waypoints[idx])
            idx = (idx + 1) % len(self.waypoints)

        return remaining