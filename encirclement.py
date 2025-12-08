# Encirclement Planner - Global planner for fire containment
# RBE 550, Firebots (course project)
# Michael Laks & Tom Kazakov
#
# MODIFIED: Sequential waypoint visiting + loop closure

import math
from enum import Enum
from typing import Optional

import numpy as np


class EncirclementState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class EncirclementPlanner:
    """
    Global planner that generates waypoints to encircle a fire zone.

    Strategy:
    1. Dynamically find the passable low-cost corridor around fire
    2. Use robot-size checks to ensure the robot can actually fit
    3. Extract contour of valid cells and order by angle around fire centroid
    4. Handle map boundaries (contour ends at edges)
    5. Sample waypoints with minimum spacing constraints
    6. Visit waypoints SEQUENTIALLY and return to start to close the loop
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        robot_size: int = 3,
        waypoint_spacing: float = 15.0,
        min_waypoint_spacing: float = 10.0,
        waypoint_reached_threshold: float = 1.0,
        min_fire_distance: float = 3.0,
        relocation_search_radius: float = 10.0,
        max_waypoint_cost: float = 15.0,
    ):
        """
        Initialize the encirclement planner.

        Args:
            rows: Grid height
            cols: Grid width
            robot_size: Robot footprint size (e.g., 3 for 3x3)
            waypoint_spacing: Target distance between waypoints (cells)
            min_waypoint_spacing: Minimum allowed distance between waypoints (cells)
            waypoint_reached_threshold: How close robot must be to "reach" waypoint
            min_fire_distance: Minimum distance waypoints must be from fire (cells)
            relocation_search_radius: How far to search when optimizing waypoints
            max_waypoint_cost: Maximum cost for a valid waypoint position
        """
        self.rows = rows
        self.cols = cols
        self.robot_size = robot_size
        self.robot_offset = robot_size // 2
        self.waypoint_spacing = waypoint_spacing
        self.min_waypoint_spacing = min_waypoint_spacing
        self.waypoint_reached_threshold = waypoint_reached_threshold
        self.min_fire_distance = min_fire_distance
        self.relocation_search_radius = relocation_search_radius
        self.max_waypoint_cost = max_waypoint_cost

        self.state = EncirclementState.IDLE
        self.waypoints: list[tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.start_waypoint_idx = 0  # Track where we started for loop closure
        self.skipped_waypoints: set[int] = set()
        self.visited_waypoints: set[int] = set()
        self.returning_to_start = False  # Flag for closing the loop

        # For direction selection
        self.going_clockwise = True

        # Fire centroid for reference
        self.fire_centroid: Optional[tuple[float, float]] = None

        # Map boundary flags
        self.touches_boundary = False

        # Store corridor cells for visualization
        self.corridor_cells: list[tuple[int, int]] = []

    def generate_waypoints(
        self,
        fire_grid: np.ndarray,
        fire_distance: np.ndarray,
        weight_grid: np.ndarray,
        robot_x: float,
        robot_y: float,
    ) -> bool:
        """
        Generate waypoints to encircle the fire.

        Uses dynamic corridor finding based on cost map and robot passability.

        Args:
            fire_grid: Boolean grid where True = fire cell
            fire_distance: Distance field from fire
            weight_grid: Cost grid (includes tree costs, fire costs, etc.)
            robot_x, robot_y: Robot position

        Returns:
            True if waypoints generated successfully, False if impossible
        """
        self.state = EncirclementState.IDLE
        self.waypoints.clear()
        self.current_waypoint_idx = 0
        self.start_waypoint_idx = 0
        self.skipped_waypoints.clear()
        self.visited_waypoints.clear()
        self.returning_to_start = False

        # Find fire centroid
        fire_positions = np.argwhere(fire_grid)
        if len(fire_positions) == 0:
            print("Encirclement: No fire found!")
            self.state = EncirclementState.FAILED
            return False

        self.fire_centroid = (
            float(fire_positions[:, 1].mean()) + 0.5,  # x
            float(fire_positions[:, 0].mean()) + 0.5,  # y
        )

        # Check if fire touches map boundary
        self.touches_boundary = self._fire_touches_boundary(fire_grid)

        # Find the passable corridor dynamically
        corridor_cells = self._find_passable_corridor(fire_distance, weight_grid)

        if len(corridor_cells) == 0:
            print("Encirclement: No valid corridor cells found!")
            self.state = EncirclementState.FAILED
            return False

        print(f"Encirclement: Found {len(corridor_cells)} valid corridor cells")

        # Order contour by angle around fire centroid
        ordered_contour = self._order_contour_by_angle(corridor_cells)

        if len(ordered_contour) < 4:
            print("Encirclement: Contour too small!")
            self.state = EncirclementState.FAILED
            return False

        # Handle map boundary case - find where contour has gaps
        if self.touches_boundary:
            ordered_contour = self._handle_boundary_gaps(ordered_contour)

        # Find starting point (nearest to robot) and choose direction
        start_idx, self.going_clockwise = self._find_start_and_direction(
            ordered_contour, robot_x, robot_y
        )

        # Reorder contour to start from nearest point
        if self.going_clockwise:
            ordered_contour = ordered_contour[start_idx:] + ordered_contour[:start_idx]
        else:
            # Reverse for counter-clockwise
            ordered_contour = (
                ordered_contour[start_idx::-1] + ordered_contour[:start_idx:-1]
            )

        # Sample waypoints with spacing constraints
        raw_waypoints = self._sample_waypoints_with_spacing(
            ordered_contour, weight_grid, fire_distance
        )

        if len(raw_waypoints) < 2:
            print("Encirclement: Not enough waypoints after spacing filter!")
            self.state = EncirclementState.FAILED
            return False

        self.waypoints = raw_waypoints

        if len(self.waypoints) < 2:
            print("Encirclement: All waypoints blocked!")
            self.state = EncirclementState.FAILED
            return False

        # Set start waypoint index (always 0 since we reordered the contour)
        self.start_waypoint_idx = 0
        self.current_waypoint_idx = 0

        self.state = EncirclementState.ACTIVE
        print(f"Encirclement: Generated {len(self.waypoints)} waypoints")
        print(
            f"  Direction: {'clockwise' if self.going_clockwise else 'counter-clockwise'}"
        )
        print(f"  Boundary: {'touches edge' if self.touches_boundary else 'enclosed'}")
        print(
            f"  Min fire distance: {self.min_fire_distance}, Max waypoint cost: {self.max_waypoint_cost}"
        )

        return True

    def _fire_touches_boundary(self, fire_grid: np.ndarray) -> bool:
        """Check if fire touches map boundary."""
        # Check edges
        if fire_grid[0, :].any():  # Top edge
            return True
        if fire_grid[-1, :].any():  # Bottom edge
            return True
        if fire_grid[:, 0].any():  # Left edge
            return True
        if fire_grid[:, -1].any():  # Right edge
            return True
        return False

    def _is_robot_passable(self, row: int, col: int, weight_grid: np.ndarray) -> bool:
        """
        Check if the robot can be centered at this cell.

        The robot has a footprint of robot_size x robot_size, so we need to check
        that all cells in that footprint are passable (finite cost).
        """
        offset = self.robot_offset

        # Check bounds - robot footprint must fit within grid
        if row < offset or row >= self.rows - offset:
            return False
        if col < offset or col >= self.cols - offset:
            return False

        # Check all cells in robot footprint
        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
                check_row = row + dr
                check_col = col + dc
                if not np.isfinite(weight_grid[check_row, check_col]):
                    return False

        return True

    def _find_passable_corridor(
        self,
        fire_distance: np.ndarray,
        weight_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Dynamically find the lowest-cost passable ring around fire.

        Strategy:
        1. For each angle around fire centroid, find the cell with minimum cost
        2. Filter to only include cells where robot can fit
        3. This traces the "valley" of the potential field

        Returns array of (row, col) positions forming the optimal ring.
        """
        if self.fire_centroid is None:
            return np.array([]).reshape(0, 2)

        cx, cy = self.fire_centroid

        # Sample angles around the fire (every 2 degrees = 180 samples)
        num_angles = 180
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        # For each angle, search outward from fire to find minimum cost cell
        ring_cells = []

        # Search parameters
        max_search_dist = max(self.rows, self.cols)

        for angle in angles:
            best_cell = None
            best_cost = float("inf")

            # Ray march outward from fire centroid
            for dist in np.arange(self.min_fire_distance, max_search_dist, 0.5):
                # Calculate cell position along ray
                x = cx + dist * math.cos(angle)
                y = cy + dist * math.sin(angle)

                col = int(x)
                row = int(y)

                # Skip if out of bounds
                if not (0 <= row < self.rows and 0 <= col < self.cols):
                    break  # Hit map edge, stop searching this ray

                # Skip if robot can't fit
                if not self._is_robot_passable(row, col, weight_grid):
                    continue

                cost = weight_grid[row, col]

                # Skip infinite cost
                if not np.isfinite(cost):
                    continue

                # Track minimum cost along this ray
                if cost < best_cost:
                    best_cost = cost
                    best_cell = (row, col)

                # If cost is increasing significantly, we've passed the valley
                # Stop searching to avoid going too far from fire
                if best_cell is not None and cost > best_cost + 5.0:
                    break

            if best_cell is not None:
                # Avoid duplicate cells
                if best_cell not in ring_cells:
                    ring_cells.append(best_cell)

        # Store for visualization
        self.corridor_cells = ring_cells.copy()

        print(f"  Corridor search: Found {len(ring_cells)} ring cells via ray marching")

        # Debug: show cost range of found cells
        if ring_cells:
            costs = [weight_grid[r, c] for r, c in ring_cells]
            print(
                f"    Ring cost range: min={min(costs):.1f}, max={max(costs):.1f}, avg={sum(costs) / len(costs):.1f}"
            )

        return np.array(ring_cells) if ring_cells else np.array([]).reshape(0, 2)

    def get_corridor_cells(self) -> list[tuple[int, int]]:
        """Get the corridor cells for visualization."""
        return self.corridor_cells

    def _order_contour_by_angle(
        self,
        contour_cells: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Order contour cells by angle around fire centroid."""
        if self.fire_centroid is None or len(contour_cells) == 0:
            return []

        cx, cy = self.fire_centroid

        # Calculate angle for each contour cell
        angles = np.arctan2(
            contour_cells[:, 0] + 0.5 - cy,  # row -> y
            contour_cells[:, 1] + 0.5 - cx,  # col -> x
        )

        # Sort by angle
        sorted_indices = np.argsort(angles)
        ordered = contour_cells[sorted_indices]

        # Convert to (x, y) format
        return [(float(c) + 0.5, float(r) + 0.5) for r, c in ordered]

    def _handle_boundary_gaps(
        self,
        contour: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Handle map boundary case by finding angular gaps.
        Reorders contour to start/end at boundary gaps.
        """
        if len(contour) < 3 or self.fire_centroid is None:
            return contour

        cx, cy = self.fire_centroid

        # Find largest angular gap between consecutive points
        angles = [math.atan2(y - cy, x - cx) for x, y in contour]

        max_gap = 0.0
        gap_idx = 0

        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i]
            if gap < 0:
                gap += 2 * math.pi

            if gap > max_gap:
                max_gap = gap
                gap_idx = next_i

        # If gap is significant (> 45 degrees), we have a boundary case
        if max_gap > math.pi / 4:
            # Reorder to start after the gap
            contour = contour[gap_idx:] + contour[:gap_idx]

        return contour

    def _find_start_and_direction(
        self,
        contour: list[tuple[float, float]],
        robot_x: float,
        robot_y: float,
    ) -> tuple[int, bool]:
        """
        Find the starting index (nearest to robot) and optimal direction.

        Returns:
            (start_index, going_clockwise)
        """
        if len(contour) == 0:
            return 0, True

        # Find nearest contour point to robot
        min_dist = float("inf")
        nearest_idx = 0

        for i, (x, y) in enumerate(contour):
            dx = x - robot_x
            dy = y - robot_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Determine direction - which way is shorter to complete the loop?
        # For simplicity, go in the direction of the next nearest point
        n = len(contour)
        if n < 3:
            return nearest_idx, True

        # Check distances to neighbors
        prev_idx = (nearest_idx - 1) % n
        next_idx = (nearest_idx + 1) % n

        prev_x, prev_y = contour[prev_idx]
        next_x, next_y = contour[next_idx]

        dist_to_prev = math.sqrt((prev_x - robot_x) ** 2 + (prev_y - robot_y) ** 2)
        dist_to_next = math.sqrt((next_x - robot_x) ** 2 + (next_y - robot_y) ** 2)

        # Go toward the closer neighbor (clockwise if next is closer)
        going_clockwise = dist_to_next <= dist_to_prev

        return nearest_idx, going_clockwise

    def _sample_waypoints_with_spacing(
        self,
        contour: list[tuple[float, float]],
        weight_grid: np.ndarray,
        fire_distance: np.ndarray,
    ) -> list[tuple[float, float]]:
        """
        Sample waypoints along contour with proper spacing and validation.

        Each waypoint must:
        1. Be at least min_waypoint_spacing from the previous waypoint
        2. Be at least min_fire_distance from fire
        3. Be in a robot-passable location
        4. Have cost below corridor_cost_threshold

        Waypoints that fail validation are dropped (not relocated).
        """
        if len(contour) < 2:
            return list(contour)

        waypoints = []
        accumulated_dist = 0.0

        for i, (x, y) in enumerate(contour):
            row, col = int(y), int(x)

            # Validate this position
            if not self._is_valid_waypoint_position(
                row, col, weight_grid, fire_distance
            ):
                continue

            # First waypoint - just add it
            if len(waypoints) == 0:
                waypoints.append((x, y))
                continue

            # Calculate distance from last waypoint
            prev_x, prev_y = waypoints[-1]
            dx = x - prev_x
            dy = y - prev_y
            dist = math.sqrt(dx * dx + dy * dy)

            # Only add if we've traveled enough distance
            if dist >= self.waypoint_spacing:
                waypoints.append((x, y))

        # Try to add last point if far enough from previous
        if len(contour) > 0 and len(waypoints) > 0:
            last_x, last_y = contour[-1]
            last_row, last_col = int(last_y), int(last_x)

            if self._is_valid_waypoint_position(
                last_row, last_col, weight_grid, fire_distance
            ):
                prev_x, prev_y = waypoints[-1]
                dx = last_x - prev_x
                dy = last_y - prev_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist >= self.min_waypoint_spacing:
                    waypoints.append((last_x, last_y))

        return waypoints

    def _is_valid_waypoint_position(
        self,
        row: int,
        col: int,
        weight_grid: np.ndarray,
        fire_distance: np.ndarray,
        max_cost: float = None,
    ) -> bool:
        """
        Check if a position is valid for a waypoint.

        Must be:
        1. Within bounds
        2. At least min_fire_distance from fire
        3. Robot-passable (3x3 footprint can fit)
        4. Cost below max_cost threshold

        Args:
            row, col: Cell position
            weight_grid: Cost grid
            fire_distance: Distance from fire grid
            max_cost: Maximum acceptable cost (uses self.max_waypoint_cost if None)
        """
        if max_cost is None:
            max_cost = self.max_waypoint_cost

        # Bounds check
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        # Fire distance check
        fd = fire_distance[row, col]
        if fd < self.min_fire_distance:
            return False

        # Robot passability check (3x3 footprint)
        if not self._is_robot_passable(row, col, weight_grid):
            return False

        # Cost check - must be finite AND below threshold
        cost = weight_grid[row, col]
        if not np.isfinite(cost):
            return False

        if cost > max_cost:
            return False

        return True

    def _validate_waypoints(
        self,
        waypoints: list[tuple[float, float]],
        weight_grid: np.ndarray,
        fire_distance: np.ndarray,
    ) -> list[tuple[float, float]]:
        """
        Validate waypoints and optimize each to lowest-cost nearby position.
        Then enforce minimum spacing between waypoints.

        Returns list of valid waypoints (optimized for lowest cost, properly spaced).
        """
        valid_waypoints = []

        for i, (wx, wy) in enumerate(waypoints):
            row, col = int(wy), int(wx)

            # Check if in bounds
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                continue

            # Find optimal position (lowest cost nearby)
            optimized = self._find_lowest_cost_position(
                wx,
                wy,
                weight_grid,
                fire_distance,
                valid_waypoints[-1] if valid_waypoints else None,
            )

            if optimized is not None:
                if optimized != (wx, wy):
                    print(
                        f"  Waypoint {i} optimized: ({wx:.1f}, {wy:.1f}) -> ({optimized[0]:.1f}, {optimized[1]:.1f})"
                    )
                valid_waypoints.append(optimized)
            else:
                print(
                    f"  Waypoint {i} skipped: ({wx:.1f}, {wy:.1f}) - no valid position found"
                )

        # Enforce minimum spacing between waypoints
        valid_waypoints = self._enforce_minimum_spacing(valid_waypoints)

        return valid_waypoints

    def _find_lowest_cost_position(
        self,
        wx: float,
        wy: float,
        weight_grid: np.ndarray,
        fire_distance: np.ndarray,
        prev_waypoint: Optional[tuple[float, float]],
        search_radius: int = None,
    ) -> Optional[tuple[float, float]]:
        """
        Find the lowest-cost position near the given waypoint.

        Searches nearby cells and returns the position with minimum cost,
        while respecting fire distance constraints, robot passability,
        cost thresholds, and not going backward.

        Returns optimized position or None if no valid position found.
        """
        if search_radius is None:
            search_radius = int(self.relocation_search_radius)

        best_pos = None
        best_cost = float("inf")

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nx = wx + dc
                ny = wy + dr

                row, col = int(ny), int(nx)

                # Use the unified validity check with cost threshold
                if not self._is_valid_waypoint_position(
                    row, col, weight_grid, fire_distance
                ):
                    continue

                cell_cost = weight_grid[row, col]

                # Skip if this would be going backward toward previous waypoint
                if prev_waypoint is not None:
                    dist_to_prev = math.sqrt(
                        (nx - prev_waypoint[0]) ** 2 + (ny - prev_waypoint[1]) ** 2
                    )
                    orig_to_prev = math.sqrt(
                        (wx - prev_waypoint[0]) ** 2 + (wy - prev_waypoint[1]) ** 2
                    )
                    # Don't relocate if it puts us too close to previous waypoint
                    if dist_to_prev < orig_to_prev * 0.5:
                        continue
                    # Also enforce minimum spacing from previous waypoint
                    if dist_to_prev < self.min_waypoint_spacing:
                        continue

                # Compute total score: actual cell cost + distance penalty
                dist_from_original = math.sqrt(dc * dc + dr * dr)

                # Prefer staying close to original position (small penalty)
                distance_penalty = dist_from_original * 0.1

                # The main factor is the actual cell cost from weight_grid
                total_score = cell_cost + distance_penalty

                if total_score < best_cost:
                    best_cost = total_score
                    best_pos = (nx, ny)

        # If we found a position, verify it's actually acceptable
        if best_pos is not None:
            row, col = int(best_pos[1]), int(best_pos[0])
            actual_cost = weight_grid[row, col]
            if actual_cost > self.max_waypoint_cost:
                # Even the best position is too costly - reject it
                print(
                    f"    Best position cost {actual_cost:.1f} exceeds threshold {self.max_waypoint_cost:.1f}"
                )
                return None

        return best_pos

    def _enforce_minimum_spacing(
        self,
        waypoints: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Remove waypoints that are too close to each other.
        Keeps first waypoint, then only adds subsequent waypoints if they're
        far enough from the last kept waypoint.

        Returns filtered list of waypoints with minimum spacing enforced.
        """
        if len(waypoints) <= 1:
            return waypoints

        filtered = [waypoints[0]]

        for i in range(1, len(waypoints)):
            wx, wy = waypoints[i]
            last_x, last_y = filtered[-1]

            dx = wx - last_x
            dy = wy - last_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist >= self.min_waypoint_spacing:
                filtered.append((wx, wy))
            else:
                # Skip this waypoint - too close
                pass

        # Special case: make sure we keep the last waypoint if it's the loop closure
        # and it's reasonably far from the second-to-last
        if len(waypoints) > 2 and len(filtered) > 1:
            last_original = waypoints[-1]
            last_filtered = filtered[-1]

            # Check if original last waypoint is different from filtered last
            if last_original != last_filtered:
                dx = last_original[0] - last_filtered[0]
                dy = last_original[1] - last_filtered[1]
                dist = math.sqrt(dx * dx + dy * dy)

                # Add it back if it's far enough (for loop closure)
                if dist >= self.min_waypoint_spacing * 0.7:
                    filtered.append(last_original)

        removed_count = len(waypoints) - len(filtered)
        if removed_count > 0:
            print(f"  Removed {removed_count} waypoints too close together")

        return filtered

    def get_current_waypoint(self) -> Optional[tuple[float, float]]:
        """Get the current target waypoint."""
        if self.state != EncirclementState.ACTIVE:
            return None

        if self.current_waypoint_idx >= len(self.waypoints):
            return None

        return self.waypoints[self.current_waypoint_idx]

    def get_current_waypoint_index(self) -> int:
        """Get the current waypoint index."""
        return self.current_waypoint_idx

    def waypoint_reached(self, robot_x: float, robot_y: float) -> bool:
        """Check if robot has reached current waypoint."""
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False

        dx = waypoint[0] - robot_x
        dy = waypoint[1] - robot_y
        return math.sqrt(dx * dx + dy * dy) < self.waypoint_reached_threshold

    def mark_current_visited(self):
        """Mark the current waypoint as visited."""
        if self.current_waypoint_idx < len(self.waypoints):
            self.visited_waypoints.add(self.current_waypoint_idx)

    def mark_current_skipped(self):
        """Mark the current waypoint as skipped."""
        if self.current_waypoint_idx < len(self.waypoints):
            self.skipped_waypoints.add(self.current_waypoint_idx)
            print(
                f"Encirclement: Marking waypoint {self.current_waypoint_idx} as skipped"
            )

    def get_unvisited_waypoints(self) -> list[tuple[int, tuple[float, float]]]:
        """Get list of (index, position) for all unvisited, non-skipped waypoints."""
        unvisited = []
        for i, wp in enumerate(self.waypoints):
            if i not in self.visited_waypoints and i not in self.skipped_waypoints:
                unvisited.append((i, wp))
        return unvisited

    def set_current_waypoint(self, idx: int):
        """Set the current waypoint index."""
        if 0 <= idx < len(self.waypoints):
            self.current_waypoint_idx = idx

    def direction_name(self) -> str:
        """Get human-readable direction name."""
        return "clockwise" if self.going_clockwise else "counterclockwise"

    def advance_waypoint(self) -> bool:
        """
        Mark current waypoint as visited and move to next in SEQUENCE.
        Returns True if there are more waypoints, False if complete.

        SIMPLIFIED: Just goes 0 -> 1 -> 2 -> ... -> n-1 -> back to 0
        """
        # Mark current as visited
        self.mark_current_visited()

        n = len(self.waypoints)

        # Check if we just completed the return to start
        if (
            self.returning_to_start
            and self.current_waypoint_idx == self.start_waypoint_idx
        ):
            coverage = len(self.visited_waypoints) / n
            self.state = EncirclementState.COMPLETE
            print(
                f"Encirclement COMPLETE: {coverage * 100:.0f}% coverage, loop closed!"
            )
            return False

        # Move to next waypoint sequentially
        next_idx = self.current_waypoint_idx + 1

        # Check if we've gone through all waypoints
        if next_idx >= n:
            # Time to return to start
            self.returning_to_start = True
            self.current_waypoint_idx = self.start_waypoint_idx
            print(
                f"Returning to start waypoint {self.start_waypoint_idx} to close loop"
            )
            return True

        # Skip any already visited or skipped waypoints
        while next_idx < n:
            if (
                next_idx not in self.visited_waypoints
                and next_idx not in self.skipped_waypoints
            ):
                break
            next_idx += 1

        # If we exhausted all waypoints, return to start
        if next_idx >= n:
            self.returning_to_start = True
            self.current_waypoint_idx = self.start_waypoint_idx
            print(
                f"All waypoints visited, returning to start {self.start_waypoint_idx}"
            )
            return True

        self.current_waypoint_idx = next_idx
        return True

    def check_completion(self) -> bool:
        """
        Check if encirclement is complete or failed.
        Returns True if still active, False if complete or failed.
        """
        return self.state == EncirclementState.ACTIVE

    def skip_current_waypoint(self) -> bool:
        """
        Skip the current waypoint and move to next.
        Returns True if there are more waypoints, False if done.
        """
        self.mark_current_skipped()

        n = len(self.waypoints)
        next_idx = self.current_waypoint_idx + 1

        # Skip any visited/skipped
        while next_idx < n:
            if (
                next_idx not in self.visited_waypoints
                and next_idx not in self.skipped_waypoints
            ):
                break
            next_idx += 1

        if next_idx >= n:
            # Return to start
            self.returning_to_start = True
            self.current_waypoint_idx = self.start_waypoint_idx
            return True

        self.current_waypoint_idx = next_idx
        return True

    def update_waypoints_for_new_obstacles(
        self,
        weight_grid: np.ndarray,
        fire_distance: np.ndarray,
    ) -> tuple[bool, bool]:
        """
        Re-optimize all future waypoints based on updated cost map.
        Called when new obstacles are discovered or fire spreads.

        Each waypoint is moved to the lowest-cost position nearby.
        Also removes waypoints that end up too close together.

        Returns:
            Tuple of (success, current_waypoint_changed):
            - success: True if all remaining waypoints are valid (possibly relocated)
            - current_waypoint_changed: True if the current target waypoint was relocated
        """
        relocated_count = 0
        skipped_count = 0
        current_waypoint_changed = False

        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            if i in self.skipped_waypoints:
                continue

            wx, wy = self.waypoints[i]
            prev_wp = self.waypoints[i - 1] if i > 0 else None

            # First check: is the current position still valid?
            current_row, current_col = int(wy), int(wx)
            current_valid = self._is_valid_waypoint_position(
                current_row, current_col, weight_grid, fire_distance
            )

            if current_valid:
                current_cost = weight_grid[current_row, current_col]
            else:
                current_cost = float("inf")

            # If current position is invalid or too costly, must find better spot
            needs_relocation = (
                not current_valid or current_cost > self.max_waypoint_cost
            )

            if needs_relocation:
                print(
                    f"  Waypoint {i} at ({wx:.1f}, {wy:.1f}) needs relocation (cost={current_cost:.1f}, valid={current_valid})"
                )

            # Find the best (lowest cost) position for this waypoint
            optimized = self._find_lowest_cost_position(
                wx, wy, weight_grid, fire_distance, prev_wp
            )

            if optimized is not None:
                if optimized != (wx, wy):
                    new_row, new_col = int(optimized[1]), int(optimized[0])
                    new_cost = weight_grid[new_row, new_col]

                    # Only report if there's a meaningful change
                    if needs_relocation or new_cost < current_cost - 0.5:
                        print(
                            f"  Waypoint {i} relocated: ({wx:.1f}, {wy:.1f}) -> ({optimized[0]:.1f}, {optimized[1]:.1f}), cost {current_cost:.1f} -> {new_cost:.1f}"
                        )
                        relocated_count += 1

                        # Track if current waypoint changed
                        if i == self.current_waypoint_idx:
                            current_waypoint_changed = True

                    self.waypoints[i] = optimized
            else:
                # No valid position found - skip this waypoint
                self.skipped_waypoints.add(i)
                print(
                    f"  Waypoint {i} now unreachable (no valid position in search radius), skipping"
                )
                skipped_count += 1

                # If current waypoint was skipped, that's also a change
                if i == self.current_waypoint_idx:
                    current_waypoint_changed = True

        # Check for waypoints that are now too close together and mark for skipping
        spacing_skipped = 0
        for i in range(self.current_waypoint_idx + 1, len(self.waypoints)):
            if i in self.skipped_waypoints:
                continue

            # Find previous non-skipped waypoint
            prev_idx = i - 1
            while (
                prev_idx >= self.current_waypoint_idx
                and prev_idx in self.skipped_waypoints
            ):
                prev_idx -= 1

            if prev_idx < self.current_waypoint_idx:
                continue

            wx, wy = self.waypoints[i]
            prev_x, prev_y = self.waypoints[prev_idx]

            dx = wx - prev_x
            dy = wy - prev_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.min_waypoint_spacing:
                self.skipped_waypoints.add(i)
                spacing_skipped += 1

        if spacing_skipped > 0:
            print(f"  Removed {spacing_skipped} waypoints too close together")
            skipped_count += spacing_skipped

        if relocated_count > 0 or skipped_count > 0:
            print(
                f"Encirclement update: {relocated_count} optimized, {skipped_count} skipped"
            )

        return True, current_waypoint_changed

    def regenerate_for_fire_spread(
        self,
        fire_grid: np.ndarray,
        fire_distance: np.ndarray,
        weight_grid: np.ndarray,
        robot_x: float,
        robot_y: float,
    ) -> bool:
        """
        Regenerate waypoints when fire has spread significantly.
        Tries to preserve progress by starting from current position.

        Returns:
            True if regeneration successful
        """
        # Remember how many waypoints we've visited
        progress = len(self.visited_waypoints)

        # Regenerate waypoints
        success = self.generate_waypoints(
            fire_grid,
            fire_distance,
            weight_grid,
            robot_x,
            robot_y,
        )

        if success:
            print(
                f"Encirclement: Regenerated waypoints (was at {progress}/{len(self.waypoints)})"
            )

        return success

    def get_waypoints(self) -> list[tuple[float, float]]:
        """Get all waypoints."""
        return self.waypoints.copy()

    def get_waypoint_states(self) -> list[str]:
        """
        Get state of each waypoint for visualization.

        Returns:
            List of states: "visited", "current", "pending", "skipped"
        """
        states = []
        for i in range(len(self.waypoints)):
            if i in self.visited_waypoints:
                states.append("visited")
            elif i == self.current_waypoint_idx:
                states.append("current")
            elif i in self.skipped_waypoints:
                states.append("skipped")
            else:
                states.append("pending")
        return states

    def is_complete(self) -> bool:
        """Check if encirclement is complete."""
        return self.state == EncirclementState.COMPLETE

    def is_active(self) -> bool:
        """Check if encirclement is in progress."""
        return self.state == EncirclementState.ACTIVE

    def is_failed(self) -> bool:
        """Check if encirclement has failed."""
        return self.state == EncirclementState.FAILED

    def reset(self):
        """Reset the planner to idle state."""
        self.state = EncirclementState.IDLE
        self.waypoints.clear()
        self.current_waypoint_idx = 0
        self.start_waypoint_idx = 0
        self.skipped_waypoints.clear()
        self.visited_waypoints.clear()
        self.returning_to_start = False
        self.fire_centroid = None
