# D* Lite Path Planner with String-Pulling
# RBE 550, Firebots (course project)
#
# This is the user's original working D* Lite with minimal additions:
# - String-pulling for smoother paths
# - Output in (x, y) format for DWA compatibility

import heapq
import math
import numpy as np


class FieldDStar:
    """
    D* Lite incremental path planner.
    Plans from goal to start so replanning is efficient when robot moves forward.

    (Renamed from DStarLite for compatibility with main.py imports)
    """

    def __init__(self, rows: int, cols: int, robot_size: int = 3):
        self.rows = rows
        self.cols = cols
        self.robot_size = robot_size
        self.robot_offset = robot_size // 2  # 1 for 3x3 robot

        # 8-connected neighbors: (dr, dc, cost)
        self.neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.41421356237), (-1, 1, 1.41421356237),
            (1, -1, 1.41421356237), (1, 1, 1.41421356237),
        )

        # Precompute footprint offsets once
        offset = self.robot_offset
        self.footprint_offsets = [
            (dr, dc)
            for dr in range(-offset, offset + 1)
            for dc in range(-offset, offset + 1)
        ]

        # Cache for passability: -1 unknown, 0 no, 1 yes
        self.passable_cache = np.full((rows, cols), -1, dtype=np.int8)

        # Heuristic constant
        self.SQRT2 = math.sqrt(2.0)

        self.reset()

    def reset(self):
        """Clear all planning data."""
        self.g = np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        self.rhs = np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        self.open_list = []
        self.open_set = set()
        self.km = 0.0
        self.start = None
        self.goal = None
        self.cost_grid = None
        # Reset cache
        self.passable_cache.fill(-1)

    def _heuristic(self, r1, c1, r2, c2):
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return max(dr, dc) + (self.SQRT2 - 1.0) * min(dr, dc)

    def _key(self, row, col):
        """Calculate priority key for a cell."""
        g_val = self.g[row, col]
        rhs_val = self.rhs[row, col]
        min_val = min(g_val, rhs_val)

        if self.start:
            h = self._heuristic(row, col, self.start[0], self.start[1])
        else:
            h = 0.0

        return (min_val + h + self.km, min_val)

    def _is_passable(self, row, col) -> bool:
        """Check if robot's footprint fits at this cell (cached)."""
        # Bounds check
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False

        # Check cache first
        cached = self.passable_cache[row, col]
        if cached != -1:
            return bool(cached)

        offset = self.robot_offset

        # Quick bounds check for footprint
        if row < offset or row >= self.rows - offset or col < offset or col >= self.cols - offset:
            self.passable_cache[row, col] = 0
            return False

        # Check all cells in footprint
        for dr, dc in self.footprint_offsets:
            rr = row + dr
            cc = col + dc
            if not np.isfinite(self.cost_grid[rr, cc]):
                self.passable_cache[row, col] = 0
                return False

        self.passable_cache[row, col] = 1
        return True

    def _edge_cost(self, r1, c1, r2, c2, move_cost):
        """Get cost of moving between two cells. Assumes both are passable."""
        # Average cell cost times movement cost
        cell_cost = (self.cost_grid[r1, c1] + self.cost_grid[r2, c2]) * 0.5
        return move_cost * (1.0 + cell_cost * 0.5)

    def _update_vertex(self, row, col):
        """Update a cell's rhs and open list status."""
        if (row, col) == self.goal:
            return

        if not self._is_passable(row, col):
            self.rhs[row, col] = np.inf
        else:
            min_rhs = np.inf
            for dr, dc, move_cost in self.neighbors:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self._is_passable(nr, nc):
                        total = self.g[nr, nc] + self._edge_cost(row, col, nr, nc, move_cost)
                        if total < min_rhs:
                            min_rhs = total
            self.rhs[row, col] = min_rhs

        # Remove from open list if present
        self.open_set.discard((row, col))

        # Add back if inconsistent
        if self.g[row, col] != self.rhs[row, col]:
            key = self._key(row, col)
            heapq.heappush(self.open_list, (key, (row, col)))
            self.open_set.add((row, col))

    def initialize(self, start: tuple, goal: tuple, cost_grid: np.ndarray,
                   obstacle_grid: np.ndarray = None):
        self.reset()
        self.start = start
        self.goal = goal

        if obstacle_grid is not None and obstacle_grid.any():
            self.cost_grid = cost_grid.copy()
            self.cost_grid[obstacle_grid] = np.inf
        else:
            self.cost_grid = cost_grid

        # invalidate passability cache
        self.passable_cache.fill(-1)

        self.rhs[goal[0], goal[1]] = 0.0
        key = self._key(goal[0], goal[1])
        heapq.heappush(self.open_list, (key, goal))
        self.open_set.add(goal)

    def compute_shortest_path(self, max_iterations: int = 100000):
        """Expand cells until start is consistent."""
        iterations = 0
        sr, sc = self.start

        while self.open_list and iterations < max_iterations:
            iterations += 1

            # Check termination
            start_key = self._key(sr, sc)
            top_key = self.open_list[0][0]

            if top_key >= start_key and self.g[sr, sc] == self.rhs[sr, sc]:
                break

            # Pop best cell
            _, (row, col) = heapq.heappop(self.open_list)

            # Skip if not in open set (stale entry)
            if (row, col) not in self.open_set:
                continue
            self.open_set.discard((row, col))

            g_val = self.g[row, col]
            rhs_val = self.rhs[row, col]

            if g_val > rhs_val:
                # Overconsistent - lower g
                self.g[row, col] = rhs_val
                for dr, dc, _ in self.neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self._update_vertex(nr, nc)
            else:
                # Underconsistent - raise g
                self.g[row, col] = np.inf
                self._update_vertex(row, col)
                for dr, dc, _ in self.neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self._update_vertex(nr, nc)

        path_exists = np.isfinite(self.g[sr, sc])
        print(f"D* Lite: {iterations} iterations, path={'found' if path_exists else 'NOT FOUND'}")
        return path_exists

    def _extract_grid_path(self) -> list[tuple[int, int]]:
        """Extract path from start to goal by following best neighbors."""
        if self.start is None or self.goal is None:
            return []

        if not np.isfinite(self.g[self.start[0], self.start[1]]):
            return []

        path = [self.start]
        current = self.start
        max_steps = self.rows * self.cols
        visited = {self.start}

        for _ in range(max_steps):
            if current == self.goal:
                break

            row, col = current
            best_next = None
            best_cost = np.inf

            for dr, dc, move_cost in self.neighbors:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if (nr, nc) in visited:
                    continue
                if not self._is_passable(nr, nc):
                    continue

                total = self.g[nr, nc] + self._edge_cost(row, col, nr, nc, move_cost)
                if total < best_cost:
                    best_cost = total
                    best_next = (nr, nc)

            if best_next is None:
                break

            path.append(best_next)
            visited.add(best_next)
            current = best_next

        return path

    # =========================================================================
    # STRING-PULLING ADDITIONS
    # =========================================================================

    def _line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        Check line of sight between two points (x, y format).
        Samples along the line and checks passability at each point.
        """
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.5:
            return True

        # Sample every ~0.5 cells
        steps = max(2, int(dist * 2))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * dx
            y = y1 + t * dy

            # Convert to row, col and check passability
            row, col = int(y), int(x)
            if not self._is_passable(row, col):
                return False

        return True

    def _string_pull(self, grid_path: list[tuple[int, int]]) -> list[tuple[float, float]]:
        """
        Convert grid path (row, col) to smooth path (x, y) via string-pulling.

        Connects furthest visible points directly, skipping intermediate waypoints.
        """
        if not grid_path:
            return []

        if len(grid_path) == 1:
            r, c = grid_path[0]
            return [(c + 0.5, r + 0.5)]

        # Convert to (x, y) at cell centers
        # grid_path is (row, col), we need (x, y) = (col, row)
        continuous = [(c + 0.5, r + 0.5) for r, c in grid_path]

        # String pulling
        result = [continuous[0]]
        current = 0

        while current < len(continuous) - 1:
            # Find furthest visible point
            best = current + 1

            # Limit lookahead for performance
            max_lookahead = min(len(continuous), current + 40)

            for target in range(current + 2, max_lookahead):
                if self._line_of_sight(
                        continuous[current][0], continuous[current][1],
                        continuous[target][0], continuous[target][1]
                ):
                    best = target

            result.append(continuous[best])
            current = best

        return result

    def extract_path(self) -> list[tuple[float, float]]:
        """
        Extract smooth path as (x, y) coordinates.

        Uses string-pulling to reduce waypoints and create any-angle paths.
        """
        # Get grid path (row, col)
        grid_path = self._extract_grid_path()
        if not grid_path:
            return []

        # String-pull to smooth (x, y) path
        smooth_path = self._string_pull(grid_path)

        return smooth_path

    # =========================================================================
    # ORIGINAL METHODS (kept for compatibility)
    # =========================================================================

    def update_start(self, new_start: tuple):
        """Call when robot moves."""
        if self.start:
            self.km += self._heuristic(self.start[0], self.start[1],
                                       new_start[0], new_start[1])
        self.start = new_start

    def update_costs(self, changed_cells: list[tuple[int, int]],
                     new_cost_grid: np.ndarray,
                     new_obstacle_grid: np.ndarray = None):
        """Update cost grid when obstacles/terrain change."""

        # Update cost grid
        if new_obstacle_grid is not None and new_obstacle_grid.any():
            self.cost_grid = new_cost_grid.copy()
            self.cost_grid[new_obstacle_grid] = np.inf
        else:
            self.cost_grid = new_cost_grid

        # Invalidate passability cache (most important performance fix)
        self.passable_cache.fill(-1)

        # Update affected cells and their neighbors
        for row, col in changed_cells:
            self._update_vertex(row, col)
            for dr, dc, _ in self.neighbors:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    self._update_vertex(nr, nc)

    def get_next_waypoint(self, path: list, current_index: int,
                          robot_x: float, robot_y: float,
                          lookahead: float = 3.0) -> int:
        """Find waypoint along path at approximately lookahead distance."""
        if current_index >= len(path):
            return current_index

        # Advance past waypoints we've passed
        while current_index < len(path) - 1:
            row, col = path[current_index]
            dx = col - robot_x
            dy = row - robot_y
            if dx * dx + dy * dy < 1.0:
                current_index += 1
            else:
                break

        # Find waypoint at lookahead distance
        best_index = current_index
        lookahead_sq = lookahead * lookahead

        for i in range(current_index, min(current_index + 20, len(path))):
            row, col = path[i]
            dx = col - robot_x
            dy = row - robot_y
            if dx * dx + dy * dy <= lookahead_sq:
                best_index = i
            else:
                break

        return best_index