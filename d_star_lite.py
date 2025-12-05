# D* Lite Path Planner
# RBE 550, Firebots (course project)

import heapq
import math
import numpy as np


class DStarLite:
    """
    D* Lite incremental path planner.
    Plans from goal to start so replanning is efficient when robot moves forward.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

        # 8-connected neighbors: (dr, dc, cost)
        self.neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        self.reset()

    def reset(self):
        """Clear all planning data."""
        self.g = np.full((self.rows, self.cols), np.inf)
        self.rhs = np.full((self.rows, self.cols), np.inf)
        self.open_list = []
        self.open_set = set()
        self.km = 0.0
        self.start = None
        self.goal = None
        self.cost_grid = None

    def _heuristic(self, r1, c1, r2, c2):
        """Euclidean distance heuristic."""
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    def _key(self, row, col):
        """Calculate priority key for a cell."""
        g_val = self.g[row, col]
        rhs_val = self.rhs[row, col]
        min_val = min(g_val, rhs_val)

        if self.start:
            h = self._heuristic(row, col, self.start[0], self.start[1])
        else:
            h = 0

        return (min_val + h + self.km, min_val)

    def _is_valid(self, row, col):
        """Check if cell is in bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_passable(self, row, col, obstacle_grid=None):
        """Check if robot's 3x3 footprint fits at this cell."""
        if obstacle_grid is None:
            return np.isfinite(self.cost_grid[row, col])

        # Check 3x3 footprint
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    return False
                if obstacle_grid[nr, nc]:
                    return False
                if not np.isfinite(self.cost_grid[nr, nc]):
                    return False
        return True

    def _cost(self, r1, c1, r2, c2, move_cost):
        """Get cost of moving between two cells."""
        # Check footprint collision
        if not self._is_passable(r1, c1, self.obstacle_grid):
            return np.inf
        if not self._is_passable(r2, c2, self.obstacle_grid):
            return np.inf

        # Average cell cost times movement cost
        cell_cost = (self.cost_grid[r1, c1] + self.cost_grid[r2, c2]) / 2.0
        return move_cost * (1.0 + cell_cost * 0.5)

    def _update_vertex(self, row, col):
        """Update a cell's rhs and open list status."""
        if (row, col) != self.goal:
            # rhs = min cost to reach goal through any neighbor
            min_rhs = np.inf
            for dr, dc, move_cost in self.neighbors:
                nr, nc = row + dr, col + dc
                if self._is_valid(nr, nc):
                    cost = self._cost(row, col, nr, nc, move_cost)
                    min_rhs = min(min_rhs, self.g[nr, nc] + cost)
            self.rhs[row, col] = min_rhs

        # Remove from open list if present
        if (row, col) in self.open_set:
            self.open_set.discard((row, col))

        # Add back if inconsistent
        if self.g[row, col] != self.rhs[row, col]:
            key = self._key(row, col)
            heapq.heappush(self.open_list, (key, (row, col)))
            self.open_set.add((row, col))

    def initialize(self, start: tuple, goal: tuple, cost_grid: np.ndarray, obstacle_grid: np.ndarray = None):
        """Initialize planner."""
        self.reset()
        self.start = start
        self.goal = goal
        self.cost_grid = cost_grid
        self.obstacle_grid = obstacle_grid  # Store it

        # Goal has rhs = 0
        self.rhs[goal[0], goal[1]] = 0.0

        key = self._key(goal[0], goal[1])
        heapq.heappush(self.open_list, (key, goal))
        self.open_set.add(goal)

    def compute_shortest_path(self, max_iterations: int = 100000):
        """Expand cells until start is consistent."""
        iterations = 0

        while self.open_list and iterations < max_iterations:
            iterations += 1

            # Check termination
            start_key = self._key(self.start[0], self.start[1])
            top_key = self.open_list[0][0]

            start_consistent = (self.g[self.start[0], self.start[1]] ==
                                self.rhs[self.start[0], self.start[1]])

            if top_key >= start_key and start_consistent:
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
                    if self._is_valid(nr, nc):
                        self._update_vertex(nr, nc)
            else:
                # Underconsistent - raise g
                self.g[row, col] = np.inf
                self._update_vertex(row, col)
                for dr, dc, _ in self.neighbors:
                    nr, nc = row + dr, col + dc
                    if self._is_valid(nr, nc):
                        self._update_vertex(nr, nc)

        print(f"D* Lite: {iterations} iterations, path exists: {np.isfinite(self.g[self.start[0], self.start[1]])}")
        return iterations < max_iterations

    def extract_path(self) -> list[tuple[int, int]]:
        """Extract path from start to goal by following best neighbors."""
        if self.start is None or self.goal is None:
            return []

        if not np.isfinite(self.g[self.start[0], self.start[1]]):
            return []

        path = [self.start]
        current = self.start
        max_steps = self.rows * self.cols

        for _ in range(max_steps):
            if current == self.goal:
                break

            row, col = current
            best_next = None
            best_cost = np.inf

            for dr, dc, move_cost in self.neighbors:
                nr, nc = row + dr, col + dc
                if self._is_valid(nr, nc):
                    cost = self._cost(row, col, nr, nc, move_cost)
                    total = self.g[nr, nc] + cost
                    if total < best_cost:
                        best_cost = total
                        best_next = (nr, nc)

            if best_next is None or best_next in path:
                break

            path.append(best_next)
            current = best_next

        return path

    def update_start(self, new_start: tuple):
        """Call when robot moves."""
        if self.start:
            self.km += self._heuristic(self.start[0], self.start[1],
                                       new_start[0], new_start[1])
        self.start = new_start

    def update_costs(self, changed_cells: list[tuple[int, int]], new_cost_grid: np.ndarray):
        """Call when obstacles are discovered."""
        self.cost_grid = new_cost_grid

        for row, col in changed_cells:
            self._update_vertex(row, col)
            for dr, dc, _ in self.neighbors:
                nr, nc = row + dr, col + dc
                if self._is_valid(nr, nc):
                    self._update_vertex(nr, nc)

    def get_next_waypoint(self, path: list, current_index: int, robot_x: float, robot_y: float,
                          lookahead: float = 3.0) -> int:
        """
        Find waypoint along path that is approximately lookahead distance away.
        Skips waypoints that are behind or too close to the robot.
        """
        if current_index >= len(path):
            return current_index

        # First, advance past any waypoints we've already passed
        while current_index < len(path) - 1:
            row, col = path[current_index]
            dx = col - robot_x
            dy = row - robot_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1.0:  # Close enough to skip
                current_index += 1
            else:
                break

        # Now find a waypoint at approximately lookahead distance
        best_index = current_index
        for i in range(current_index, min(current_index + 20, len(path))):
            row, col = path[i]
            dx = col - robot_x
            dy = row - robot_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= lookahead:
                best_index = i
            else:
                break

        return best_index