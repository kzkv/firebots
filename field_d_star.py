# Field D* Path Planner - Optimized
# RBE 550, Firebots (course project)
#
# Hybrid approach:
# 1. D* Lite with caching for g-values
# 2. Gradient descent with smoothing for any-angle path extraction

import heapq
import math

import numpy as np


class FieldDStar:
    """
    Optimized Field D* with passability caching and smooth path extraction.
    """

    def __init__(self, rows: int, cols: int, robot_size: int = 3):
        self.rows = rows
        self.cols = cols
        self.robot_size = robot_size
        self.robot_offset = robot_size // 2

        # 8-connected neighbors: (dr, dc, cost)
        self.neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.41421356237),
            (-1, 1, 1.41421356237),
            (1, -1, 1.41421356237),
            (1, 1, 1.41421356237),
        )

        # Precompute footprint offsets once
        offset = self.robot_offset
        self.footprint_offsets = tuple(
            (dr, dc)
            for dr in range(-offset, offset + 1)
            for dc in range(-offset, offset + 1)
        )

        # Cache for passability: -1 unknown, 0 no, 1 yes
        self.passable_cache = np.full((rows, cols), -1, dtype=np.int8)

        # Precompute constants
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
        self.passable_cache.fill(-1)

    def _heuristic(self, r1, c1, r2, c2):
        """Octile distance heuristic."""
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
        cached = self.passable_cache[row, col]
        if cached != -1:
            return bool(cached)

        offset = self.robot_offset

        if (
            row < offset
            or row >= self.rows - offset
            or col < offset
            or col >= self.cols - offset
        ):
            self.passable_cache[row, col] = 0
            return False

        for dr, dc in self.footprint_offsets:
            if not np.isfinite(self.cost_grid[row + dr, col + dc]):
                self.passable_cache[row, col] = 0
                return False

        self.passable_cache[row, col] = 1
        return True

    def _edge_cost(self, r1, c1, r2, c2, move_cost):
        """Get cost of moving between two cells."""
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
                        total = self.g[nr, nc] + self._edge_cost(
                            row, col, nr, nc, move_cost
                        )
                        if total < min_rhs:
                            min_rhs = total
            self.rhs[row, col] = min_rhs

        self.open_set.discard((row, col))

        if self.g[row, col] != self.rhs[row, col]:
            key = self._key(row, col)
            heapq.heappush(self.open_list, (key, (row, col)))
            self.open_set.add((row, col))

    def initialize(
        self,
        start: tuple,
        goal: tuple,
        cost_grid: np.ndarray,
        obstacle_grid: np.ndarray = None,
    ):
        """Initialize planner."""
        self.reset()
        self.start = start
        self.goal = goal

        if obstacle_grid is not None and obstacle_grid.any():
            self.cost_grid = cost_grid.copy()
            self.cost_grid[obstacle_grid] = np.inf
        else:
            self.cost_grid = cost_grid

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

            start_key = self._key(sr, sc)
            top_key = self.open_list[0][0]

            if top_key >= start_key and self.g[sr, sc] == self.rhs[sr, sc]:
                break

            _, (row, col) = heapq.heappop(self.open_list)

            if (row, col) not in self.open_set:
                continue
            self.open_set.discard((row, col))

            g_val = self.g[row, col]
            rhs_val = self.rhs[row, col]

            if g_val > rhs_val:
                self.g[row, col] = rhs_val
                for dr, dc, _ in self.neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self._update_vertex(nr, nc)
            else:
                self.g[row, col] = np.inf
                self._update_vertex(row, col)
                for dr, dc, _ in self.neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self._update_vertex(nr, nc)

        path_exists = np.isfinite(self.g[sr, sc])
        print(
            f"Field D*: {iterations} iterations, path={'found' if path_exists else 'NOT FOUND'}"
        )
        return path_exists

    def _get_g_safe(self, row, col) -> float:
        """Get g-value with bounds checking, returns large value if out of bounds."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return 1e9
        g = self.g[row, col]
        return min(g, 1e9) if np.isfinite(g) else 1e9

    def _compute_gradient_at_cell(self, row: int, col: int) -> tuple:
        """
        Compute gradient direction at cell center using neighboring g-values.
        Returns normalized (dx, dy) pointing toward goal (decreasing g).
        """
        # Get g-values of neighbors
        g_up = self._get_g_safe(row - 1, col)
        g_down = self._get_g_safe(row + 1, col)
        g_left = self._get_g_safe(row, col - 1)
        g_right = self._get_g_safe(row, col + 1)

        # Central differences - gradient points toward increasing g
        # We want to go toward decreasing g (toward goal)
        grad_x = (g_left - g_right) / 2.0  # Positive = go right (toward lower g)
        grad_y = (g_up - g_down) / 2.0  # Positive = go down (toward lower g)

        # Normalize
        mag = math.sqrt(grad_x * grad_x + grad_y * grad_y)
        if mag > 1e-6:
            grad_x /= mag
            grad_y /= mag
        else:
            grad_x, grad_y = 0.0, 0.0

        return grad_x, grad_y

    def extract_path(self) -> list[tuple[float, float]]:
        """
        Extract smooth any-angle path using gradient descent on g-field.
        Returns list of (x, y) coordinates.

        Uses larger steps and line-of-sight optimization to reduce oscillation.
        """
        if self.start is None or self.goal is None:
            return []

        sr, sc = self.start
        if not np.isfinite(self.g[sr, sc]):
            return []

        # Start at cell center
        x, y = float(sc) + 0.5, float(sr) + 0.5
        goal_x, goal_y = float(self.goal[1]) + 0.5, float(self.goal[0]) + 0.5

        path = [(x, y)]
        step_size = 1.0  # Larger steps = smoother path
        max_steps = (self.rows + self.cols) * 2

        prev_grad = (0.0, 0.0)
        momentum = 0.3  # Blend with previous gradient to reduce oscillation

        for step in range(max_steps):
            # Check if reached goal
            dx = goal_x - x
            dy = goal_y - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1.0:
                path.append((goal_x, goal_y))
                break

            # Get current cell
            col = int(x)
            row = int(y)
            col = max(0, min(col, self.cols - 1))
            row = max(0, min(row, self.rows - 1))

            # Compute gradient at current cell
            grad_x, grad_y = self._compute_gradient_at_cell(row, col)

            # If gradient is zero, head directly toward goal
            if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
                if dist > 0.01:
                    grad_x = dx / dist
                    grad_y = dy / dist
                else:
                    break

            # Apply momentum to reduce oscillation
            grad_x = momentum * prev_grad[0] + (1 - momentum) * grad_x
            grad_y = momentum * prev_grad[1] + (1 - momentum) * grad_y

            # Re-normalize after momentum
            mag = math.sqrt(grad_x * grad_x + grad_y * grad_y)
            if mag > 1e-6:
                grad_x /= mag
                grad_y /= mag

            prev_grad = (grad_x, grad_y)

            # Step in gradient direction
            new_x = x + grad_x * step_size
            new_y = y + grad_y * step_size

            # Clamp to grid
            new_x = max(1.0, min(new_x, self.cols - 1.0))
            new_y = max(1.0, min(new_y, self.rows - 1.0))

            # Check if new position is passable
            new_row = int(new_y)
            new_col = int(new_x)
            if not self._is_passable(new_row, new_col):
                # Try smaller step
                new_x = x + grad_x * step_size * 0.5
                new_y = y + grad_y * step_size * 0.5
                new_row = int(new_y)
                new_col = int(new_x)
                if not self._is_passable(new_row, new_col):
                    break  # Stuck

            x, y = new_x, new_y
            path.append((x, y))

        # Smooth the path with a simple moving average
        if len(path) > 3:
            path = self._smooth_path(path)

        return path

    def _smooth_path(
        self, path: list, iterations: int = 2, weight: float = 0.3
    ) -> list:
        """
        Smooth path using weighted average with neighbors.
        Keeps first and last points fixed.
        """
        if len(path) <= 2:
            return path

        smoothed = list(path)

        for _ in range(iterations):
            new_path = [smoothed[0]]  # Keep start
            for i in range(1, len(smoothed) - 1):
                prev_x, prev_y = smoothed[i - 1]
                curr_x, curr_y = smoothed[i]
                next_x, next_y = smoothed[i + 1]

                # Weighted average
                new_x = curr_x + weight * ((prev_x + next_x) / 2 - curr_x)
                new_y = curr_y + weight * ((prev_y + next_y) / 2 - curr_y)

                # Check passability
                row, col = int(new_y), int(new_x)
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    if self._is_passable(row, col):
                        new_path.append((new_x, new_y))
                    else:
                        new_path.append((curr_x, curr_y))
                else:
                    new_path.append((curr_x, curr_y))

            new_path.append(smoothed[-1])  # Keep end
            smoothed = new_path

        return smoothed

    def extract_path_cell_coords(self) -> list[tuple[int, int]]:
        """Extract path as cell coordinates (row, col) for compatibility."""
        smooth_path = self.extract_path()
        if not smooth_path:
            return []

        cell_path = []
        for x, y in smooth_path:
            row, col = int(y), int(x)
            row = max(0, min(row, self.rows - 1))
            col = max(0, min(col, self.cols - 1))
            if not cell_path or cell_path[-1] != (row, col):
                cell_path.append((row, col))

        return cell_path

    def update_start(self, new_start: tuple):
        """Update start position for incremental replanning."""
        if self.start:
            self.km += self._heuristic(
                self.start[0], self.start[1], new_start[0], new_start[1]
            )
        self.start = new_start
