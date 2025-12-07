# Fire Spread Simulation
# RBE 550, Firebots (course project)

import math

import numpy as np


class FireSpread:
    """Simulates fire propagation with fireline blocking."""

    def __init__(
        self,
        fire_grid: np.ndarray,
        spread_pace: float = 1.0,
    ):
        self.fire_grid = fire_grid
        self.rows, self.cols = fire_grid.shape
        self.fireline_grid = np.zeros((self.rows, self.cols), dtype=bool)
        self.spread_pace = spread_pace
        self.accumulator = 0.0

    def update(self, dt: float) -> bool:
        """
        Advance fire spread simulation.
        Returns True if fire expanded this frame.
        """
        self.accumulator += dt
        expanded = False

        while self.accumulator >= self.spread_pace:
            self.accumulator -= self.spread_pace
            if self._ignite_one_neighbor():
                expanded = True

        return expanded

    def _ignite_one_neighbor(self) -> bool:
        """Ignite one random valid neighbor of the fire. Returns True if successful."""
        candidates = self._find_spread_candidates()
        if not candidates:
            return False

        # Pick a random candidate
        idx = np.random.randint(len(candidates))
        row, col = candidates[idx]
        self.fire_grid[row, col] = True
        return True

    def _find_spread_candidates(self) -> list[tuple[int, int]]:
        """Find all cells that could be ignited (4-connected neighbors only, no diagonals)."""
        candidates = []
        fire_positions = np.argwhere(self.fire_grid)

        # 4-connected neighbors only (no diagonals) to prevent leaking through gaps
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for fire_row, fire_col in fire_positions:
            for dr, dc in directions:
                nr, nc = fire_row + dr, fire_col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not self.fire_grid[nr, nc] and not self.fireline_grid[nr, nc]:
                        candidates.append((nr, nc))

        return candidates

    def mark_fireline(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        blade_width: float = 2.5,
        robot_size: int = 3,
    ):
        """
        Mark cells under the robot's front blade as protected.
        The blade is a line segment perpendicular to robot heading at the front edge.
        Uses Bresenham's algorithm to keep the line at most 2 cells wide diagonally.
        """
        front_offset = robot_size / 2.0
        front_x = robot_x + front_offset * math.cos(robot_theta)
        front_y = robot_y + front_offset * math.sin(robot_theta)

        half_width = blade_width / 2.0
        perp_x = -math.sin(robot_theta)
        perp_y = math.cos(robot_theta)

        col0 = int(round(front_x - half_width * perp_x))
        row0 = int(round(front_y - half_width * perp_y))
        col1 = int(round(front_x + half_width * perp_x))
        row1 = int(round(front_y + half_width * perp_y))

        for col, row in self._bresenham_line(col0, row0, col1, row1):
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.fireline_grid[row, col] = True

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[tuple[int, int]]:
        """Generate cells along a line using Bresenham's algorithm."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break

            if x == x1:
                y += sy
            elif y == y1:
                x += sx
            else:
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dy
                    y += sy

        return cells
