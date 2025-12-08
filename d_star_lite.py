# D* Lite Path Planner with Subcell Resolution
# RBE 550, Firebots (course project)

import heapq
import math

import numpy as np


class DStarLite:
    """
    D* Lite with subcell resolution for tight gap navigation.

    Optimized for fast replanning with precomputed grids.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        robot_size: int = 3,
        heuristic_weight: float = 1.0,
        planning_resolution: int = 2,
    ):
        self.world_rows = rows
        self.world_cols = cols
        self.robot_size = robot_size
        self.robot_offset = robot_size // 2
        self.heuristic_weight = heuristic_weight

        # Subcell resolution
        self.resolution = planning_resolution
        self.rows = rows * planning_resolution
        self.cols = cols * planning_resolution
        self.scale = 1.0 / planning_resolution

        # 8-connected neighbors: (dr, dc, cost) - scaled by resolution
        base_cost = self.scale  # Movement cost scaled for finer grid
        diag_cost = self.scale * 1.41421356237
        self.neighbors = (
            (-1, 0, base_cost),
            (1, 0, base_cost),
            (0, -1, base_cost),
            (0, 1, base_cost),
            (-1, -1, diag_cost),
            (-1, 1, diag_cost),
            (1, -1, diag_cost),
            (1, 1, diag_cost),
        )

        self.SQRT2 = math.sqrt(2.0)
        self.SQRT2_MINUS_1 = self.SQRT2 - 1.0

        # Precomputed grids (filled in initialize())
        self.passable = None  # bool array at planning resolution
        self.plan_cost = None  # cost array at planning resolution

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
        self.start_world = None
        self.goal_world = None

    def _world_to_planning(self, world_row: int, world_col: int) -> tuple[int, int]:
        """Convert world cell to planning grid cell (center of world cell)."""
        pr = world_row * self.resolution + self.resolution // 2
        pc = world_col * self.resolution + self.resolution // 2
        return (pr, pc)

    def _planning_to_world_float(
        self, plan_row: int, plan_col: int
    ) -> tuple[float, float]:
        """Convert planning grid cell to world coordinates (continuous)."""
        world_x = (plan_col + 0.5) * self.scale
        world_y = (plan_row + 0.5) * self.scale
        return (world_x, world_y)

    def _precompute_passability(self, cost_grid: np.ndarray):
        """
        Precompute passability for all planning cells using vectorized numpy.

        This is the key optimization - instead of checking each cell on-demand,
        we compute everything upfront using fast array operations.
        """
        # Create planning-resolution coordinate grids
        plan_rows = np.arange(self.rows)
        plan_cols = np.arange(self.cols)
        pc_grid, pr_grid = np.meshgrid(plan_cols, plan_rows)

        # Convert to world coordinates (center of each planning cell)
        world_x = (pc_grid + 0.5) * self.scale
        world_y = (pr_grid + 0.5) * self.scale

        # World cell indices for robot center
        center_col = np.floor(world_x).astype(int)
        center_row = np.floor(world_y).astype(int)

        offset = self.robot_offset

        # Start with all passable
        self.passable = np.ones((self.rows, self.cols), dtype=bool)

        # Bounds check - robot footprint must fit
        self.passable[center_row < offset] = False
        self.passable[center_row >= self.world_rows - offset] = False
        self.passable[center_col < offset] = False
        self.passable[center_col >= self.world_cols - offset] = False

        # Check footprint against obstacles
        # For each footprint offset, check if that cell is blocked
        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
                # Compute which world cell this footprint position hits
                check_row = center_row + dr
                check_col = center_col + dc

                # Clamp to valid range for indexing (invalid positions already marked impassable)
                check_row_safe = np.clip(check_row, 0, self.world_rows - 1)
                check_col_safe = np.clip(check_col, 0, self.world_cols - 1)

                # Check if this footprint cell hits an obstacle
                footprint_blocked = ~np.isfinite(
                    cost_grid[check_row_safe, check_col_safe]
                )
                self.passable[footprint_blocked] = False

        # Allow cells near start to be passable (escape from tight spots)
        if self.start is not None:
            sr, sc = self.start
            for dr in range(-self.resolution, self.resolution + 1):
                for dc in range(-self.resolution, self.resolution + 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        # Only allow if not completely blocked
                        wr, wc = int((nc + 0.5) * self.scale), int(
                            (nr + 0.5) * self.scale
                        )
                        if 0 <= wr < self.world_rows and 0 <= wc < self.world_cols:
                            self.passable[nr, nc] = True

    def _precompute_costs(self, cost_grid: np.ndarray):
        """
        Precompute cost grid at planning resolution.

        Uses bilinear-ish sampling from world grid.
        """
        # Simple nearest-neighbor sampling for speed
        plan_rows = np.arange(self.rows)
        plan_cols = np.arange(self.cols)

        # World coordinates for each planning cell
        world_row = ((plan_rows + 0.5) * self.scale).astype(int)
        world_col = ((plan_cols + 0.5) * self.scale).astype(int)

        # Clamp to valid range
        world_row = np.clip(world_row, 0, self.world_rows - 1)
        world_col = np.clip(world_col, 0, self.world_cols - 1)

        # Sample cost grid
        self.plan_cost = cost_grid[world_row][:, world_col].astype(np.float32)

    def initialize(
        self,
        start: tuple,
        goal: tuple,
        cost_grid: np.ndarray,
        obstacle_grid: np.ndarray = None,
    ):
        """
        Initialize planner with precomputed grids for fast planning.
        """
        self.reset()

        self.start_world = start
        self.goal_world = goal
        self.start = self._world_to_planning(start[0], start[1])
        self.goal = self._world_to_planning(goal[0], goal[1])

        # Merge obstacle grid into cost grid
        if obstacle_grid is not None and obstacle_grid.any():
            merged_cost = cost_grid.copy()
            merged_cost[obstacle_grid] = np.inf
        else:
            merged_cost = cost_grid

        # Precompute grids (the key optimization)
        self._precompute_costs(merged_cost)
        self._precompute_passability(merged_cost)

        # Initialize goal
        gr, gc = self.goal
        self.rhs[gr, gc] = 0.0
        key = self._key(gr, gc)
        heapq.heappush(self.open_list, (key, (gr, gc)))
        self.open_set.add((gr, gc))

    def _heuristic(self, r1, c1, r2, c2):
        """Octile distance heuristic."""
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return (max(dr, dc) + self.SQRT2_MINUS_1 * min(dr, dc)) * self.scale

    def _key(self, row, col):
        """Calculate priority key for a cell."""
        g_val = self.g[row, col]
        rhs_val = self.rhs[row, col]
        min_val = min(g_val, rhs_val)

        if self.start:
            h = self._heuristic(row, col, self.start[0], self.start[1])
        else:
            h = 0.0

        return (min_val + h * self.heuristic_weight + self.km, min_val)

    def compute_shortest_path(self, max_iterations: int = 100000):
        """
        Expand cells until start is consistent.

        Optimized with local variable caching and inlined operations.
        """
        iterations = 0
        sr, sc = self.start
        gr, gc = self.goal

        # Cache as local variables for faster access
        g = self.g
        rhs = self.rhs
        open_list = self.open_list
        open_set = self.open_set
        rows = self.rows
        cols = self.cols
        neighbors = self.neighbors
        km = self.km
        passable = self.passable
        plan_cost = self.plan_cost
        h_weight = self.heuristic_weight
        scale = self.scale
        SQRT2_M1 = self.SQRT2_MINUS_1

        while open_list and iterations < max_iterations:
            iterations += 1

            # Compute start key inline
            g_s = g[sr, sc]
            rhs_s = rhs[sr, sc]
            min_s = g_s if g_s < rhs_s else rhs_s
            # Heuristic from start to start is 0
            start_key = (min_s + km, min_s)

            if open_list[0][0] >= start_key and g_s == rhs_s:
                break

            _, (row, col) = heapq.heappop(open_list)

            if (row, col) not in open_set:
                continue
            open_set.discard((row, col))

            g_val = g[row, col]
            rhs_val = rhs[row, col]

            if g_val > rhs_val:
                g[row, col] = rhs_val
                # Update all neighbors
                for dr, dc, move_cost in neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Inline _update_vertex for speed
                        if (nr, nc) != (gr, gc):
                            if not passable[nr, nc]:
                                rhs[nr, nc] = np.inf
                            else:
                                min_rhs = float("inf")
                                for dr2, dc2, mc2 in neighbors:
                                    nnr, nnc = nr + dr2, nc + dc2
                                    if 0 <= nnr < rows and 0 <= nnc < cols:
                                        if passable[nnr, nnc]:
                                            # Edge cost
                                            c1 = plan_cost[nr, nc]
                                            c2 = plan_cost[nnr, nnc]
                                            edge = mc2 * (1.0 + (c1 + c2) * 0.25)
                                            total = g[nnr, nnc] + edge
                                            if total < min_rhs:
                                                min_rhs = total
                                rhs[nr, nc] = min_rhs

                            open_set.discard((nr, nc))
                            if g[nr, nc] != rhs[nr, nc]:
                                # Compute key inline
                                gv = g[nr, nc]
                                rv = rhs[nr, nc]
                                mv = gv if gv < rv else rv
                                hdr = abs(nr - sr)
                                hdc = abs(nc - sc)
                                h = (max(hdr, hdc) + SQRT2_M1 * min(hdr, hdc)) * scale
                                key = (mv + h * h_weight + km, mv)
                                heapq.heappush(open_list, (key, (nr, nc)))
                                open_set.add((nr, nc))
            else:
                g[row, col] = np.inf
                # Update this cell and all neighbors
                cells_to_update = [(row, col)]
                for dr, dc, _ in neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        cells_to_update.append((nr, nc))

                for ur, uc in cells_to_update:
                    if (ur, uc) != (gr, gc):
                        if not passable[ur, uc]:
                            rhs[ur, uc] = np.inf
                        else:
                            min_rhs = float("inf")
                            for dr2, dc2, mc2 in neighbors:
                                nnr, nnc = ur + dr2, uc + dc2
                                if 0 <= nnr < rows and 0 <= nnc < cols:
                                    if passable[nnr, nnc]:
                                        c1 = plan_cost[ur, uc]
                                        c2 = plan_cost[nnr, nnc]
                                        edge = mc2 * (1.0 + (c1 + c2) * 0.25)
                                        total = g[nnr, nnc] + edge
                                        if total < min_rhs:
                                            min_rhs = total
                            rhs[ur, uc] = min_rhs

                        open_set.discard((ur, uc))
                        if g[ur, uc] != rhs[ur, uc]:
                            gv = g[ur, uc]
                            rv = rhs[ur, uc]
                            mv = gv if gv < rv else rv
                            hdr = abs(ur - sr)
                            hdc = abs(uc - sc)
                            h = (max(hdr, hdc) + SQRT2_M1 * min(hdr, hdc)) * scale
                            key = (mv + h * h_weight + km, mv)
                            heapq.heappush(open_list, (key, (ur, uc)))
                            open_set.add((ur, uc))

        path_exists = np.isfinite(g[sr, sc])
        if iterations > 1000:
            print(
                f"D* Lite: {iterations} iters, path={'found' if path_exists else 'NONE'}"
            )
        return path_exists

    def extract_path(
        self,
        smooth_iterations: int = 1,
        smooth_weight: float = 0.2,
        step_size: float = 1.0,
        momentum: float = 0.3,
        goal_tolerance: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Extract optimal path in world coordinates."""
        if self.start is None or self.goal is None:
            return []

        sr, sc = self.start
        if not np.isfinite(self.g[sr, sc]):
            return []

        # Extract path in planning coordinates
        planning_path = self._follow_optimal_path()

        if len(planning_path) < 1:
            return []

        # Convert to world coordinates
        path = []
        for pr, pc in planning_path:
            wx, wy = self._planning_to_world_float(pr, pc)
            path.append((wx, wy))

        # Optional smoothing
        if len(path) > 3 and smooth_iterations > 0:
            path = self._smooth_path(path, smooth_iterations, smooth_weight)

        return path

    def _follow_optimal_path(self) -> list[tuple[int, int]]:
        """Extract path by following lowest g-values."""
        path = [self.start]
        current = self.start
        visited = {self.start}
        max_steps = self.rows * self.cols

        g = self.g
        passable = self.passable
        neighbors = self.neighbors
        rows, cols = self.rows, self.cols

        for _ in range(max_steps):
            if current == self.goal:
                break

            row, col = current
            best_next = None
            best_g = float("inf")

            for dr, dc, _ in neighbors:
                nr, nc = row + dr, col + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in visited:
                    continue
                if not passable[nr, nc]:
                    continue

                gv = g[nr, nc]
                if np.isfinite(gv) and gv < best_g:
                    best_g = gv
                    best_next = (nr, nc)

            if best_next is None:
                break

            path.append(best_next)
            visited.add(best_next)
            current = best_next

        return path

    def _smooth_path(
        self, path: list, iterations: int = 1, weight: float = 0.2
    ) -> list:
        """Smooth path while checking passability."""
        if len(path) <= 2:
            return path

        smoothed = list(path)
        passable = self.passable
        res = self.resolution

        for _ in range(iterations):
            new_path = [smoothed[0]]
            for i in range(1, len(smoothed) - 1):
                prev_x, prev_y = smoothed[i - 1]
                curr_x, curr_y = smoothed[i]
                next_x, next_y = smoothed[i + 1]

                new_x = curr_x + weight * ((prev_x + next_x) / 2 - curr_x)
                new_y = curr_y + weight * ((prev_y + next_y) / 2 - curr_y)

                # Check passability
                pr = int(new_y * res)
                pc = int(new_x * res)
                pr = max(0, min(pr, self.rows - 1))
                pc = max(0, min(pc, self.cols - 1))

                if passable[pr, pc]:
                    new_path.append((new_x, new_y))
                else:
                    new_path.append((curr_x, curr_y))

            new_path.append(smoothed[-1])
            smoothed = new_path

        return smoothed

    def is_path_valid(
        self, path: list[tuple[float, float]], max_segment_length: float = 5.0
    ) -> bool:
        """Check if a path is reasonable."""
        if len(path) < 2:
            return len(path) == 1

        passable = self.passable
        res = self.resolution

        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_segment_length:
                return False

            pr = int(y2 * res)
            pc = int(x2 * res)
            pr = max(0, min(pr, self.rows - 1))
            pc = max(0, min(pc, self.cols - 1))

            if not passable[pr, pc]:
                return False

        return True

    def extract_path_cell_coords(self) -> list[tuple[int, int]]:
        """Extract path as world cell coordinates."""
        smooth_path = self.extract_path()
        if not smooth_path:
            return []

        cell_path = []
        for x, y in smooth_path:
            row, col = int(y), int(x)
            row = max(0, min(row, self.world_rows - 1))
            col = max(0, min(col, self.world_cols - 1))
            if not cell_path or cell_path[-1] != (row, col):
                cell_path.append((row, col))

        return cell_path

    def update_start(self, new_start: tuple):
        """Update start position for incremental replanning."""
        new_start_planning = self._world_to_planning(new_start[0], new_start[1])
        if self.start:
            self.km += self._heuristic(
                self.start[0],
                self.start[1],
                new_start_planning[0],
                new_start_planning[1],
            )
        self.start = new_start_planning
        self.start_world = new_start
