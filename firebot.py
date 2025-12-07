# Firebot: Unicycle/Differential-Drive Kinematics
# RBE 550, Firebots (course project)
# Updated with pure pursuit path following

import math
import numpy as np


class Firebot:
    """
    Unicycle kinematic model for a differential-drive robot.

    State: (x, y, theta) where x,y are in cell coordinates (continuous),
           theta is heading in radians (0 = +x direction, counter-clockwise positive).

    Robot footprint: 3x3 cells (9x9 ft at 1 cell = 3 ft scale).
    """

    def __init__(
            self,
            x: float,
            y: float,
            theta: float = 0.0,
            sensor_radius: float = 10.0,
            fire_approach_margin: float = 0.5,
    ):
        # Position in cell coordinates (fractional, continuous)
        self.x = x
        self.y = y
        self.theta = theta  # radians, 0 = facing right (+x), CCW positive

        # Sensor radius for fog of war (in fractional cells)
        self.sensor_radius = sensor_radius

        # Margin to keep from fire when approaching (in fractional cells)
        self.fire_approach_margin = fire_approach_margin

        # Robot parameters (in cells)
        self.size = 3  # 3x3 cell footprint
        self.wheel_base = 2.0  # distance between wheels in cells (for diff-drive)

        # Velocity limits (cells per second, radians per second)
        self.max_linear_vel = 2.0  # cells/s
        self.max_angular_vel = math.pi  # rad/s (180 deg/s)

        # Current velocities
        self.v = 0.0  # linear velocity (cells/s)
        self.omega = 0.0  # angular velocity (rad/s)

        # Motion controller state
        self.target = None  # (x, y) target position
        self.state = "idle"  # "idle", "rotating", "driving"

        # Controller gains
        self.k_angular = 3.0  # proportional gain for rotation
        self.k_linear = 2.0  # proportional gain for linear motion
        self.angle_threshold = 0.1  # rad (~6 deg) - when to stop rotating
        self.distance_threshold = 0.5  # cells - when to consider "arrived"

        # Fireline cutting mode
        self.cutting_fireline = True  # when True, record path as fireline
        # Each sample is (x, y, theta) - front position and heading for shovel line
        self.fireline_path: list[tuple[float, float, float]] = []
        self._last_fireline_pos: tuple[float, float] | None = None
        self._last_fireline_theta: float | None = None
        self.fireline_sample_dist = 0.3  # min distance between samples (cells)
        self.fireline_sample_angle = 0.15  # min angle change (~8.5 degrees)

        # Path following state
        self._prev_v = 0.0
        self._prev_omega = 0.0
        self._path_index = 0  # Current waypoint index

    def update(self, dt: float):
        """
        Update robot state using unicycle kinematics:
            x_dot = v * cos(theta)
            y_dot = v * sin(theta)
            theta_dot = omega
        """
        if dt <= 0:
            return

        # Integrate kinematics
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Normalize theta to [-pi, pi]
        self.theta = self._normalize_angle(self.theta)

        # Sample fireline when moving
        if self.cutting_fireline and (abs(self.v) > 0.01 or abs(self.omega) > 0.01):
            self._sample_fireline()

    def follow_path(
            self,
            path: list[tuple[float, float]],
            weight_grid: np.ndarray,
            dt: float,
            lookahead: float = 2.5,
    ) -> bool | str:
        """
        Pure pursuit path following with collision checking.

        The path from D* Lite is already collision-free, so we just need to:
        1. Track the path accurately
        2. Check for unexpected obstacles

        Args:
            path: List of (x, y) waypoints from D* Lite
            weight_grid: Cost grid (inf = obstacle)
            dt: Time step
            lookahead: How far ahead on path to target

        Returns:
            True if following path, False if reached goal, "stuck" if blocked
        """
        if not path:
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return False

        # Check if reached goal
        goal_x, goal_y = path[-1]
        dist_to_goal = math.sqrt((goal_x - self.x) ** 2 + (goal_y - self.y) ** 2)
        if dist_to_goal < 0.7:
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return False

        # Find the target point on path using pure pursuit
        target = self._get_pursuit_target(path, lookahead)

        # Compute steering toward target
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist_to_target = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - self.theta)

        # Check for collision toward target (not along robot heading)
        blocked = self._check_collision_ahead(weight_grid, target[0], target[1])

        if blocked:
            # Something unexpected blocking us
            if abs(angle_error) > 0.5:
                # Turn toward target
                self.v = 0.0
                self.omega = np.clip(self.k_angular * angle_error,
                                     -self.max_angular_vel, self.max_angular_vel)
            else:
                # Aligned but blocked - need replan
                self.v = 0.0
                self.omega = 0.0
                self.update(dt)
                return "stuck"
        else:
            # Clear path - pure pursuit control

            # Speed based on: angle error, distance to goal, curvature
            turn_slowdown = max(0.2, 1.0 - abs(angle_error) / (math.pi / 2))
            goal_slowdown = min(1.0, dist_to_goal / 2.0)

            target_v = self.max_linear_vel * turn_slowdown * goal_slowdown

            # Angular velocity proportional to angle error
            target_omega = self.k_angular * angle_error
            target_omega = np.clip(target_omega, -self.max_angular_vel, self.max_angular_vel)

            # If very misaligned, turn more aggressively
            if abs(angle_error) > math.pi / 2:
                target_v *= 0.2
                target_omega *= 1.5
                target_omega = np.clip(target_omega, -self.max_angular_vel, self.max_angular_vel)

            # Smooth velocity changes
            alpha = 0.3  # smoothing factor
            self.v = alpha * self._prev_v + (1 - alpha) * target_v
            self.omega = alpha * self._prev_omega + (1 - alpha) * target_omega

        self._prev_v = self.v
        self._prev_omega = self.omega

        self.update(dt)
        return True

    def _get_pursuit_target(self, path: list[tuple[float, float]],
                            lookahead: float) -> tuple[float, float]:
        """
        Find target point on path for pure pursuit.
        Properly tracks along the path, not just closest point.
        """
        if not path:
            return (self.x, self.y)

        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0

        for i, (px, py) in enumerate(path):
            dist = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Look ahead from closest point to find target
        lookahead_sq = lookahead * lookahead

        for i in range(closest_idx, len(path)):
            px, py = path[i]
            dist_sq = (px - self.x) ** 2 + (py - self.y) ** 2
            if dist_sq >= lookahead_sq:
                return (px, py)

        # If no point at lookahead distance, return last point
        return path[-1]

    def _check_collision_ahead(self, weight_grid: np.ndarray,
                               target_x: float, target_y: float) -> bool:
        """
        Check if there's an obstacle between robot and target.
        Only checks for inf (actual obstacles), not high costs.
        """
        rows, cols = weight_grid.shape
        half = self.size / 2.0

        # Check along direction TO TARGET, not robot heading
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.5:
            return False

        # Normalize direction
        dx /= dist
        dy /= dist

        # Check a few points toward target
        check_dists = [0.5, 1.0, 1.5, 2.0]

        for d in check_dists:
            if d > dist:
                break

            check_x = self.x + d * dx
            check_y = self.y + d * dy

            # Check footprint corners
            corners = [
                (check_x, check_y),
                (check_x + half, check_y + half),
                (check_x + half, check_y - half),
                (check_x - half, check_y + half),
                (check_x - half, check_y - half),
            ]

            for cx, cy in corners:
                col, row = int(cx), int(cy)
                if row < 0 or row >= rows or col < 0 or col >= cols:
                    return True
                if not np.isfinite(weight_grid[row, col]):
                    return True

        return False

    def set_target(self, target_x: float, target_y: float, force_rotate: bool = False):
        """
        Set a new target position to drive to.
        """
        if self.target is not None and not force_rotate:
            old_dx = self.target[0] - self.x
            old_dy = self.target[1] - self.y
            new_dx = target_x - self.x
            new_dy = target_y - self.y

            if old_dx * old_dx + old_dy * old_dy > 0.01:
                old_angle = math.atan2(old_dy, old_dx)
                new_angle = math.atan2(new_dy, new_dx)
                angle_diff = abs(self._normalize_angle(new_angle - old_angle))

                if angle_diff < 0.3 and self.state == "driving":
                    self.target = (target_x, target_y)
                    return

        self.target = (target_x, target_y)

        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        angle_error = abs(self._normalize_angle(target_angle - self.theta))

        if angle_error > self.angle_threshold * 3 or force_rotate:
            self.state = "rotating"
        else:
            self.state = "driving"

    def pure_pursuit_step(
            self, path: list[tuple[float, float]], dt: float, lookahead: float = 2.5
    ):
        """
        Pure pursuit controller for smooth path following.
        No turn-in-place - always moves forward while steering.

        Args:
            path: List of (x, y) waypoints
            dt: Time step
            lookahead: Lookahead distance in cells

        Returns:
            True if still following path, False if reached end or no path
        """
        if not path or len(path) < 1:
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return False

        # Find lookahead point on path
        lookahead_sq = lookahead * lookahead
        target_point = None

        # Find closest point on path, skipping points behind us
        min_dist_sq = float("inf")
        closest_idx = 0

        for i, (px, py) in enumerate(path):
            dx = px - self.x
            dy = py - self.y
            dist_sq = dx * dx + dy * dy

            # Skip points that are behind us
            if dist_sq > 0.25:  # Not too close
                angle_to_point = math.atan2(dy, dx)
                angle_diff = abs(self._normalize_angle(angle_to_point - self.theta))
                if angle_diff > math.pi * 0.7:  # Behind us (>126°)
                    continue

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i

        # Look ahead from closest point
        for i in range(closest_idx, len(path)):
            px, py = path[i]
            dx = px - self.x
            dy = py - self.y
            dist_sq = dx * dx + dy * dy

            if dist_sq >= lookahead_sq:
                target_point = (px, py)
                break

        # If no point at lookahead, use last point
        if target_point is None:
            target_point = path[-1]

        # Check if we've reached the goal
        goal_x, goal_y = path[-1]
        dx = goal_x - self.x
        dy = goal_y - self.y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)

        if dist_to_goal < 0.7:
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return False  # Reached goal

        # Compute control toward target point
        tx, ty = target_point
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.1:
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return True

        # Angle to target
        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - self.theta)

        # Linear velocity - slow down for sharp turns and near goal
        turn_factor = max(0.3, 1.0 - abs(angle_error) / math.pi)
        approach_factor = min(1.0, dist_to_goal / 3.0)
        self.v = self.max_linear_vel * turn_factor * approach_factor

        # Angular velocity - proportional to angle error
        self.omega = self.k_angular * angle_error
        self.omega = max(-self.max_angular_vel, min(self.max_angular_vel, self.omega))

        # If angle error is very large (> 90°), reduce speed more
        if abs(angle_error) > math.pi / 2:
            self.v *= 0.3

        self.update(dt)
        return True

    def control_step(self, dt: float):
        """
        Execute one step of the motion controller.
        Uses a rotate-then-drive approach with smooth transitions.
        """
        if self.target is None or self.state == "idle":
            self.v = 0.0
            self.omega = 0.0
            self.update(dt)
            return

        tx, ty = self.target
        dx = tx - self.x
        dy = ty - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - self.theta)

        if self.state == "rotating":
            if abs(angle_error) < self.angle_threshold:
                self.omega = 0.0
                self.state = "driving"
            else:
                self.omega = np.clip(
                    self.k_angular * angle_error,
                    -self.max_angular_vel,
                    self.max_angular_vel,
                )
                self.v = 0.0

        elif self.state == "driving":
            if distance < self.distance_threshold:
                self.v = 0.0
                self.omega = 0.0
                self.state = "idle"
                self.target = None
                self.update(dt)
                return

            if abs(angle_error) > self.angle_threshold * 4:
                self.state = "rotating"
                self.update(dt)
                return

            speed_factor = min(1.0, distance / 2.0)
            self.v = np.clip(
                self.k_linear * distance * speed_factor, 0, self.max_linear_vel
            )

            self.omega = np.clip(
                self.k_angular * 0.5 * angle_error,
                -self.max_angular_vel * 0.3,
                self.max_angular_vel * 0.3,
            )

        self.update(dt)

    def set_wheel_velocities(self, v_left: float, v_right: float):
        """Set velocities from differential drive wheel speeds."""
        self.v = (v_right + v_left) / 2.0
        self.omega = (v_right - v_left) / self.wheel_base
        self.v = np.clip(self.v, -self.max_linear_vel, self.max_linear_vel)
        self.omega = np.clip(self.omega, -self.max_angular_vel, self.max_angular_vel)

    def get_wheel_velocities(self) -> tuple[float, float]:
        """Get differential drive wheel speeds from unicycle model."""
        v_left = self.v - self.omega * self.wheel_base / 2.0
        v_right = self.v + self.omega * self.wheel_base / 2.0
        return v_left, v_right

    def get_footprint_cells(self) -> list[tuple[int, int]]:
        """Get list of (row, col) cells occupied by the robot's 3x3 footprint."""
        cells = []
        cx, cy = int(round(self.x)), int(round(self.y))
        offset = self.size // 2

        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
                cells.append((cy + dr, cx + dc))

        return cells

    def is_moving(self) -> bool:
        """Check if robot is currently moving (velocity-based)."""
        return abs(self.v) > 0.01 or abs(self.omega) > 0.01

    def stop(self):
        """Immediately stop the robot and clear target."""
        self.v = 0.0
        self.omega = 0.0
        self.state = "idle"
        self.target = None

    def _sample_fireline(self):
        """Record front position and heading for fireline if moved or rotated enough."""
        if not self.cutting_fireline:
            return

        front = self.front_position

        if self._last_fireline_pos is None or self._last_fireline_theta is None:
            self.fireline_path.append((front[0], front[1], self.theta))
            self._last_fireline_pos = front
            self._last_fireline_theta = self.theta
            return

        dx = front[0] - self._last_fireline_pos[0]
        dy = front[1] - self._last_fireline_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        angle_diff = abs(self._normalize_angle(self.theta - self._last_fireline_theta))

        if dist >= self.fireline_sample_dist or angle_diff >= self.fireline_sample_angle:
            self.fireline_path.append((front[0], front[1], self.theta))
            self._last_fireline_pos = front
            self._last_fireline_theta = self.theta

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @property
    def position(self) -> tuple[float, float]:
        """Get current position as (x, y)."""
        return (self.x, self.y)

    @property
    def front_position(self) -> tuple[float, float]:
        """Get position of the front edge center (1.5 cells ahead of center)."""
        front_offset = self.size / 2.0
        front_x = self.x + front_offset * math.cos(self.theta)
        front_y = self.y + front_offset * math.sin(self.theta)
        return (front_x, front_y)

    @property
    def heading_deg(self) -> float:
        """Get heading in degrees."""
        return math.degrees(self.theta)