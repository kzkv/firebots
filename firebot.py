# Firebot: Unicycle/Differential-Drive Kinematics
# RBE 550, Firebots (course project)

import math
import numpy as np


class Firebot:
    """
    Unicycle kinematic model for a differential-drive robot.

    State: (x, y, theta) where x,y are in cell coordinates (continuous),
           theta is heading in radians (0 = +x direction, counter-clockwise positive).

    Robot footprint: 3x3 cells (9x9 ft at 1 cell = 3 ft scale).
    """

    def __init__(self, x: float, y: float, theta: float = 0.0):
        # Position in cell coordinates (fractional, continuous)
        self.x = x
        self.y = y
        self.theta = theta  # radians, 0 = facing right (+x), CCW positive

        # Robot parameters (in cells)
        self.size = 3  # 3x3 cell footprint
        self.wheel_base = 2.0  # distance between wheels in cells (for diff-drive)

        # Velocity limits (cells per second, radians per second)
        self.max_linear_vel = 5.0  # cells/s
        self.max_angular_vel = math.pi  # rad/s (180 deg/s)

        # Current velocities
        self.v = 0.0  # linear velocity (cells/s)
        self.omega = 0.0  # angular velocity (rad/s)

        # Motion controller state
        self.target = None  # (x, y) target position
        self.state = "idle"  # "idle", "rotating", "driving", "final_rotate"

        # Controller gains
        self.k_angular = 3.0  # proportional gain for rotation
        self.k_linear = 2.0  # proportional gain for linear motion
        self.angle_threshold = 0.05  # rad (~3 deg) - when to stop rotating
        self.distance_threshold = 0.1  # cells - when to stop driving

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
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def set_target(self, target_x: float, target_y: float):
        """Set a new target position to drive to."""
        self.target = (target_x, target_y)
        self.state = "rotating"

    def control_step(self, dt: float):
        """
        Execute one step of the motion controller.
        Uses a rotate-then-drive approach:
        1. Rotate to face target
        2. Drive straight to target
        """
        if self.target is None or self.state == "idle":
            self.v = 0.0
            self.omega = 0.0
            return

        tx, ty = self.target
        dx = tx - self.x
        dy = ty - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        # Angle to target
        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - self.theta)

        if self.state == "rotating":
            # Rotate to face target
            if abs(angle_error) < self.angle_threshold:
                self.omega = 0.0
                self.state = "driving"
            else:
                # P controller for rotation
                self.omega = np.clip(
                    self.k_angular * angle_error,
                    -self.max_angular_vel,
                    self.max_angular_vel,
                )
                self.v = 0.0

        elif self.state == "driving":
            # Check if we've arrived
            if distance < self.distance_threshold:
                self.v = 0.0
                self.omega = 0.0
                self.state = "idle"
                self.target = None
                return

            # Recalculate angle error while driving (small corrections)
            if abs(angle_error) > self.angle_threshold * 2:
                # Need to re-rotate if we've drifted too much
                self.state = "rotating"
                return

            # Drive forward with small angular corrections
            self.v = np.clip(self.k_linear * distance, 0, self.max_linear_vel)
            # Small heading corrections while driving
            self.omega = np.clip(
                self.k_angular * 0.5 * angle_error,
                -self.max_angular_vel * 0.3,
                self.max_angular_vel * 0.3,
            )

        # Update state
        self.update(dt)

    def set_wheel_velocities(self, v_left: float, v_right: float):
        """
        Set velocities from differential drive wheel speeds.
        Converts to unicycle model (v, omega).

        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / wheel_base
        """
        self.v = (v_right + v_left) / 2.0
        self.omega = (v_right - v_left) / self.wheel_base

        # Clamp to limits
        self.v = np.clip(self.v, -self.max_linear_vel, self.max_linear_vel)
        self.omega = np.clip(self.omega, -self.max_angular_vel, self.max_angular_vel)

    def get_wheel_velocities(self) -> tuple[float, float]:
        """
        Get differential drive wheel speeds from unicycle model.

        v_left = v - omega * wheel_base / 2
        v_right = v + omega * wheel_base / 2
        """
        v_left = self.v - self.omega * self.wheel_base / 2.0
        v_right = self.v + self.omega * self.wheel_base / 2.0
        return v_left, v_right

    def get_footprint_cells(self) -> list[tuple[int, int]]:
        """
        Get list of (row, col) cells occupied by the robot's 3x3 footprint.
        Robot center is at (self.x, self.y).
        """
        cells = []
        cx, cy = int(round(self.x)), int(round(self.y))
        offset = self.size // 2  # 1 for 3x3

        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
                cells.append((cy + dr, cx + dc))

        return cells

    def is_moving(self) -> bool:
        """Check if robot is currently moving."""
        return self.state != "idle" or abs(self.v) > 0.01 or abs(self.omega) > 0.01

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @property
    def position(self) -> tuple[float, float]:
        """Get current position as (x, y)."""
        return (self.x, self.y)

    @property
    def heading_deg(self) -> float:
        """Get heading in degrees."""
        return math.degrees(self.theta)
