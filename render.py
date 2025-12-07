"""
Tree sprite source: https://github.com/jube/slowtree
"""

import hashlib
import math

import numpy as np
import pygame

CELL_BG_COLOR = (255, 255, 255)
CELL_GRID_COLOR = (200, 200, 200)
CELL_GRID_BOLD_COLOR = (20, 20, 20)
LEGEND_LABEL_TEXT_COLOR = (100, 100, 100)
LEGEND_BG_COLOR = (0, 0, 0)
HUD_BG_COLOR = (255, 255, 255)
FIRE_BITMAP_ALPHA = 64
FIRE_CELL_COLOR = (255, 0, 0)
FIRE_CELL_ALPHA = 128
TREE_CELL_COLOR = (155, 103, 60)
TREE_CELL_ALPHA = 200
TREE_SPRITE_ALPHA = 200
FIRELINE_CELL_COLOR = (255, 165, 0)
FIRELINE_CELL_ALPHA = 200
FIRELINE_PATH_COLOR = (139, 90, 43)  # saddle brown for cut fireline
FIRELINE_PATH_ALPHA = 180
FOG_OF_WAR_COLOR = (40, 40, 40)
FOG_OF_WAR_ALPHA = 200
FORBIDDEN_ZONE_COLOR = (180, 0, 180)
FORBIDDEN_ZONE_ALPHA = 100
INSET = 2


# TODO: homogenize sight discrepancies in the methods' code


class World:
    def __init__(self, rows, cols, cell_size):
        self.rows, self.cols = rows, cols
        self.field_width, self.field_height = cols * cell_size, rows * cell_size
        self.cell_size = cell_size
        self.hud_height = cell_size
        self.margin = cell_size  # 1 cell on each side for chessboard-style coordinates

        self.window_width = self.field_width + 2 * self.margin
        self.window_height = self.field_height + 2 * self.margin + self.hud_height

        # Establish field rect (workable area inside of the margins)
        self.field_rect = pygame.Rect(
            self.margin, self.margin, self.field_width, self.field_height
        )

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, self.cell_size - 4)
        self.tooltip_font = pygame.font.SysFont(None, max(12, self.cell_size // 2))

        self.show_weights = False
        self.show_forbidden_zone = False
        self.show_arrows = False
        self.show_fireline_grid = False
        self.hud_font = pygame.font.SysFont(None, max(12, self.cell_size - 6))
        self.hud_rect = pygame.Rect(
            0, self.field_rect.bottom + self.margin, self.window_width, self.hud_height
        )

    def clear(self):
        self.screen.fill(CELL_BG_COLOR)

    def render_grid(self):
        # Cell grid lines
        for x in range(self.cols + 1):
            X = self.field_rect.left + x * self.cell_size
            pygame.draw.line(
                self.screen,
                CELL_GRID_COLOR,
                start_pos=(X, self.field_rect.top),
                end_pos=(X, self.field_rect.bottom),
            )
        for y in range(self.rows + 1):
            Y = self.field_rect.top + y * self.cell_size
            pygame.draw.line(
                self.screen,
                CELL_GRID_COLOR,
                start_pos=(self.field_rect.left, Y),
                end_pos=(self.field_rect.right, Y),
            )

        # Prominent inner lines every 12 cells
        for x in range(12, self.cols, 12):
            X = self.field_rect.left + x * self.cell_size
            pygame.draw.line(
                self.screen,
                CELL_GRID_BOLD_COLOR,
                start_pos=(X, self.field_rect.top),
                end_pos=(X, self.field_rect.bottom),
            )
        for y in range(12, self.rows, 12):
            Y = self.field_rect.top + y * self.cell_size
            pygame.draw.line(
                self.screen,
                CELL_GRID_BOLD_COLOR,
                start_pos=(self.field_rect.left, Y),
                end_pos=(self.field_rect.right, Y),
            )

        # Chessboard-style coordinates
        for x in range(self.cols):
            rt = pygame.Rect(
                self.field_rect.left + x * self.cell_size,
                0,
                self.cell_size,
                self.margin,
            )
            rb = pygame.Rect(
                self.field_rect.left + x * self.cell_size,
                self.field_rect.bottom,
                self.cell_size,
                self.margin,
            )
            lab = self.font.render(str(x + 1), True, LEGEND_LABEL_TEXT_COLOR)
            self.screen.blit(lab, lab.get_rect(center=rt.center))
            self.screen.blit(lab, lab.get_rect(center=rb.center))

        for y in range(self.rows):
            rl = pygame.Rect(
                0, self.field_rect.top + y * self.cell_size, self.margin, self.cell_size
            )
            rr = pygame.Rect(
                self.field_rect.right,
                self.field_rect.top + y * self.cell_size,
                self.margin,
                self.cell_size,
            )
            lab = self.font.render(str(y + 1), True, LEGEND_LABEL_TEXT_COLOR)
            self.screen.blit(lab, lab.get_rect(center=rl.center))
            self.screen.blit(lab, lab.get_rect(center=rr.center))

        # HUD
        hud_rect = pygame.Rect(
            0, self.field_rect.bottom + self.margin, self.window_width, self.hud_height
        )
        pygame.draw.rect(self.screen, HUD_BG_COLOR, hud_rect)

    def fire_bitmap_overlay(self, fire_bitmap):
        # Scale if needed
        if fire_bitmap.get_size() != (self.field_rect.width, self.field_rect.height):
            fire_bitmap = pygame.transform.smoothscale(
                fire_bitmap, (self.field_rect.width, self.field_rect.height)
            )
        fire_bitmap.set_alpha(FIRE_BITMAP_ALPHA)
        self.screen.blit(fire_bitmap, self.field_rect.topleft)

    def render_fire_cells(self, fire_grid):
        fire_overlay = pygame.Surface(
            (self.field_rect.width, self.field_rect.height), pygame.SRCALPHA
        )

        cell = self.cell_size
        for row in range(self.rows):
            y = row * cell
            for col in range(self.cols):
                if fire_grid[row, col]:
                    x = col * cell
                    # inset slightly so grid lines remain visible
                    pygame.draw.rect(
                        fire_overlay,
                        (*FIRE_CELL_COLOR, FIRE_CELL_ALPHA),
                        pygame.Rect(
                            x + INSET, y + INSET, cell - 2 * INSET, cell - 2 * INSET
                        ),
                    )
        self.screen.blit(fire_overlay, self.field_rect.topleft)

    def render_trees(self, tree_grid):
        """Render tree cell overlays."""
        rows, cols = tree_grid.shape
        tree_overlay = pygame.Surface(
            (self.field_rect.width, self.field_rect.height), pygame.SRCALPHA
        )
        fill = (*TREE_CELL_COLOR, TREE_CELL_ALPHA)
        for r in range(rows):
            y = r * self.cell_size
            for c in range(cols):
                if tree_grid[r, c]:
                    x = c * self.cell_size
                    pygame.draw.rect(
                        tree_overlay,
                        fill,
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                    )
        self.screen.blit(tree_overlay, self.field_rect.topleft)

    def render_tree_sprites(self, tree_grid, rng):
        """Render tree sprites."""
        sprites = [
            pygame.image.load("assets/tree1.png").convert_alpha(),
            pygame.image.load("assets/tree2.png").convert_alpha(),
            pygame.image.load("assets/tree3.png").convert_alpha(),
            pygame.image.load("assets/tree4.png").convert_alpha(),
        ]

        sprite_px = 3 * self.cell_size
        scaled = [
            pygame.transform.smoothscale(s.convert_alpha(), (sprite_px, sprite_px))
            for s in sprites
        ]

        tree_sprite_overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)

        rows, cols = tree_grid.shape
        for row in range(rows):
            y = row * self.cell_size
            for col in range(cols):
                if not tree_grid[row, col]:
                    continue
                x = col * self.cell_size

                key = f"{row},{col}".encode()
                digest = hashlib.blake2b(key, digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % len(scaled)

                dest = (x - self.cell_size, y - self.cell_size)
                tree_sprite_overlay.blit(scaled[idx], dest)

        tree_sprite_overlay.set_alpha(TREE_SPRITE_ALPHA)
        self.screen.blit(tree_sprite_overlay, self.field_rect.topleft)

    def render_fireline_cells(self, fireline_grid):
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size
        rows, cols = fireline_grid.shape
        for r in range(rows):
            y = r * cell
            for c in range(cols):
                if fireline_grid[r, c]:
                    x = c * cell
                    pygame.draw.rect(
                        overlay,
                        (*FIRELINE_CELL_COLOR, FIRELINE_CELL_ALPHA),
                        pygame.Rect(
                            x + INSET, y + INSET, cell - 2 * INSET, cell - 2 * INSET
                        ),
                    )
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_forbidden_zone(self, forbidden_zone, resolution):
        """Render the forbidden zone (Minkowski-inflated fire) as a semi-transparent overlay.

        Args:
            forbidden_zone: Boolean grid at subgrid resolution
            resolution: Number of subcells per cell
        """
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        subcell_size = self.cell_size / resolution
        sub_rows, sub_cols = forbidden_zone.shape
        for r in range(sub_rows):
            y = r * subcell_size
            for c in range(sub_cols):
                if forbidden_zone[r, c]:
                    x = c * subcell_size
                    pygame.draw.rect(
                        overlay,
                        (*FORBIDDEN_ZONE_COLOR, FORBIDDEN_ZONE_ALPHA),
                        pygame.Rect(x, y, subcell_size + 1, subcell_size + 1),
                    )
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_fireline_path(self, path: list[tuple[float, float, float]], width_cells: float = 2.5):
        """
        Render the fireline by stamping shovel lines at each sample point.

        Args:
            path: List of (x, y, theta) - front position and heading
            width_cells: Width of the shovel in cells (default 2.5)
        """
        if len(path) < 1:
            return

        # Draw solid lines first, then apply alpha to whole surface
        overlay = pygame.Surface(self.field_rect.size)
        overlay.fill((0, 0, 0))  # black background for colorkey
        overlay.set_colorkey((0, 0, 0))  # black is transparent

        half_width = width_cells * self.cell_size / 2
        # Line thickness should be thick enough to fill gaps between samples
        line_thickness = max(3, int(self.cell_size * 0.3))

        # Draw a line (the shovel) at each sample point
        for x, y, theta in path:
            sx = x * self.cell_size
            sy = y * self.cell_size

            # Perpendicular to heading (shovel is perpendicular to travel direction)
            perp_x = -math.sin(theta) * half_width
            perp_y = math.cos(theta) * half_width

            left = (sx + perp_x, sy + perp_y)
            right = (sx - perp_x, sy - perp_y)

            pygame.draw.line(overlay, FIRELINE_PATH_COLOR, left, right, line_thickness)

        # Apply alpha to the entire surface
        overlay.set_alpha(FIRELINE_PATH_ALPHA)
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_weight_heatmap(self, weights, vmax=None):
        import numpy as np

        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size
        finite = np.isfinite(weights)
        if vmax is None:
            # robust upper bound (ignore extreme outliers)
            vmax = np.percentile(weights[finite], 95) if finite.any() else 1.0
        vmax = max(vmax, 1e-6)

        for r in range(self.rows):
            y = r * cell
            for c in range(self.cols):
                w = weights[r, c]
                if not np.isfinite(w):  # impassable
                    color = (0, 0, 0, 180)
                elif w <= 1.0:  # base cost
                    continue
                else:
                    t = min(1.0, (w - 1.0) / vmax)  # 0..1
                    # orange ramp
                    color = (255, int(200 * (1 - t)), 0, int(160 * t) + 40)
                pygame.draw.rect(overlay, color, pygame.Rect(c * cell, y, cell, cell))
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_weight_on_hover(
        self,
        weights,
        decimals=0,
        *,
        text_color=(20, 20, 20),
        bg=(255, 255, 255),
        border=(40, 40, 40),
    ):
        import numpy as np, pygame as pg

        mx, my = pg.mouse.get_pos()
        if not self.field_rect.collidepoint(mx, my):
            return
        col = (mx - self.field_rect.left) // self.cell_size
        row = (my - self.field_rect.top) // self.cell_size
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return

        w = weights[row, col]
        msg = (
            "X"
            if not np.isfinite(w)
            else (f"{w:.{decimals}f}" if decimals else str(int(round(w))))
        )
        txt = self.tooltip_font.render(msg, True, text_color)

        # simple tooltip near cursor, clamped to screen
        pad = 4
        tip_rect = txt.get_rect().inflate(pad * 2, pad * 2)
        tip_rect.topleft = (mx + 12, my + 10)
        tip_rect.clamp_ip(self.screen.get_rect())

        tip = pg.Surface(tip_rect.size, pg.SRCALPHA)
        tip.fill((*bg, 230))
        pg.draw.rect(tip, (*border, 220), tip.get_rect(), 1)
        self.screen.blit(tip, tip_rect.topleft)
        self.screen.blit(txt, txt.get_rect(center=tip_rect.center))

        # optional: highlight hovered cell
        cell_rect = pg.Rect(
            self.field_rect.left + col * self.cell_size,
            self.field_rect.top + row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pg.draw.rect(self.screen, border, cell_rect, 1)

    def _draw_checkbox(self, x, y, box, checked, label_text):
        """Draw a checkbox with label and return its rect."""
        rect = pygame.Rect(x, y, box, box)
        pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)
        if checked:
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (x + 3, y + box // 2),
                (x + box // 2, y + box - 3),
                2,
            )
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (x + box // 2, y + box - 3),
                (x + box - 3, y + 3),
                2,
            )
        label = self.hud_font.render(label_text, True, (40, 40, 40))
        self.screen.blit(label, (rect.right + 8, y + (box - label.get_height()) // 2))
        return rect, rect.right + 8 + label.get_width()

    def draw_hud(self):
        pygame.draw.rect(self.screen, HUD_BG_COLOR, self.hud_rect)
        box = self.hud_height - 8
        y = self.hud_rect.top + 4

        # Weights checkbox
        x = 10
        self.toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_weights, "Weights"
        )

        # Arrows checkbox
        x = next_x + 15
        self.arrows_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_arrows, "Arrows"
        )

        # Forbidden Zone checkbox
        x = next_x + 15
        self.forbidden_zone_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_forbidden_zone, "Forbidden"
        )

        # Fireline Grid checkbox
        x = next_x + 15
        self.fireline_grid_toggle_rect, _ = self._draw_checkbox(
            x, y, box, self.show_fireline_grid, "Fireline"
        )

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_w:
            self.show_weights = not self.show_weights
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if hasattr(self, "toggle_rect") and self.toggle_rect.collidepoint(e.pos):
                self.show_weights = not self.show_weights
            elif hasattr(self, "forbidden_zone_toggle_rect") and self.forbidden_zone_toggle_rect.collidepoint(e.pos):
                self.show_forbidden_zone = not self.show_forbidden_zone

    def load_firebot_sprite(self, path: str = "assets/firebot.png"):
        """Load and scale the firebot sprite."""
        sprite = pygame.image.load(path).convert_alpha()
        # Scale to 3x3 cells
        sprite_size = 3 * self.cell_size
        self.firebot_sprite = pygame.transform.smoothscale(
            sprite, (sprite_size, sprite_size)
        )
        # Store original for rotation
        self.firebot_sprite_original = self.firebot_sprite.copy()

    def render_firebot(self, firebot):
        """Render the firebot at its current position with the correct rotation."""
        if not hasattr(self, "firebot_sprite"):
            self.load_firebot_sprite()

        # Convert cell coordinates to screen coordinates
        # Robot center in screen coords
        screen_x = self.field_rect.left + firebot.x * self.cell_size
        screen_y = self.field_rect.top + firebot.y * self.cell_size

        # Sprite faces DOWN (+y screen), theta=0 means facing RIGHT (+x).
        # Pygame rotates CCW with positive angles.
        # Subtract 90° to align sprite's down direction with theta=0 (right).
        angle_deg = -math.degrees(firebot.theta) - 90
        rotated = pygame.transform.rotate(self.firebot_sprite_original, angle_deg)

        # Get rect centered on robot position
        rect = rotated.get_rect(center=(screen_x, screen_y))

        self.screen.blit(rotated, rect)

    def render_firebot_footprint(self, firebot, color=(0, 100, 255), alpha=80):
        """Render the robot's 3x3 cell footprint as a transparent overlay."""
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        for row, col in firebot.get_footprint_cells():
            if 0 <= row < self.rows and 0 <= col < self.cols:
                x = col * cell
                y = row * cell
                pygame.draw.rect(
                    overlay,
                    (*color, alpha),
                    pygame.Rect(x + 2, y + 2, cell - 4, cell - 4),
                )
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_target_marker(self, target_x: float, target_y: float, color=(0, 200, 0)):
        """Render a target marker at the given cell coordinates."""
        screen_x = self.field_rect.left + target_x * self.cell_size
        screen_y = self.field_rect.top + target_y * self.cell_size

        # Draw crosshair
        size = self.cell_size // 2
        pygame.draw.line(
            self.screen,
            color,
            (screen_x - size, screen_y),
            (screen_x + size, screen_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            color,
            (screen_x, screen_y - size),
            (screen_x, screen_y + size),
            2,
        )
        pygame.draw.circle(
            self.screen, color, (int(screen_x), int(screen_y)), size // 2, 2
        )

    def screen_to_cell(
        self, screen_x: int, screen_y: int
    ) -> tuple[float, float] | None:
        """Convert screen coordinates to cell coordinates, or None if outside the field."""
        if not self.field_rect.collidepoint(screen_x, screen_y):
            return None

        cell_x = (screen_x - self.field_rect.left) / self.cell_size
        cell_y = (screen_y - self.field_rect.top) / self.cell_size

        return cell_x, cell_y

    def is_cell_visible(self, row: int, col: int, firebot) -> bool:
        """Check if a cell is within the firebot's sensor radius."""
        # Calculate distance from cell center to firebot position
        cell_center_x = col + 0.5
        cell_center_y = row + 0.5
        dx = cell_center_x - firebot.x
        dy = cell_center_y - firebot.y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= firebot.sensor_radius

    def render_fog_of_war(self, firebot):
        """
        Render fog of war overlay that obscures areas outside the sensor radius.
        Uses a circular reveal centered on the firebot's position.
        """
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        # Fill with fog color
        overlay.fill((*FOG_OF_WAR_COLOR, FOG_OF_WAR_ALPHA))

        # Create a circular mask to reveal visible area
        # Convert firebot position to pixel coordinates (relative to field)
        center_x = firebot.x * cell
        center_y = firebot.y * cell
        radius_px = firebot.sensor_radius * cell

        # Draw a transparent circle to reveal the visible area
        pygame.draw.circle(
            overlay,
            (0, 0, 0, 0),  # Fully transparent
            (int(center_x), int(center_y)),
            int(radius_px),
        )

        self.screen.blit(overlay, self.field_rect.topleft)

    def render_sensor_radius_outline(self, firebot, color=(100, 200, 255), width=2):
        """Render an outline showing the sensor radius."""
        center_x = self.field_rect.left + firebot.x * self.cell_size
        center_y = self.field_rect.top + firebot.y * self.cell_size
        radius_px = firebot.sensor_radius * self.cell_size

        pygame.draw.circle(
            self.screen,
            color,
            (int(center_x), int(center_y)),
            int(radius_px),
            width,
        )

    def render_corridor(self, weight_grid, low_threshold=1.5):
        """Highlight the low-cost corridor in green."""
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        for r in range(self.rows):
            y = r * cell
            for c in range(self.cols):
                w = weight_grid[r, c]
                if np.isfinite(w) and w <= low_threshold:
                    x = c * cell
                    pygame.draw.rect(
                        overlay,
                        (0, 200, 0, 60),  # Green, semi-transparent
                        pygame.Rect(x, y, cell, cell),
                    )

        self.screen.blit(overlay, self.field_rect.topleft)

    def render_gradient_arrows(self, cost_grid, spacing: int = 1):
        """Draw arrows showing the gradient direction (downhill flow)."""
        arrow_color = (0, 0, 200)

        for row in range(spacing, self.rows - spacing, spacing):
            for col in range(spacing, self.cols - spacing, spacing):
                center_cost = cost_grid[row, col]
                if not np.isfinite(center_cost):
                    continue

                # Compute gradient
                left = cost_grid[row, col - 1] if np.isfinite(cost_grid[row, col - 1]) else center_cost + 10
                right = cost_grid[row, col + 1] if np.isfinite(cost_grid[row, col + 1]) else center_cost + 10
                up = cost_grid[row - 1, col] if np.isfinite(cost_grid[row - 1, col]) else center_cost + 10
                down = cost_grid[row + 1, col] if np.isfinite(cost_grid[row + 1, col]) else center_cost + 10

                # Gradient points downhill (toward lower cost)
                grad_x = -(right - left) / 2.0
                grad_y = -(down - up) / 2.0

                mag = math.sqrt(grad_x ** 2 + grad_y ** 2)
                if mag < 0.01:
                    continue

                # Normalize and scale for display
                grad_x /= mag
                grad_y /= mag
                arrow_len = self.cell_size * 1.5

                # Screen coordinates (as integers)
                start_x = int(self.field_rect.left + col * self.cell_size + self.cell_size // 2)
                start_y = int(self.field_rect.top + row * self.cell_size + self.cell_size // 2)
                end_x = int(start_x + grad_x * arrow_len)
                end_y = int(start_y + grad_y * arrow_len)

                # Arrow line
                pygame.draw.line(self.screen, arrow_color, (start_x, start_y), (end_x, end_y), 2)

                # Arrowhead
                angle = math.atan2(grad_y, grad_x)
                head_len = 5
                pygame.draw.line(
                    self.screen, arrow_color,
                    (end_x, end_y),
                    (int(end_x - head_len * math.cos(angle - 0.5)), int(end_y - head_len * math.sin(angle - 0.5))),
                    2
                )
                pygame.draw.line(
                    self.screen, arrow_color,
                    (end_x, end_y),
                    (int(end_x - head_len * math.cos(angle + 0.5)), int(end_y - head_len * math.sin(angle + 0.5))),
                    2
                )

    def render_path(self, path: list[tuple[int, int]], color=(0, 100, 255), width=3):
        """Draw a path as a connected line."""
        if len(path) < 2:
            return

        # Convert cell coordinates to screen coordinates
        points = []
        for row, col in path:
            x = self.field_rect.left + col * self.cell_size + self.cell_size // 2
            y = self.field_rect.top + row * self.cell_size + self.cell_size // 2
            points.append((x, y))

        pygame.draw.lines(self.screen, color, False, points, width)

        # Draw start and end markers
        if points:
            pygame.draw.circle(self.screen, (0, 200, 0), points[0], 6)  # Start: green
            pygame.draw.circle(self.screen, (200, 0, 0), points[-1], 6)  # Goal: red
