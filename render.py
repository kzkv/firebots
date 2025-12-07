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
TARGET_MARKER_COLOR = (0, 255, 255)  # Cyan - distinct from path markers
INSET = 2


class World:
    def __init__(self, rows, cols, cell_size):
        self.rows, self.cols = rows, cols
        self.field_width, self.field_height = cols * cell_size, rows * cell_size
        self.cell_size = cell_size
        self.hud_height = cell_size
        self.margin = cell_size

        self.window_width = self.field_width + 2 * self.margin
        self.window_height = self.field_height + 2 * self.margin + self.hud_height

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
        self.show_waypoints = True
        self.show_fog_of_war = True
        self.hud_font = pygame.font.SysFont(None, max(12, self.cell_size - 6))
        self.hud_rect = pygame.Rect(
            0, self.field_rect.bottom + self.margin, self.window_width, self.hud_height
        )

    def clear(self):
        self.screen.fill(CELL_BG_COLOR)

    def render_grid(self):
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

        hud_rect = pygame.Rect(
            0, self.field_rect.bottom + self.margin, self.window_width, self.hud_height
        )
        pygame.draw.rect(self.screen, HUD_BG_COLOR, hud_rect)

    def fire_bitmap_overlay(self, fire_bitmap, fire_bounds=None):
        """
        Render fire bitmap overlay.

        Args:
            fire_bitmap: The pygame surface with fire image
            fire_bounds: Optional (col_offset, row_offset, fire_cols, fire_rows)
                        If None, scales to full field
        """
        if fire_bounds is None:
            # Legacy behavior: scale to full field
            target_width = self.field_rect.width
            target_height = self.field_rect.height
            x = self.field_rect.left
            y = self.field_rect.top
        else:
            col_offset, row_offset, fire_cols, fire_rows = fire_bounds
            target_width = fire_cols * self.cell_size
            target_height = fire_rows * self.cell_size
            x = self.field_rect.left + col_offset * self.cell_size
            y = self.field_rect.top + row_offset * self.cell_size

        if fire_bitmap.get_size() != (target_width, target_height):
            fire_bitmap = pygame.transform.smoothscale(
                fire_bitmap, (target_width, target_height)
            )
        fire_bitmap.set_alpha(FIRE_BITMAP_ALPHA)
        self.screen.blit(fire_bitmap, (x, y))

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
                    pygame.draw.rect(
                        fire_overlay,
                        (*FIRE_CELL_COLOR, FIRE_CELL_ALPHA),
                        pygame.Rect(
                            x + INSET, y + INSET, cell - 2 * INSET, cell - 2 * INSET
                        ),
                    )
        self.screen.blit(fire_overlay, self.field_rect.topleft)

    def render_trees(self, tree_grid):
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
        if len(path) < 1:
            return

        overlay = pygame.Surface(self.field_rect.size)
        overlay.fill((0, 0, 0))
        overlay.set_colorkey((0, 0, 0))

        half_width = width_cells * self.cell_size / 2
        line_thickness = max(3, int(self.cell_size * 0.3))

        for x, y, theta in path:
            sx = x * self.cell_size
            sy = y * self.cell_size

            perp_x = -math.sin(theta) * half_width
            perp_y = math.cos(theta) * half_width

            left = (sx + perp_x, sy + perp_y)
            right = (sx - perp_x, sy - perp_y)

            pygame.draw.line(overlay, FIRELINE_PATH_COLOR, left, right, line_thickness)

        overlay.set_alpha(FIRELINE_PATH_ALPHA)
        self.screen.blit(overlay, self.field_rect.topleft)

    def render_weight_heatmap(self, weights, vmax=None):
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size
        finite = np.isfinite(weights)
        if vmax is None:
            vmax = np.percentile(weights[finite], 95) if finite.any() else 1.0
        vmax = max(vmax, 1e-6)

        for r in range(self.rows):
            y = r * cell
            for c in range(self.cols):
                w = weights[r, c]
                if not np.isfinite(w):
                    color = (0, 0, 0, 180)
                elif w <= 1.0:
                    continue
                else:
                    t = min(1.0, (w - 1.0) / vmax)
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
        mx, my = pygame.mouse.get_pos()
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

        pad = 4
        tip_rect = txt.get_rect().inflate(pad * 2, pad * 2)
        tip_rect.topleft = (mx + 12, my + 10)
        tip_rect.clamp_ip(self.screen.get_rect())

        tip = pygame.Surface(tip_rect.size, pygame.SRCALPHA)
        tip.fill((*bg, 230))
        pygame.draw.rect(tip, (*border, 220), tip.get_rect(), 1)
        self.screen.blit(tip, tip_rect.topleft)
        self.screen.blit(txt, txt.get_rect(center=tip_rect.center))

        cell_rect = pygame.Rect(
            self.field_rect.left + col * self.cell_size,
            self.field_rect.top + row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, border, cell_rect, 1)

    def _draw_checkbox(self, x, y, box, checked, label_text):
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

    def draw_hud(self, encirclement_state: str = None):
        pygame.draw.rect(self.screen, HUD_BG_COLOR, self.hud_rect)
        box = self.hud_height - 8
        y = self.hud_rect.top + 4

        x = 10
        self.toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_weights, "Weights"
        )

        x = next_x + 15
        self.arrows_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_arrows, "Arrows"
        )

        x = next_x + 15
        self.forbidden_zone_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_forbidden_zone, "Forbidden"
        )

        x = next_x + 15
        self.fireline_grid_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_fireline_grid, "Fireline"
        )

        x = next_x + 15
        self.waypoints_toggle_rect, next_x = self._draw_checkbox(
            x, y, box, self.show_waypoints, "Waypoints"
        )

        # Show encirclement state if provided
        if encirclement_state:
            state_colors = {
                "idle": (150, 150, 150),
                "active": (0, 150, 255),
                "complete": (0, 200, 0),
                "failed": (255, 50, 50),
            }
            color = state_colors.get(encirclement_state.lower(), (100, 100, 100))
            state_text = self.hud_font.render(f"Encircle: {encirclement_state.upper()}", True, color)
            self.screen.blit(state_text, (self.window_width - state_text.get_width() - 10, y + 2))

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_w:
            self.show_weights = not self.show_weights
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if hasattr(self, "toggle_rect") and self.toggle_rect.collidepoint(e.pos):
                self.show_weights = not self.show_weights
            elif hasattr(self, "forbidden_zone_toggle_rect") and self.forbidden_zone_toggle_rect.collidepoint(e.pos):
                self.show_forbidden_zone = not self.show_forbidden_zone
            elif hasattr(self, "waypoints_toggle_rect") and self.waypoints_toggle_rect.collidepoint(e.pos):
                self.show_waypoints = not self.show_waypoints

    def load_firebot_sprite(self, path: str = "assets/firebot.png"):
        sprite = pygame.image.load(path).convert_alpha()
        sprite_size = 3 * self.cell_size
        self.firebot_sprite = pygame.transform.smoothscale(
            sprite, (sprite_size, sprite_size)
        )
        self.firebot_sprite_original = self.firebot_sprite.copy()

    def render_firebot(self, firebot):
        if not hasattr(self, "firebot_sprite"):
            self.load_firebot_sprite()

        screen_x = self.field_rect.left + firebot.x * self.cell_size
        screen_y = self.field_rect.top + firebot.y * self.cell_size

        angle_deg = -math.degrees(firebot.theta) - 90
        rotated = pygame.transform.rotate(self.firebot_sprite_original, angle_deg)

        rect = rotated.get_rect(center=(screen_x, screen_y))

        self.screen.blit(rotated, rect)

    def render_firebot_footprint(self, firebot, color=(0, 100, 255), alpha=80):
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

    def render_target_marker(self, target_x: float, target_y: float, color=None):
        """Render a target marker (cyan crosshair) at the given cell coordinates."""
        if color is None:
            color = TARGET_MARKER_COLOR

        screen_x = self.field_rect.left + target_x * self.cell_size
        screen_y = self.field_rect.top + target_y * self.cell_size

        # Draw crosshair - larger and more visible than path markers
        size = self.cell_size
        thickness = 3
        pygame.draw.line(
            self.screen,
            color,
            (screen_x - size, screen_y),
            (screen_x + size, screen_y),
            thickness,
        )
        pygame.draw.line(
            self.screen,
            color,
            (screen_x, screen_y - size),
            (screen_x, screen_y + size),
            thickness,
        )
        pygame.draw.circle(
            self.screen, color, (int(screen_x), int(screen_y)), size // 2, thickness
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
        cell_center_x = col + 0.5
        cell_center_y = row + 0.5
        dx = cell_center_x - firebot.x
        dy = cell_center_y - firebot.y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= firebot.sensor_radius

    def render_fog_of_war(self, firebot):
        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        overlay.fill((*FOG_OF_WAR_COLOR, FOG_OF_WAR_ALPHA))

        center_x = firebot.x * cell
        center_y = firebot.y * cell
        radius_px = firebot.sensor_radius * cell

        pygame.draw.circle(
            overlay,
            (0, 0, 0, 0),
            (int(center_x), int(center_y)),
            int(radius_px),
        )

        self.screen.blit(overlay, self.field_rect.topleft)

    def render_sensor_radius_outline(self, firebot, color=(100, 200, 255), width=2):
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
                        (0, 200, 0, 60),
                        pygame.Rect(x, y, cell, cell),
                    )

        self.screen.blit(overlay, self.field_rect.topleft)

    def render_tree_repulsion(self, tree_grid, min_distance=1.5, max_radius=5):
        """
        Render the tree repulsion zones as colored overlays.

        Args:
            tree_grid: Boolean grid where True = tree
            min_distance: Cells within this distance are impassable (red)
            max_radius: Max radius of repulsion influence (gradient)
        """
        if not tree_grid.any():
            return

        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        # Compute distance from trees
        from planning import _chamfer_distance8
        dist_tree = _chamfer_distance8(tree_grid.astype(bool))

        for r in range(self.rows):
            y = r * cell
            for c in range(self.cols):
                d = dist_tree[r, c]
                x = c * cell

                if d < min_distance:
                    # Impassable zone - red
                    pygame.draw.rect(
                        overlay,
                        (255, 0, 0, 100),
                        pygame.Rect(x + 2, y + 2, cell - 4, cell - 4),
                    )
                elif d < max_radius:
                    # Repulsion zone - orange gradient fading out
                    t = (d - min_distance) / (max_radius - min_distance)
                    alpha = int(80 * (1 - t))
                    pygame.draw.rect(
                        overlay,
                        (255, 150, 0, alpha),
                        pygame.Rect(x + 2, y + 2, cell - 4, cell - 4),
                    )

        self.screen.blit(overlay, self.field_rect.topleft)

    def render_gradient_arrows(self, cost_grid, spacing: int = 1):
        arrow_color = (0, 0, 200)

        for row in range(spacing, self.rows - spacing, spacing):
            for col in range(spacing, self.cols - spacing, spacing):
                center_cost = cost_grid[row, col]
                if not np.isfinite(center_cost):
                    continue

                left = cost_grid[row, col - 1] if np.isfinite(cost_grid[row, col - 1]) else center_cost + 10
                right = cost_grid[row, col + 1] if np.isfinite(cost_grid[row, col + 1]) else center_cost + 10
                up = cost_grid[row - 1, col] if np.isfinite(cost_grid[row - 1, col]) else center_cost + 10
                down = cost_grid[row + 1, col] if np.isfinite(cost_grid[row + 1, col]) else center_cost + 10

                grad_x = -(right - left) / 2.0
                grad_y = -(down - up) / 2.0

                mag = math.sqrt(grad_x ** 2 + grad_y ** 2)
                if mag < 0.01:
                    continue

                grad_x /= mag
                grad_y /= mag
                arrow_len = self.cell_size * 1.5

                start_x = int(self.field_rect.left + col * self.cell_size + self.cell_size // 2)
                start_y = int(self.field_rect.top + row * self.cell_size + self.cell_size // 2)
                end_x = int(start_x + grad_x * arrow_len)
                end_y = int(start_y + grad_y * arrow_len)

                pygame.draw.line(self.screen, arrow_color, (start_x, start_y), (end_x, end_y), 2)

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

        points = []
        for row, col in path:
            x = self.field_rect.left + col * self.cell_size + self.cell_size // 2
            y = self.field_rect.top + row * self.cell_size + self.cell_size // 2
            points.append((x, y))

        pygame.draw.lines(self.screen, color, False, points, width)

        # Draw start marker (green) - but NOT end marker to avoid confusion with target
        if points:
            pygame.draw.circle(self.screen, (0, 200, 0), points[0], 6)  # Start: green
            # Removed red end marker - target_marker shows the actual goal

    def render_encirclement_waypoints(
        self,
        waypoints: list[tuple[float, float]],
        states: list[str],
        current_idx: int = -1,
    ):
        """
        Render encirclement waypoints with state-based colors.

        Args:
            waypoints: List of (x, y) waypoint positions
            states: List of states ("visited", "current", "pending", "skipped")
            current_idx: Index of current waypoint (for highlighting)
        """
        if not waypoints or not self.show_waypoints:
            return

        # Colors for different states
        colors = {
            "visited": (0, 200, 0),      # Green - completed
            "current": (255, 255, 0),    # Yellow - active target
            "pending": (100, 150, 255),  # Light blue - upcoming
            "skipped": (150, 150, 150),  # Gray - skipped
        }

        # Draw connecting lines first (behind waypoints)
        if len(waypoints) >= 2:
            points = []
            for wx, wy in waypoints:
                sx = int(self.field_rect.left + wx * self.cell_size)
                sy = int(self.field_rect.top + wy * self.cell_size)
                points.append((sx, sy))

            # Draw line connecting all waypoints
            pygame.draw.lines(self.screen, (100, 100, 200), False, points, 2)

            # Draw closing line if it's a loop (first and last waypoints are same or very close)
            if len(waypoints) > 2:
                first = waypoints[0]
                last = waypoints[-1]
                dx = first[0] - last[0]
                dy = first[1] - last[1]
                if math.sqrt(dx*dx + dy*dy) < 2.0:  # Close enough to be a loop
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 200),
                        points[-1],
                        points[0],
                        2
                    )

        # Draw waypoint markers
        for i, (wx, wy) in enumerate(waypoints):
            sx = int(self.field_rect.left + wx * self.cell_size)
            sy = int(self.field_rect.top + wy * self.cell_size)

            state = states[i] if i < len(states) else "pending"
            color = colors.get(state, (200, 200, 200))

            # Size based on state
            if state == "current":
                radius = 8
                # Draw pulsing ring for current waypoint
                pygame.draw.circle(self.screen, color, (sx, sy), radius + 4, 2)
            elif state == "visited":
                radius = 5
            else:
                radius = 6

            # Draw filled circle
            pygame.draw.circle(self.screen, color, (sx, sy), radius)

            # Draw border
            border_color = (50, 50, 50) if state != "skipped" else (100, 100, 100)
            pygame.draw.circle(self.screen, border_color, (sx, sy), radius, 1)

            # Draw waypoint number
            if self.cell_size >= 12:
                num_text = self.tooltip_font.render(str(i), True, (0, 0, 0))
                text_rect = num_text.get_rect(center=(sx, sy - radius - 8))
                self.screen.blit(num_text, text_rect)

    def render_corridor_cells(
        self,
        corridor_cells: list[tuple[int, int]],
        color=(0, 255, 255),
        alpha=80,
    ):
        """
        Render the corridor cells that were found for encirclement.

        Args:
            corridor_cells: List of (row, col) positions
            color: RGB color for corridor cells
            alpha: Transparency (0-255)
        """
        if not corridor_cells:
            return

        overlay = pygame.Surface(self.field_rect.size, pygame.SRCALPHA)
        cell = self.cell_size

        for row, col in corridor_cells:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                x = col * cell
                y = row * cell
                pygame.draw.rect(
                    overlay,
                    (*color, alpha),
                    pygame.Rect(x + 2, y + 2, cell - 4, cell - 4),
                )

        self.screen.blit(overlay, self.field_rect.topleft)
