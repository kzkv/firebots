"""
Tree sprite source: https://github.com/jube/slowtree
"""

import math
import hashlib
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

        self.show_weights = True
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
        rows, cols = tree_grid.shape
        # colored overlay
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

                # Deterministic index from (seed,row,col)
                key = f"{row},{col}".encode()
                digest = hashlib.blake2b(key, digest_size=8).digest()  # 64-bit
                idx = int.from_bytes(digest, "little") % len(scaled)

                # Center 3x3 sprite on the cell (offset by 1 cell up/left)
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

    def draw_hud(self):
        pygame.draw.rect(self.screen, HUD_BG_COLOR, self.hud_rect)
        # simple checkbox + label
        box = self.hud_height - 8
        x, y = 10, self.hud_rect.top + 4
        self.toggle_rect = pygame.Rect(x, y, box, box)
        pygame.draw.rect(self.screen, (40, 40, 40), self.toggle_rect, 1)
        if self.show_weights:
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
        label = self.hud_font.render(
            f"Weights {'ON' if self.show_weights else 'OFF'}", True, (40, 40, 40)
        )
        self.screen.blit(
            label, (self.toggle_rect.right + 8, y + (box - label.get_height()) // 2)
        )

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_w:
            self.show_weights = not self.show_weights
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if hasattr(self, "toggle_rect") and self.toggle_rect.collidepoint(e.pos):
                self.show_weights = not self.show_weights

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
