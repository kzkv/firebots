"""
Tree sprite source: https://github.com/jube/slowtree
"""

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
INSET = 2


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
            y_cell = row * self.cell_size
            for col in range(cols):
                if not tree_grid[row, col]:
                    continue
                x_cell = col * self.cell_size

                # Deterministic index from (seed,row,col)
                key = f"{row},{col}".encode()
                digest = hashlib.blake2b(key, digest_size=8).digest()  # 64-bit
                idx = int.from_bytes(digest, "little") % len(scaled)

                # Center 3x3 sprite on the cell (offset by 1 cell up/left)
                dest = (x_cell - self.cell_size, y_cell - self.cell_size)
                tree_sprite_overlay.blit(scaled[idx], dest)

        tree_sprite_overlay.set_alpha(TREE_SPRITE_ALPHA)
        self.screen.blit(tree_sprite_overlay, self.field_rect.topleft)
