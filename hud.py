# Michael Laks
# 9/15/2025
# RBE 550
#
# Simple HUD bar for: steps, enemies, husks, teleports

import pygame
import numpy as np
from players import HUSK

# Height of the HUD bar 
HUD_H = 48

# Count husks on the grid
def count_husks(grid):
    return int(np.count_nonzero(grid == HUSK))


def draw_hud(screen, grid_w, grid_h, steps, enemies_count, husks_count, teleports):
    # Colors
    DARK  = (30, 30, 30)
    LIGHT = (210, 210, 210)
    WHITE = (255, 255, 255)

    # HUD background at the bottom
    hud_rect = pygame.Rect(0, grid_h, grid_w, HUD_H)
    pygame.draw.rect(screen, DARK, hud_rect)

    # Separator line
    pygame.draw.line(screen, LIGHT, (0, grid_h), (grid_w, grid_h), 1)

    # Build label texts
    items = [
        f"Steps: {int(steps)}",
        f"Enemies: {int(enemies_count)}",
        f"Husks: {int(husks_count)}",
        f"Teleports: {int(teleports)}",
    ]

    # Render left-to-right
    font = pygame.font.SysFont(None, 22)
    x = 10
    y = grid_h + (HUD_H - font.get_height()) // 2
    for s in items:
        surf = font.render(s, True, WHITE)
        screen.blit(surf, (x, y))
        x += surf.get_width() + 24
