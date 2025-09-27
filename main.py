# Michael Laks
# 9/11/2025
# RBE 550

# Flatland Assignment

# Imports
import numpy as np
import pygame
from gen_occ_grid import generateTetField
from players import *
from grid_rendering import render_grid
from a_star import *
from danger_render import *
from enemy_movement import *
from hero_movement_v2 import *
from hud import HUD_H, draw_hud, count_husks

# Initalize PyGame Session
pygame.init()
pygame.display.set_caption("Flatland 2: Electric Bugaloo")

# Make random random
rng = np.random.default_rng()

# Define Constants
grid_rows = 64
grid_cols = 64
cell_size = 18
grid_w = grid_rows * cell_size
grid_h = grid_cols * cell_size

game_over = False
outcome = ""  # "You win!" or "You were caught!"
waiting_to_start = True


def draw_game_over(screen, grid_w, grid_h, text):
    import pygame
    # dim the scene
    overlay = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))
    # text
    font_big = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 24)
    t1 = font_big.render(text, True, (255, 255, 255))
    t2 = font_small.render("Press ESC or close window to quit", True, (220, 220, 220))
    screen.blit(t1, (grid_w // 2 - t1.get_width() // 2, grid_h // 2 - t1.get_height()))
    screen.blit(t2, (grid_w // 2 - t2.get_width() // 2, grid_h // 2 + 8))


def draw_start_overlay(screen, grid_w, grid_h, text="Press SPACE to start"):
    overlay = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))
    font_big = pygame.font.SysFont(None, 44)
    font_small = pygame.font.SysFont(None, 24)
    t1 = font_big.render(text, True, (255, 255, 255))
    t2 = font_small.render("ESC or Q to quit", True, (220, 220, 220))
    screen.blit(t1, (grid_w // 2 - t1.get_width() // 2, grid_h // 2 - t1.get_height()))
    screen.blit(t2, (grid_w // 2 - t2.get_width() // 2, grid_h // 2 + 8))


# Create Screen
screen = pygame.display.set_mode((grid_w, grid_h + HUD_H))

# Initalize Game Vars and Helpers

# Start Game Clock
clock = pygame.time.Clock()
# Set Game to Running
running = True
# Set First Run Flag
firstRun = True

# Main Game Loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_SPACE and waiting_to_start:
                waiting_to_start = False
                steps_taken = 0

    if firstRun:
        grid, hero_rc, goal, enemy_rcs = generateTetField(grid_rows, grid_cols, density=.2, seed=68443)
        hero = Hero(hero_rc[0], hero_rc[1])
        enemies = [Enemy(r, c) for (r, c) in enemy_rcs]

        dangerZone = buildDangerZones(grid, enemies)
        path = compute_A_star(grid, hero.pos(), goal, dangerZone, 1)

        if path:
            paintPath(path, grid)

        safe_since = 0
        teleports = 0
        steps_taken = 0
        last_hero_pos = hero.pos()
        firstRun = False

    if not game_over:

        dangerZone = buildDangerZones(grid, enemies)

        if waiting_to_start and not game_over:
            render_grid(grid, screen, cell_size)
            render_danger_overlay(dangerZone, screen, cell_size)
            enemies_count = len(enemies)
            husks_count = count_husks(grid)
            draw_hud(screen, grid_w, grid_h, steps_taken, enemies_count, husks_count, teleports)
            draw_start_overlay(screen, grid_w, grid_h)
            pygame.display.flip()
            clock.tick(8)
            continue  # wait here until SPACE

        weWon, path, safe_since, teleports = heroically_rage(
            grid, hero, dangerZone, path, goal, teleports, safe_since, enemies)

        if weWon:
            outcome = "You win!"
            game_over = True
        else:

            enemies, caught = blindy_rage(grid, enemies, hero)
            if caught:
                outcome = "You were caught!"
                game_over = True

        if path:
            paintPath(path, grid)

        grid[goal[0], goal[1]] = GOAL

        new_pos = hero.pos()
        if new_pos != last_hero_pos:
            steps_taken += 1
            last_hero_pos = new_pos

        render_grid(grid, screen, cell_size)
        render_danger_overlay(dangerZone, screen, cell_size)

    else:

        render_grid(grid, screen, cell_size)

        render_danger_overlay(dangerZone, screen, cell_size)

        draw_game_over(screen, grid_w, grid_h, outcome)

    enemies_count = len(enemies)
    husks_count = count_husks(grid)
    draw_hud(screen, grid_w, grid_h, steps_taken, enemies_count, husks_count, teleports)

    pygame.display.flip()
    clock.tick(4)

pygame.quit()
