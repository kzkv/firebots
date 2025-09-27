# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro

import pygame

from fire_bitmap import load_fire_bitmap
from render import World

# 6x3' field, 72x36", scale is 1" -> 1'
cols = 72
rows = 36
cell_size = 30
hud_height = cell_size

pygame.init()
world = World(rows, cols, cell_size)

running = True

# Ingest fire bitmap
fire_surface, fire_cells = load_fire_bitmap("fire1.png", world.cols, world.rows)


while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    world.clear()
    world.fire_bitmap_overlay(fire_surface)
    world.render_grid()
    world.render_fire_cells(fire_cells)

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()
