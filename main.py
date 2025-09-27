# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: ChatGPT, Junie Pro

import pygame

from render import World

# 6x3' field, 72x36", scale is 1" -> 1'
cols = 72
rows = 36
cell_size = 20
hud_height = cell_size

pygame.init()
world = World(rows, cols, cell_size)

running = True


while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    world.render_grid()

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()
