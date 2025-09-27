# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: Chat GPT, Junie Pro

import pygame

pygame.init()
pygame.display.set_caption("Firebots")

# 6x3' field, 72x36", scale is 1" -> 1'
# Target width of a fire line is 3', 3 cells wide
field_width = 72
field_height = 36

# Fire will be defined by a higher-resolution bitmap (10 pixels per cell)
# Obstacle location will be randomly generated with a cell-based overlay
# Typical trees: 1', 2', or 3' square trunks, 3' "cross", s-shaped 4x4' "large tree"
# TODO: Boulders and relief will be added later and can be any shape
# TODO: Consider brush or flammables affecting fire propagation
# TODO: Consider different types of terrain with various movement cost

