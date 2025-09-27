# Michael Laks
# 9/14/2025
# RBE 550

# Flatland Assignment

# Program Defines Cells for the Hero and Enemy

import numpy as np

# Make random random
rng = np.random.default_rng()

# Define cell constants
EMPTY = 0
TET = 1
HERO = 2
ENEMY = 3
GOAL = 4
PATH = 5
HUSK = 6


# Base Class for Grid Entities (Hero, Enemy, etc.)
class Entity:
    def __init__(self, r, c, cell_value):
        self.r = r
        self.c = c
        self.cell_value = cell_value  # integer value this entity writes to grid

    # Get Position
    def pos(self):
        return (self.r, self.c)

    # Move Entity on Grid
    def move_to(self, r, c, grid):
        # Clear Old Position
        grid[self.r, self.c] = EMPTY
        # Change Position
        self.r, self.c = r, c
        # Rewrite to Grid in New Pos
        grid[self.r, self.c] = self.cell_value

    # Husking Function
    def husk_self(self, grid):
        # Rewrite to Grid as Husk
        grid[self.r, self.c] = HUSK

    # Func to get random unoccupied cell
    def get_rand_cell(self, grid):
        empty = np.where(grid == 0)
        idx = rng.integers(0, len(empty[0]))
        row = empty[0][idx]
        col = empty[1][idx]
        return row, col

    # Func to Teleport
    def teleportHero(self, grid):
        empty = np.where(grid == 0)
        idx = rng.integers(0, len(empty[0]))
        row = empty[0][idx]
        col = empty[1][idx]
        self.move_to(row, col, grid)


# Hero Class
class Hero(Entity):
    def __init__(self, r, c):
        # Call base constructor with HERO cell value
        super().__init__(r, c, HERO)


# Enemy Class
class Enemy(Entity):
    def __init__(self, r, c):
        # Call base constructor with ENEMY cell value
        super().__init__(r, c, ENEMY)
