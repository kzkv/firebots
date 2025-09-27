# Michael Laks
# 9/9/2025
# RBE 550

# Program that can be imported and create tetromino fields

import numpy as np
import random as rnd

# Define cell constants
EMPTY = 0
HERO = 2
ENEMY = 3
GOAL = 4

# Create Tetromino Shapes (Ignoring Duplicate Rotation Shapes (Z,J))


I = [[1], [1], [1], [1]]

I = np.array(I)

L = [[1, 1], [0, 1], [0, 1]]
L = np.array(L)

O = [[1, 1], [1, 1]]
O = np.array(O)

S = [[1, 0], [1, 1], [0, 1]]
S = np.array(S)

T = [[0, 1], [1, 1], [0, 1]]
T = np.array(T)

# These tetrominos don't know where they are
tetrominos = [I, L, O, S, T]


# Function to Randomize Tetromino and Rotation
def getTetromino():
    # Get Random Tetromino
    tet = rnd.choice(tetrominos)

    # Randomize rotation of Tetromino
    tet = np.rot90(tet, rnd.randint(0, 3))

    return tet


# Function to grab random tetromino and placement location
def getValidTetPlacement(tet, grid):
    isValid = False
    rows, cols = grid.shape
    height, width = tet.shape

    while isValid == False:

        # Get random index to place tetromino
        r_row = np.random.randint(0, rows - height + 1)
        r_col = np.random.randint(0, cols - width + 1)

        # Cut slice to check if empty
        tet_region = grid[r_row : r_row + height, r_col : r_col + width]

        # Make sure all indexes to be filled by tet are empty
        if (tet_region[tet == 1] == 0).all():
            isValid = True

    # Return Valid Coords
    return r_row, r_col


# Placement Helper Function
def place_tetromino(tet, row, col, grid):
    height, width = tet.shape
    slice = grid[row : row + height, col : col + width]
    slice[tet == 1] = 1


# Get Valid Hero/Enemy/Goal Location
# Func to get random unoccupied cell
def getRandCell(grid):
    empty = np.where(grid == 0)
    idx = np.random.randint(0, len(empty[0]))
    row = empty[0][idx]
    col = empty[1][idx]
    return row, col


# Place Actors/Goal on Grid
# Actor Values:
# 2 = hero
# 3 = enemy
# 4 = goal
def placeInitalEntity(grid, actor):
    r, c = getRandCell(grid)
    grid[r, c] = actor
    return [r, c]


# From here we need to start adding tetrominos
def generateTetField(rows, cols, density, seed=None):
    # Check args
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1]")

    # Determine if to use seed or not
    if seed is not None:
        rnd.seed(seed)
        np.random.seed(seed)

    # Basic Grid
    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Density Value will be translated to number of grid spaces
    filled_spaces = 0
    filled_goal = (rows * cols) * density

    # loop and add tetrominos
    while filled_spaces < filled_goal:
        # Get a random tetromino
        tet = getTetromino()

        # Get valid placement coords
        row, col = getValidTetPlacement(tet, grid)

        # Place tetromio at valid coords
        place_tetromino(tet, row, col, grid)

        # Update filled spaces count
        filled = np.sum(tet)
        filled_spaces = filled_spaces + filled

    # Place Hero, Goal, and Entities into Field
    hero = placeInitalEntity(grid, HERO)
    goal = placeInitalEntity(grid, GOAL)
    enemies = [placeInitalEntity(grid, ENEMY) for _ in range(70)]

    return grid, hero, goal, enemies
