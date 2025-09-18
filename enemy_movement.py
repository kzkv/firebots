# Michael Laks
# 9/14/2025
# RBE 550

# Flatland Assignment

from players import *
from a_star import getDangerNeighbors, heuristic, euclidean

# Program Defines Movement Functions for Enemies 

# Function to check next enemy step
# Return:
# 0 = Valid
# 99 = Not Valid
# 1 = Husked
# 2 = Captured Hero
def next_step_okay(grid, r, c, hero):

    rows, cols = grid.shape
    value = grid[r, c]
    # Check if step is within grid bounds
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 99 # Not Valid should never happen
    if (r, c) == hero.pos():
        return 2 # Captured Hero
    if (value == EMPTY) or (value == PATH) or (value == GOAL) or (value == ENEMY):
        return 0 # All Good (Valid)
    if (value == TET) or (value == HUSK):
        return 1 # Husk dat BOI
    

# Function to Choose Next Space for Enemy
def next_step(grid, enemy, hero):

    # Get Positions
    er, ec = enemy.pos()
    hr, hc = hero.pos()

    # Get Neighbor Cells to Move to
    next_cells = getDangerNeighbors(grid, enemy.pos())

    # Distance
    dist = float('inf')

    # Find Closest Cell to Hero
    for cell in next_cells:
        cur_dist = euclidean(cell, hero.pos())
        if cur_dist <= dist:
            dist = cur_dist
            best_step = cell

    return best_step


# Function to move the enemies and check if they've husked or caught the hero
def blindy_rage(grid, enemies, hero):

    # Build list of target cells
    enemy_targets = []
    reserved = set()

    # Did we catch the guy
    caught = False

    # Loop through our enemies and determine where to move them
    for idx, baddie in enumerate(enemies):
        
        # Get target for each enemy
        target = next_step(grid, baddie, hero)

        # Check outcome of movement
        tr, tc = target
        outcome = next_step_okay(grid, tr, tc, hero)

        # If movement is to a spot an enemy already is heading to then the second one there husks
        if outcome == 0 and (target in reserved):
            outcome = 1

        # Check all the outcomes and determine what do
        if outcome == 1:
            enemy_targets.append((idx, None, outcome)) # Husk
        elif outcome == 2:
            enemy_targets.append((idx, target, outcome)) # Caught Hero
            reserved.add(target)
            caught = True
        else:
            enemy_targets.append((idx, target, outcome)) # Just Move
            reserved.add(target)

    # Clear All Enemies From Map/Husk the ones that need Husking and move them
    updated_enemies = []
    for idx, baddie in enumerate(enemies):
        target, outcome = enemy_targets[idx][1], enemy_targets[idx][2]

        if outcome == 1 or target is None:
            # blocked -> turn into husk in place
            baddie.husk_self(grid)
            continue

        # valid or capture -> move (move_to clears old cell and writes new)
        tr, tc = target
        baddie.move_to(tr, tc, grid)
        updated_enemies.append(baddie)

    return updated_enemies, caught



    


        






















