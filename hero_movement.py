# Michael Laks
# 9/14/2025
# RBE 550

# Flatland Assignment

import numpy as np
from a_star import compute_A_star, buildDangerZones
from players import *

# Program Defines Movement Functions for Hero

# Move the hero 1 step along path
def heroically_rage(grid, hero, dangerZones, path, goal, safe_steps):

    weWon = False
    goal = (goal[0], goal[1]) 

    # Check if next step is goal
    if hero.pos() == goal:
        weWon = True
        return weWon, path, safe_steps

    # Get first location in path
    possibleNextStep = path[1] if len(path) > 1 else path[0]

    # If the next step is the goal, move and win
    if possibleNextStep == goal:
        tr, tc = possibleNextStep
        hero.move_to(tr, tc, grid)
        return True, path, safe_steps
    
    if not hero_cell_walkable(grid, possibleNextStep):
        replanned = compute_A_star(grid, hero.pos(), goal, dangerZones, 1)
        if replanned and len(replanned) > 1 and hero_cell_walkable(grid, replanned[1]):
            path = replanned
            possibleNextStep = path[1]
        else:
            # still blocked or no path; don't move this tick
            return False, path, safe_steps
    
    # Check saftey of next step
    if onHighwayToDangerZone(dangerZones, possibleNextStep) == True:
        # If unsafe then recompute A*
        replanned = compute_A_star(grid, hero.pos(), goal, dangerZones, 1)
        if replanned and len(replanned) > 1:
            path = replanned
            possibleNextStep = path[1]  # take the next step of the new plan
        safe_steps = 0
    else:
        safe_steps += 1
        replanned = []
        if safe_steps >= 5:
            replanned = compute_A_star(grid, hero.pos(), goal, dangerZones, 1)
            if len(replanned) > 1:
                path = replanned
                possibleNextStep = path[1]
            safe_steps = 0    
    
    # Pop Next Step and move to
    tr, tc = possibleNextStep
    hero.move_to(tr,tc,grid)

    return weWon, path[1:], safe_steps


# Function to Determine if Next Step is into a danger zone
def onHighwayToDangerZone(dangerZone, nextStep):

    # Check if next step is in danger
    nr, nc = nextStep
    if dangerZone[nr, nc] > 5:
        return True

    return False

def hero_cell_walkable(grid, rc):
    r, c = rc
    return grid[r, c] in (EMPTY, GOAL, PATH)