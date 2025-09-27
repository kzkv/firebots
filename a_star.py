# Michael Laks
# 9/14/2025
# RBE 550

# Flatland Assignment

# This program contains functions and logic for motion planning with A*

import numpy as np
from queue import PriorityQueue
from players import *

# Utility Functions


# Define Function for our Heuristic
# Using Manhattan Distance
def heuristic(start, goal):
    manhattan_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return manhattan_distance


def euclidean(start, goal):
    dr = start[0] - goal[0]
    dc = start[1] - goal[1]
    return (dr * dr + dc * dc) ** 0.5


# Find All Neighbors
def getNeighbors(grid, cell):

    # Get size so we can determine bounds
    rows, cols = grid.shape

    # Get row col of cell
    r = cell[0]
    c = cell[1]

    # List of Neighbors
    neighbors = []

    # Check for cell above (make sure its not out of bounds)
    if r > 0:
        # Check that cell is empty or goal
        if (
            (grid[r - 1, c] == EMPTY)
            or (grid[r - 1, c] == GOAL)
            or (grid[r - 1, c] == PATH)
        ):
            neighbors.append((r - 1, c))
    # Check for cell down
    if r < rows - 1:
        # Check that cell is empty or goal
        if (
            (grid[r + 1, c] == EMPTY)
            or (grid[r + 1, c] == GOAL)
            or (grid[r + 1, c] == PATH)
        ):
            neighbors.append((r + 1, c))
    # Check for cell left
    if c > 0:
        # Check that cell is empty or goal
        if (
            (grid[r, c - 1] == EMPTY)
            or (grid[r, c - 1] == GOAL)
            or (grid[r, c - 1] == PATH)
        ):
            neighbors.append((r, c - 1))
    # Check for cell right
    if c < cols - 1:
        # Check that cell is empty or goal
        if (
            (grid[r, c + 1] == EMPTY)
            or (grid[r, c + 1] == GOAL)
            or (grid[r, c + 1] == PATH)
        ):
            neighbors.append((r, c + 1))

    return neighbors


# Find Danger Neighbors
def getDangerNeighbors(grid, cell):

    # Get size so we can determine bounds
    rows, cols = grid.shape

    # Get row col of cell
    r = cell[0]
    c = cell[1]

    # List of Neighbors
    neighbors = []

    # Check for cell above
    if r > 0:
        neighbors.append((r - 1, c))
    # Check for cell down
    if r < rows - 1:
        neighbors.append((r + 1, c))
    # Check for cell left
    if c > 0:
        neighbors.append((r, c - 1))
    # Check for cell right
    if c < cols - 1:
        neighbors.append((r, c + 1))

    return neighbors


# Function to get cells within a Manhattan radius around a center cell
def getCellRadius(grid, start, radius):
    start = (start[0], start[1])

    # Map of cell -> distance from start (0..radius)
    rings = {start: 0}

    # Frontier = cells at the current "ring" we are expanding from
    frontier = set([start])

    # Expand outward radius times
    for d in range(1, radius + 1):
        next_frontier = set()
        # For every cell in the current ring
        for cell in frontier:
            # Get its 4-neighbors (already in-bounds)
            for neighbor in getDangerNeighbors(grid, cell):
                # If we haven't recorded this neighbor yet, assign distance d
                if neighbor not in rings:
                    rings[neighbor] = d
                    next_frontier.add(neighbor)
        # Move to the next ring
        frontier = next_frontier
        # Early exit if there is nothing more to expand
        if not frontier:
            break

    return rings  # dict: {(r,c): distance}


# We are going to build a danger grid that will create danger zones around enemies
# we will run this at the start of the run to try to work around enemies
# we will then check the distance to enemies at every time step
# we will then rerun this and a* if the enemies get to close so we can plan around them
# Highway to the danger zone, please don't ride into the danger zone
def buildDangerZones(grid, enemies):
    # Make grid the same size as occ grid
    r, c = grid.shape
    dangerGrid = np.zeros((r, c), dtype=np.uint16)
    dangerRadius = 8
    maxDanger = 10
    dangerDecay = maxDanger / float(dangerRadius)

    # Loop through the enemies and build danger zones
    for baddie in enemies:
        row, col = baddie.pos()
        cell = (row, col)
        # Get cells within the radius and assign them decending danger scores
        dist_map = getCellRadius(dangerGrid, cell, dangerRadius)

        # Add a penalty that decreases with distance
        for (rr, cc), d in dist_map.items():
            # Linear falloff: max at d=0, down to ~0 at d=radius
            raw = maxDanger - dangerDecay * d
            score = int(round(raw)) if raw > 0 else 0
            if score > 0:
                dangerGrid[rr, cc] += score

    return dangerGrid


# Function to return path
def buildPath(parents, goal):
    path = [goal]
    current = goal
    while current in parents:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


# Function to overlay path onto grid
def paintPath(path, grid):
    grid[grid == PATH] = EMPTY
    for r, c in path[1:-1]:
        if grid[r, c] == EMPTY:
            grid[r, c] = PATH


# Function to Perform A* Search and Create a Path
def compute_A_star(grid, start, goal, dangerZones, dangerWeight):

    # Initalize a step cost
    base_step_cost = 1

    start = tuple(start)
    goal = tuple(goal)

    # Initalize our cost so far score (g score)
    g = {start: 0}

    # Initalize Parents list, which will track connections
    #   parents[child] = parent_cell (cell that child was added from)
    parents = {}

    # Create Priority Queue for the frontier ordered by f = g + h
    frontier_queue = PriorityQueue()

    # Add the start cell to the frontier
    fs = g[start] + heuristic(start, goal)
    frontier_queue.put((fs, start))

    # Loop for A*
    # While there is still more frontier to explore
    while not frontier_queue.empty():

        # Pop the cell with lowest f value from the frontier
        f, current = frontier_queue.get()

        # Check if this cell is our goal
        if current == goal:
            path = buildPath(parents, goal)
            return path

        # Look at all the cells neighbors
        for next in getNeighbors(grid, current):

            # Update Danger Cost
            dr, dc = next
            danger_cost = int(dangerZones[dr, dc]) * dangerWeight

            new_g_cost = g[current] + base_step_cost + danger_cost

            # If we havent seen the neighbor cell, or found a better route to it
            if next not in g or new_g_cost < g[next]:
                # Save cost so far to this cell
                g[next] = new_g_cost
                # Generate priority cost
                priority_cost = new_g_cost + heuristic(next, goal)
                # Add next to parent map to be able to breadcrumb the path together
                parents[next] = current
                # Push next into the frontier
                frontier_queue.put((priority_cost, next))

    # If we cant find a path
    itbroke = []
    return itbroke
