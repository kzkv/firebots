# Michael Laks & Tom Kazakov
# RBE 550, Firebots (course project)
# AI usage: Chat GPT, Junie Pro

# This program contains functions and logic for motion planning with A*

from players import *
import networkx as nx


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
# Refactored to use NetworkX instead of custom A* using Junie Pro
def compute_A_star(grid, start, goal, dangerZones, dangerWeight):

    # Initialize a step cost
    base_step_cost = 1

    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    # Build a directed graph where moving into a neighbor costs
    # base_step_cost + dangerWeight * danger at the neighbor
    rows, cols = grid.shape
    G = nx.DiGraph()

    def is_walkable(rc):
        r, c = rc
        return grid[r, c] in (EMPTY, GOAL, PATH)

    # Add edges from every cell to walkable neighbors. This allows starting
    # from a HERO cell (not walkable) while only moving into walkable cells,
    # matching the original getNeighbors behavior.
    for r in range(rows):
        for c in range(cols):
            # Current cell can be any; neighbors must be walkable
            cur = (r, c)
            # Up
            if r > 0:
                nb = (r - 1, c)
                if is_walkable(nb):
                    cost = (
                        base_step_cost + int(dangerZones[nb[0], nb[1]]) * dangerWeight
                    )
                    G.add_edge(cur, nb, weight=cost)
            # Down
            if r < rows - 1:
                nb = (r + 1, c)
                if is_walkable(nb):
                    cost = (
                        base_step_cost + int(dangerZones[nb[0], nb[1]]) * dangerWeight
                    )
                    G.add_edge(cur, nb, weight=cost)
            # Left
            if c > 0:
                nb = (r, c - 1)
                if is_walkable(nb):
                    cost = (
                        base_step_cost + int(dangerZones[nb[0], nb[1]]) * dangerWeight
                    )
                    G.add_edge(cur, nb, weight=cost)
            # Right
            if c < cols - 1:
                nb = (r, c + 1)
                if is_walkable(nb):
                    cost = (
                        base_step_cost + int(dangerZones[nb[0], nb[1]]) * dangerWeight
                    )
                    G.add_edge(cur, nb, weight=cost)

    # If start or goal are not present as nodes, add them so astar can consider them.
    if start not in G:
        G.add_node(start)
    if goal not in G:
        G.add_node(goal)

    def manhattan(u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])

    try:
        path = nx.astar_path(G, start, goal, heuristic=manhattan, weight="weight")
        # Ensure the first is start and last is goal like original
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
