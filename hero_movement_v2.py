import numpy as np
from a_star import compute_A_star
from players import *


def hero_cell_walkable(grid, rc):
    r, c = rc
    return grid[r, c] in (EMPTY, GOAL, PATH)


def in_danger(dangerGrid, rc, threshold=5):
    r, c = rc
    return int(dangerGrid[r, c]) > threshold


def about_to_be_captured(grid, enemies, target_rc):
    tr, tc = target_rc

    if grid[tr, tc] in (TET, HUSK):
        return False
    for e in enemies:
        er, ec = e.pos() if hasattr(e, "pos") else (int(e[0]), int(e[1]))
        if abs(er - tr) + abs(ec - tc) == 1:
            return True
    return False


def heroically_rage(
    grid,
    hero,
    dangerGrid,
    path,
    goal,
    teleports,
    safe_steps,
    enemies,
    replan_every=5,
    danger_weight=1.0,
    danger_threshold=5,
    max_teleports=5,
):

    weWon = False
    goal = (goal[0], goal[1])

    if hero.pos() == goal:
        return True, path, safe_steps, teleports

    if not path or len(path) == 0 or path[0] != hero.pos():
        path = compute_A_star(grid, hero.pos(), goal, dangerGrid, danger_weight)
        if not path or len(path) < 2:
            if teleports < max_teleports:
                teleports += 1
                hero.teleportHero(grid)
                return False, None, 0, teleports
            return False, path, safe_steps, teleports

    next_step = path[1]

    if about_to_be_captured(grid, enemies, next_step):
        if teleports < max_teleports:
            teleports += 1
            hero.teleportHero(grid)
            return False, None, 0, teleports

    if not hero_cell_walkable(grid, next_step):
        replanned = compute_A_star(grid, hero.pos(), goal, dangerGrid, danger_weight)
        if replanned and len(replanned) > 1 and hero_cell_walkable(grid, replanned[1]):
            path = replanned
            next_step = path[1]
            safe_steps = 0
        else:

            if teleports < max_teleports:
                teleports += 1
                hero.teleportHero(grid)
                return False, None, 0, teleports
            return False, path, safe_steps, teleports

    elif in_danger(dangerGrid, next_step, threshold=danger_threshold):
        replanned = compute_A_star(grid, hero.pos(), goal, dangerGrid, danger_weight)
        if replanned and len(replanned) > 1:
            path = replanned
            next_step = path[1]
        safe_steps = 0

    else:
        safe_steps += 1
        if safe_steps >= replan_every:
            replanned = compute_A_star(
                grid, hero.pos(), goal, dangerGrid, danger_weight
            )
            if replanned and len(replanned) > 1:
                path = replanned
                next_step = path[1]
            safe_steps = 0

    hr, hc = next_step
    hero.move_to(hr, hc, grid)

    if next_step == goal:
        return True, path, safe_steps, teleports

    return False, path[1:], safe_steps, teleports
