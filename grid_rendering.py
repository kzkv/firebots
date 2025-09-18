import pygame
import numpy as np

WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0, 255,   0)
RED    = (255,  0,   0)
PURPLE = (102, 0, 204)
BLUE   = (102, 178, 255)
ORANGE = (255, 128, 0)

# Function to Render a the Grid with all associated items
def render_grid(grid, screen, cell_size):

    rows, cols = grid.shape
    screen.fill(WHITE)

    # Define some constants
    margin = 2                          
    tri_margin = 2
    circle_radius = cell_size // 2 - margin

    # Draw cells by value
    for r in range(rows):
        y = r * cell_size
        for c in range(cols):
            x = c * cell_size
            v = int(grid[r, c])

            if v == 0:
                # Empty: White Square
                pygame.draw.rect(screen, WHITE, (x, y, cell_size, cell_size))
            if v == 1:
                # Obstacle: Black Square
                pygame.draw.rect(screen, BLACK, (x, y, cell_size, cell_size))

            elif v == 2:
                # Hero: Green Circle
                cx = x + cell_size // 2
                cy = y + cell_size // 2
                pygame.draw.circle(screen, PURPLE, (cx, cy), circle_radius)

            elif v == 3:
                # Enemy: Red Triangle
                cx = x + cell_size // 2
                cy = y + cell_size // 2
                half = cell_size // 2 - tri_margin
                points = [
                    (cx, cy - half),   
                    (cx - half, cy + half),   
                    (cx + half, cy + half),
                ]
                pygame.draw.polygon(screen, RED, points)
                
            elif v == 4:
                # Goal: Solid Green Square
                pygame.draw.rect(screen, GREEN, (x, y, cell_size, cell_size))

            elif v == 5:
                # Path: Solid Blue Square
                pygame.draw.rect(screen, BLUE, (x, y, cell_size, cell_size))
            elif v == 6:
                # HUSK: Orange Triangle
                cx = x + cell_size // 2
                cy = y + cell_size // 2
                half = cell_size // 2 - tri_margin
                points = [
                    (cx, cy - half),   
                    (cx - half, cy + half),   
                    (cx + half, cy + half),
                ]
                pygame.draw.polygon(screen, ORANGE, points)

    # Draw grid lines
    w = cols * cell_size
    h = rows * cell_size
    for c in range(cols + 1):
        x = c * cell_size
        pygame.draw.line(screen, BLACK, (x, 0), (x, h), 1)
    for r in range(rows + 1):
        y = r * cell_size
        pygame.draw.line(screen, BLACK, (0, y), (w, y), 1)
