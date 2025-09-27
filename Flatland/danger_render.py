import pygame

# FUNCTION CREATED BY CHATGPT TO HELP ME VISUALIZE MY DANGER ZONES
# https://chatgpt.com/s/t_68c79668ec40819189cda73f03ab6089


# Render Danger Overlay
#  - danger: 2D np.uint8 array (0..255) from buildDangerZones
#  - screen: your Pygame display surface
#  - cell_size: pixel size of each grid cell
#  - max_alpha: cap for transparency (0..255). Higher = less transparent
def render_danger_overlay(danger, screen, cell_size, max_alpha=180):
    # Get grid size from danger map
    rows, cols = danger.shape

    # Make a transparent surface to paint per-cell alpha
    overlay = pygame.Surface((cols * cell_size, rows * cell_size), pygame.SRCALPHA)

    # Avoid division by zero: if all zeros, just skip
    maxv = int(danger.max())
    if maxv == 0:
        return

    # Loop over every cell and draw a red rectangle with alpha proportional to danger
    for r in range(rows):
        for c in range(cols):
            val = int(danger[r, c])  # 0..255 danger
            if val == 0:
                continue  # skip empty cells for speed
            # Scale cell danger (0..maxv) -> alpha (0..max_alpha)
            alpha = int(max_alpha * (val / maxv))
            # Rectangle in screen coords for this cell
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            # Draw red with per-rect alpha on the overlay (RGBA)
            pygame.draw.rect(overlay, (255, 0, 0, alpha), rect)

    # Blit the translucent overlay onto the main screen
    screen.blit(overlay, (0, 0))
