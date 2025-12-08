# Potential Field Visualization
# RBE 550, Firebots (course project)
#
# Interactive 3D depth chart of the potential field cost grid

import matplotlib.pyplot as plt
import numpy as np

from fire_bitmap import load_fire_bitmap
from obstacles import place_trees
from planning import (
    compute_fire_distance_field,
    rebuild_weight_grid,
)


def visualize_potential_field_3d(
    weight_grid: np.ndarray,
    title: str = "Potential Field Cost Surface",
    clamp_max: float = 50.0,
    downsample: int = 1,
):
    """
    Visualize the potential field as an interactive 3D surface plot.

    Args:
        weight_grid: Cost grid from rebuild_weight_grid()
        title: Plot title
        clamp_max: Clamp infinite/high values to this for visualization
        downsample: Downsample factor for large grids (1 = no downsampling)
    """
    # Prepare data - clamp infinities for visualization
    Z = weight_grid.copy()
    Z[~np.isfinite(Z)] = clamp_max
    Z = np.clip(Z, 0, clamp_max)

    # Downsample if needed
    if downsample > 1:
        Z = Z[::downsample, ::downsample]

    rows, cols = Z.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="terrain",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Cost", fontsize=12)

    # Labels
    ax.set_xlabel("X (cells)", fontsize=11)
    ax.set_ylabel("Y (cells)", fontsize=11)
    ax.set_zlabel("Cost", fontsize=11)
    ax.set_title(title, fontsize=14)

    ax.view_init(elev=30, azim=-45)

    z_range = (Z.max() - Z.min()) * 0.3
    ax.set_box_aspect([cols, rows, z_range])

    tick_spacing = 20
    ax.set_xticks(np.arange(0, cols + 1, tick_spacing))
    ax.set_yticks(np.arange(0, rows + 1, tick_spacing))
    ax.set_zticks(np.arange(0, rows, tick_spacing))

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Grid dimensions (same as main.py)
    COLS = 100
    ROWS = 60
    TREE_COUNT = 50

    # Create RNG for reproducibility
    rng = np.random.default_rng(seed=42)

    # Load fire bitmap (need a minimal pygame init for the surface)
    import pygame

    pygame.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)

    try:
        _, fire_grid, _ = load_fire_bitmap("fire2.png", COLS, ROWS)
    except FileNotFoundError:
        print("fire2.png not found, creating synthetic fire")
        fire_grid = np.zeros((ROWS, COLS), dtype=bool)
        # Create a simple fire region
        fire_grid[25:35, 60:80] = True

    pygame.quit()

    # Generate trees
    tree_grid = place_trees(COLS, ROWS, count=TREE_COUNT, rng=rng)

    # Compute fire distance field
    fire_distance = compute_fire_distance_field(fire_grid)

    # Build weight grid
    weight_grid = rebuild_weight_grid(fire_grid, fire_distance, tree_grid)

    # Print stats
    finite = weight_grid[np.isfinite(weight_grid)]
    print(f"Weight grid shape: {weight_grid.shape}")
    print(
        f"Finite values: min={finite.min():.2f}, max={finite.max():.2f}, mean={finite.mean():.2f}"
    )
    print(f"Infinite cells: {np.sum(~np.isfinite(weight_grid))}")

    # Visualize - interactive 3D surface
    visualize_potential_field_3d(
        weight_grid,
        title="Potential Field Cost Surface",
        clamp_max=40.0,
        downsample=1,
    )
    plt.show()
