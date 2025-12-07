# Exploration Map: Tracks discovered terrain
# RBE 550, Firebots (course project)
# Opus Pocus Coffee and Focus 

import numpy as np


class ExplorationMap:
    """
    Tracks what the robot has discovered through exploration.
    Separates "ground truth" from "robot's knowledge".
    """
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        
        # Which cells have been seen (within sensor radius at some point)
        self.explored = np.zeros((rows, cols), dtype=bool)
        
        # Discovered obstacles - only True when we've seen a tree there
        self.discovered_trees = np.zeros((rows, cols), dtype=bool)
        
        # Track newly discovered obstacles this frame (for replanning triggers)
        self.newly_discovered = np.zeros((rows, cols), dtype=bool)
    
    def update(
        self,
        robot_x: float,
        robot_y: float,
        sensor_radius: float,
        true_tree_grid: np.ndarray,
    ) -> bool:
        """
        Update exploration based on robot's current position and sensor range.
        Returns True if any new obstacles were discovered.
        """
        self.newly_discovered.fill(False)
        found_new = False
        
        # Only check cells within bounding box for efficiency
        min_col = max(0, int(robot_x - sensor_radius) - 1)
        max_col = min(self.cols, int(robot_x + sensor_radius) + 2)
        min_row = max(0, int(robot_y - sensor_radius) - 1)
        max_row = min(self.rows, int(robot_y + sensor_radius) + 2)
        
        radius_sq = sensor_radius * sensor_radius
        
        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                if self.explored[row, col]:
                    continue
                
                # Check if cell center is within sensor radius
                cell_center_x = col + 0.5
                cell_center_y = row + 0.5
                dx = cell_center_x - robot_x
                dy = cell_center_y - robot_y
                
                if dx * dx + dy * dy <= radius_sq:
                    self.explored[row, col] = True
                    
                    if true_tree_grid[row, col]:
                        self.discovered_trees[row, col] = True
                        self.newly_discovered[row, col] = True
                        found_new = True
        
        return found_new
    
    def get_known_obstacles(self) -> np.ndarray:
        """Get the grid of obstacles the robot knows about."""
        return self.discovered_trees
    
    def get_newly_discovered(self) -> np.ndarray:
        """Get obstacles discovered in the last update."""
        return self.newly_discovered
    
    def get_exploration_percentage(self) -> float:
        """Get percentage of map that has been explored."""
        return 100.0 * np.sum(self.explored) / (self.rows * self.cols)