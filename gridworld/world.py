# world.py
import numpy as np
from gridworld.tile_types import TileType

GRID_SIZE = 10

instance_counts = {
    TileType.tree.value: 10,
    TileType.log.value: 0
}

class World:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = GRID_SIZE
        self.grid = np.full((self.grid_size, self.grid_size), fill_value=TileType.empty.value, dtype=int)
        self.reset()
    
    def reset(self):
        self.grid.fill(TileType.empty.value)
        for tile_type, count in instance_counts.items():
            for _ in range(count):
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)
                self.grid[y, x] = tile_type

    def is_complete(self):
        if np.count_nonzero(self.grid == TileType.tree.value) == 0:
            return True
        return False
    
    def in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
