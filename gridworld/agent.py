# agent.py
import numpy as np
from gridworld.tile_types import TileType
from gridworld.world import World
import torch

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = ['up', 'down', 'left', 'right', 'interact']
        self.num_ticks_per_inference: int = 50
        self.vision_range = 2
        
    def step(self, action: str | int, world: World) -> float:
        if isinstance(action, int):
            action = self.actions[action]
        reward = 0.0
        if action == 'up':
            self.y = max(0, self.y - 1)
        elif action == 'down':
            self.y = min(world.grid.shape[0] - 1, self.y + 1)
        elif action == 'left':
            self.x = max(0, self.x - 1)
        elif action == 'right':
            self.x = min(world.grid.shape[1] - 1, self.x + 1)
        elif action == 'interact':
            if world.grid[self.y][self.x] == TileType.tree.value:
                world.grid[self.y][self.x] = TileType.log.value
                reward += 1.0
            else:
                reward -= 1.0 # Wasted a turn
        return reward
    
