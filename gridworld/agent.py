# agent.py
import numpy as np
from gridworld.tile_types import TileType
from gridworld.world import World
import torch

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = ['up', 'down', 'left', 'right', 'interact', 'random_move']
        self.learning_rate = 1e-1
        self.num_ticks_per_inference: int = 50
        
        self.visual_field: torch.Tensor = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.float32).flatten()
        
        self.input_size = self.visual_field.numel()
        self.hidden_size = 20
        self.output_size = len(self.actions)
        
    def update_visual_field(self, world: World) -> torch.Tensor:
        self.visual_field = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.float32)
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                if world.in_bounds(self.x + dx, self.y + dy):
                    tile = world.grid[self.y + dy, self.x + dx]
                else:
                    tile = TileType.empty.value
                self.visual_field[dy+1, dx+1][tile] = 1
        self.visual_field = self.visual_field.flatten()
        return self.visual_field
        
    def step(self, action: str | int, world: World) -> float:
        if isinstance(action, int):
            action = self.actions[action]
        reward = 0.0
        if action == 'random_move':
            action = self.actions[int(torch.randint(self.actions.index('up'), self.actions.index('right'), size=(1,)))]
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
        return reward
    
