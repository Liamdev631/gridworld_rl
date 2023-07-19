# agent.py
import numpy as np
from gridworld.tile_types import TileType
from gridworld.world import World
import torch

carryable_objects : list[int] = [TileType.log.value]
MAX_INVENTORY_SIZE = 10

class Agent:
    def __init__(self, name: str, possible_actions: list[str] = []):
        self.name = name
        self.possible_actions = possible_actions
        self.x = 0
        self.y = 0
        self.vision_range = 2
        self.inventory: torch.Tensor = torch.zeros(size=(len(carryable_objects), 1), dtype=torch.int32)
        
    def step(self, action: str | int, world: World) -> float:
        return 0.0
    
    def attempt_pickup(self, world: World) -> bool:
        tile_under_foot: int = int(world.grid[self.y, self.x].item())
        if tile_under_foot in carryable_objects and self.inventory.sum() < MAX_INVENTORY_SIZE:
            return False
        inventory_slot: int = carryable_objects.index(tile_under_foot)
        self.inventory[inventory_slot] += 1
        world.grid[self.y, self.x] = TileType.empty.value
        return True
    
    def attempt_drop(self, world: World) -> bool:
        tile_under_foot: int = int(world.grid[self.y, self.x].item())
        if tile_under_foot != TileType.empty.value or self.inventory[tile_under_foot] <= 0:
            return False
        inventory_slot: int = carryable_objects.index(tile_under_foot)
        self.inventory[inventory_slot] -= 1
        world.grid[self.x, self.y] = tile_under_foot
        return True
        
    
class LoggerAgent(Agent):
    def __init__(self):
        super().__init__(name='Logger', possible_actions=['up', 'down', 'left', 'right', 'chop'])
        
    def step(self, action: str | int, world: World) -> float:
        super().step(action, world)
        if isinstance(action, int):
            action = self.possible_actions[action]
        reward = 0.0
        if action == 'up':
            self.y = max(0, self.y - 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'down':
            self.y = min(world.grid.shape[0] - 1, self.y + 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'left':
            self.x = max(0, self.x - 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'right':
            self.x = min(world.grid.shape[1] - 1, self.x + 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'chop':
            if world.grid[self.y][self.x] == TileType.tree.value:
                world.grid[self.y][self.x] = TileType.log.value
                reward += 10.0
            else:
                reward -= 2.0 # Wasted a turn
        return reward
    
class SweeperAgent(Agent):
    def __init__(self):
        super().__init__(name='Sweeper', possible_actions=['up', 'down', 'left', 'right', 'pickup', 'drop'])
        
    def step(self, action: str | int, world: World):
        super().step(action, world)
        if isinstance(action, int):
            action = self.possible_actions[action]
        reward = 0.0
        if action == 'up':
            self.y = max(0, self.y - 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'down':
            self.y = min(world.grid.shape[0] - 1, self.y + 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'left':
            self.x = max(0, self.x - 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'right':
            self.x = min(world.grid.shape[1] - 1, self.x + 1)
            reward -= 1 # Metabolic cost of moving
        elif action == 'pickup':
            if self.attempt_pickup(world):
                reward += 10.0
            else:
                reward -= 2.0
        elif action == 'drop':
            reward -= 2.0
