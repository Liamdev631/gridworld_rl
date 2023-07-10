# agent.py
import numpy as np
from tile_types import TileType
from model import InputLayer
from world import World
import torch

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = ['up', 'down', 'left', 'right', 'interact']
        self.p_random = 0.1
        
        num_neurons = len(self.actions)
        theta_open = 1e-3
        learning_rate = 1e-3
        exploration_prob = 1.0
        decay_rate = 0.99
        final_exploration_prob = 0.01
        gamma = 0.9
        td_eta = 1e-3
        td_decay_rate = 0.99
        
        self.visual_field: torch.Tensor = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.int32)
        self.input_layer = InputLayer(num_neurons, 9 * len(TileType), theta_open, learning_rate, exploration_prob, decay_rate, final_exploration_prob, gamma, td_eta, td_decay_rate)
        
    def update_visual_field(self, world: World) -> None:
        self.visual_field = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.int32)
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                if world.in_bounds(self.x + dx, self.y + dy):
                    tile = world.grid[self.y + dy, self.x + dx]
                else:
                    tile = TileType.empty.value
                self.visual_field[dy+1, dx+1][tile] = 1
        
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
        return reward
                
    def select_action(self, visual_field):
        values = self.input_layer.calculate_values(visual_field.flatten())
        action = self.input_layer.select_action(values)

        return action
    
    def update_reward(self, action, reward):
        self.input_layer.adapt_actor_critic(action, reward)
