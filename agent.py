# agent.py
import numpy as np
from rstdp import RSTDPNetwork
from tile_types import TileType
from world import World
import torch

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = ['up', 'down', 'left', 'right', 'interact']
        self.learning_rate = 1e-1
        self.num_ticks_per_inference: int = 50
        
        self.visual_field: torch.Tensor = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.float32)
        
        self.input_size = self.visual_field.numel()
        self.hidden_size = 20
        self.output_size = len(self.actions)
        
        self.model: RSTDPNetwork = RSTDPNetwork(self.input_size, self.hidden_size, self.output_size, self.learning_rate)
        
    def update_visual_field(self, world: World) -> None:
        self.visual_field = torch.full((3, 3, len(TileType)), fill_value=TileType.empty.value, dtype=torch.float32)
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                if world.in_bounds(self.x + dx, self.y + dy):
                    tile = world.grid[self.y + dy, self.x + dx]
                else:
                    tile = TileType.empty.value
                self.visual_field[dy+1, dx+1][tile] = 1
        self.visual_field = self.visual_field.flatten()
        
    def step(self, action: str | int, world: World) -> float:
        if isinstance(action, int):
            action = self.actions[action]
        reward = 0.0
        if action == 'random':
            action = self.actions[int(torch.randint(0, len(self.actions)-1, size=(1,)))]
            
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
                
    def select_action(self, previous_reward: float = 0.0, p_random: float = 0.1, learning: bool = True) -> str:
        # Compute action and reward
        if torch.rand(1) < p_random:
            action = self.actions[int(torch.randint(0, len(self.actions), size=(1,)))]
        else:
            action = self.compute_action(previous_reward, learning)
        return action
    
    def compute_action(self, previous_reward: float = 0.0, learning: bool = False) -> str:
        # Reset spike counters 
        self.model.snn.input_layer.spike_count = self.model.snn.input_layer.spike_count.zero_()
        self.model.snn.hidden_layer.spike_count = self.model.snn.hidden_layer.spike_count.zero_()
        
        # Collect input spikes as long as required to ensure the response is accurate
        for _ in range(self.num_ticks_per_inference):
            input_spikes = torch.rand_like(self.visual_field).lt(self.visual_field * 0.5).float()
            self.model.forward(input_spikes, previous_reward)
        action_index: int = torch.argmax(self.model.snn.hidden_layer.spike_count).item() # type: ignore
        
        # Update the weights
        if learning:
            self.model.update_weights()
        
        return self.actions[action_index]
