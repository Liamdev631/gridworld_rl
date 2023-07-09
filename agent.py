# agent.py
import numpy as np
from tile_types import TileType
from model import InputLayer

class Agent:
    def __init__(self, grid_size):
        self.x = np.random.randint(grid_size)
        self.y = np.random.randint(grid_size)
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
        
        self.input_layer = InputLayer(num_neurons, 9 * len(TileType), theta_open, learning_rate, exploration_prob, decay_rate, final_exploration_prob, gamma, td_eta, td_decay_rate)
        
    def step(self, action, grid):
        reward = 0.0
        if action == 'up':
            self.y = max(0, self.y - 1)
        elif action == 'down':
            self.y = min(grid.shape[0] - 1, self.y + 1)
        elif action == 'left':
            self.x = max(0, self.x - 1)
        elif action == 'right':
            self.x = min(grid.shape[1] - 1, self.x + 1)
        elif action == 'interact':
            if grid[self.y][self.x] == TileType.tree.value:
                grid[self.y][self.x] = TileType.log.value
                reward += 1.0
        return reward
                
    def select_action(self, visual_field):
        visual_field = visual_field.flatten()
        in_1 = np.zeros((9, len(TileType)))
        for i in range(9):
            in_1[i, visual_field[i]] = 1.0
        in_1 = in_1.flatten()
        
        values = self.input_layer.calculate_values(in_1)
        winner_neuron = self.input_layer.select_winner_neuron(values)

        if np.random.rand() < self.p_random:
            # Randomly select an action
            action = np.random.choice(self.actions)
        else:
            # Select action from the network
            action = winner_neuron

        return action
    
    def update_reward(self, action, reward):
        self.input_layer.adapt_actor_critic(action, reward)
