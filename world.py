# world.py
import numpy as np
from tile_types import TileType
from agent import Agent

GRID_SIZE = 10

def init_world():
    instance_counts = {
        TileType.tree.value: 10,
        TileType.log.value: 0
    }
    grid = np.full((GRID_SIZE, GRID_SIZE), fill_value=TileType.empty.value, dtype=int)
    for tile_type, count in instance_counts.items():
        for i in range(count):
            x = np.random.randint(GRID_SIZE)
            y = np.random.randint(GRID_SIZE)
            grid[y, x] = tile_type
    return grid

def step(agent, grid):
    visual_field = np.full((GRID_SIZE, GRID_SIZE), fill_value=TileType.empty.value, dtype=int)
    for x in range(agent.x - 1, agent.x + 1):
        for y in range(agent.y - 1, agent.y + 1):
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
                break
            visual_field[y, x] = grid[y, x]
    
    action = agent.select_action(visual_field)
    reward = agent.step(agent.actions[action], grid)
    agent.adapt(action, reward)
    return reward

def is_session_complete(grid):
    if np.count_nonzero(grid == TileType.tree.value) == 0:
        return True
    return False
