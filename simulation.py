# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from world import init_world, step, is_session_complete, GRID_SIZE

max_frames = 1000

def run_simulation_with_rendering(agent: Agent):
    import pygame
    from tile_types import TileType
    grid = init_world()
    pygame.init()
    display_size = (16 * GRID_SIZE, 16 * GRID_SIZE)
    screen = pygame.display.set_mode(display_size, depth=32)
    pygame.display.set_caption('Rimworld AI')
    clock = pygame.time.Clock()
    
    sprite_filenames = {
        TileType.empty.value: 'assets/empty.png',
        TileType.tree.value: 'assets/tree.png',
        TileType.log.value: 'assets/log.png'
    }
    sprites = {}
    for tile_type, filename in sprite_filenames.items():
        try:
            sprites[tile_type] = pygame.image.load(filename).convert_alpha()
        except pygame.error as e:
            print(f"Error loading image: {filename}")
            print(str(e))
            pygame.quit()
            raise SystemExit
    agent_sprite = pygame.image.load('assets/agent.png').convert_alpha()

    done = False
    frame = 0
    reward_history = np.zeros(max_frames)
    
    while not done:
        reward = step(agent, grid)
        reward_history[frame] = reward
        
        screen.fill((255, 255, 255))
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                tile = grid[y, x]
                sprite = sprites[tile]
                screen.blit(sprite, (x * 16, y * 16))
                
        screen.blit(agent_sprite, (agent.x * 16, agent.y * 16))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        pygame.display.flip()
        pygame.display.update()
        clock.tick(30)
        
        frame += 1
        if frame == max_frames:
            done = True
        
        if is_session_complete(grid):
            done = True

    pygame.quit()

    return reward_history, frame

def run_simulation_without_rendering(agent: Agent):
    done = False
    frame = 0
    reward_history = np.zeros(max_frames)
    grid = init_world()
    
    while not done:
        reward = step(agent, grid)
        reward_history[frame] = reward
        
        frame += 1
        if frame == max_frames:
            done = True
        
        if is_session_complete(grid):
            done = True

    return reward_history, frame
