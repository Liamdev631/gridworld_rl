import datetime
from gridworld.agent import Agent
from gridworld.tile_types import TileType
from gridworld.world import LoggerTrainingWorld
import pygame
import torch
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def run_simulation_with_rendering(policy: torch.nn.Module, device: torch.device, file_name: str = "file.gif", max_duration: int = 100):
    world = LoggerTrainingWorld()
    agent = Agent()
    frames = []
    
    pygame.init()
    display_size = (16 * world.grid_size, 16 * world.grid_size)
    screen = pygame.display.set_mode(display_size, depth=32)
    pygame.display.set_caption('Rimworld AI')
    clock = pygame.time.Clock()
    
    # Load the sprites
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

    # Load fonts
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 16)

    done = False
    frame = 0
    
    while not done:
        visual_field = world.get_visual_field_at_agent(agent).flatten().to(device)
        probs = torch.zeros(len(agent.actions)).to(device)
        for action in range(len(agent.actions)):
            probs[action] = policy(visual_field.unsqueeze(dim=0), torch.tensor([action]).to(device).unsqueeze(dim=0))
        #action = int(torch.argmax(rewards).cpu().item())
        prob_dist = Categorical(probs)
        action = int(prob_dist.sample().item())
        agent.step(action, world)
        
        screen.fill((255, 255, 255))
        
        # Render the tiles
        for x in range(world.grid_size):
            for y in range(world.grid_size):
                tile = world.grid[y, x]
                sprite = sprites[tile]
                screen.blit(sprite, (x * 16, y * 16))
                
        # Render the agent
        screen.blit(agent_sprite, (agent.x * 16, agent.y * 16))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        # Draw the frame number in the left upper corner
        text = font.render(f"Frame: {frame + 1}", True, (0, 0, 0))
        screen.blit(text, dest=(0, 0))
        
        pygame.display.flip()
        pygame.display.update()
        frames.append(np.copy(pygame.surfarray.array3d(screen).transpose(1, 0, 2)))
        #clock.tick(2)
        
        # End if time has elapsed
        frame += 1
        if frame == max_duration:
            done = True
        
        # End if all trees are gone
        if world.is_complete():
            done = True

    pygame.quit()
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    ax.axis("off")

    # Create the initial frame
    im = ax.imshow(frames[0])

    # Function to update the frame
    def update_frame(i):
        im.set_array(frames[i])
        return im,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=250, blit=True)

    # Save the animation as a GIF
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"policy_{timestamp}.gif"
    ani.save(file_name, writer="pillow")

    return frame