import torch
from gridworld.exploration import ExplorationMethod
from gridworld.agent import Agent
from gridworld.world import World


def simulate_episode(agent: Agent, world: World, policy: torch.nn.Module, exploration_method: ExplorationMethod, device, max_duration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    agent.x, agent.y = world.get_random_coordinates()
    observations = torch.zeros((max_duration, (agent.vision_range * 2 + 1)**2), dtype=torch.int32)
    actions = torch.zeros((max_duration, 1), dtype=torch.int32)
    rewards = torch.zeros(max_duration, 1, dtype=torch.float32)
    
    t = 0
    action: torch.Tensor = torch.zeros(1, dtype=torch.int32) 
    for t in range(max_duration):
        visual_field = world.get_visual_field_at_agent(agent).flatten().to(device)
        action = exploration_method.select_action(policy, visual_field, len(agent.actions))
        
        observations[t] = visual_field
        actions[t] = action
        rewards[t] = agent.step(int(action.item()), world)
        
    return observations[:t], actions[:t], rewards[:t], t