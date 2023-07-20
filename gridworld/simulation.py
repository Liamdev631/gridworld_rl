import torch
from gridworld.exploration import ExplorationMethod
from gridworld.agent import Agent
from gridworld.world import World


def simulate_episode(agent: Agent, world: World, policy: torch.nn.Module, exploration_method: ExplorationMethod, device, max_duration: int) \
    -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    agent.x, agent.y = world.get_random_coordinates()
    observations = torch.zeros((max_duration, (agent.vision_range * 2 + 1)**2), dtype=torch.int32)
    actions = torch.zeros((max_duration, 1), dtype=torch.int32)
    inventories = torch.zeros((max_duration, agent.inventory.shape[0]), dtype=torch.int32)
    rewards = torch.zeros(max_duration, 1, dtype=torch.float32)
    
    t = 0
    for t in range(max_duration):
        visual_field = world.get_visual_field_at_agent(agent).flatten().to(device)
        action = exploration_method.get_expected_rewards(policy)
        
        # Update the agent's state
        observations[t] = visual_field
        actions[t] = action
        inventories[t] = agent.inventory
        rewards[t] = agent.step(int(action.item()), world)
        
    return observations[:t], actions[:t], inventories[:t], rewards[:t], t