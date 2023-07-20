import torch
from torch.distributions import Categorical
from abc import abstractmethod

from gridworld.agent import Agent
from gridworld.world import World

def select_action_stochastically(expected_rewards):
    transformed_rewards = expected_rewards - torch.min(expected_rewards) # Shift rewards to be non-negative
    probabilities = torch.softmax(transformed_rewards, dim=0)
    action_dist = Categorical(probabilities)
    action = action_dist.sample()
    return action

class ExplorationMethod:
    def __init__(self, agent: Agent, world: World, device) -> None:
        self.agent = agent
        self.world = world
        self.device = device
        pass

    @abstractmethod
    def get_expected_rewards(self, policy: torch.nn.Module) -> torch.Tensor:
        pass
    
class TrueRandom(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device) -> None:
        super().__init__(agent, world, device)
        
    def get_expected_rewards(self, policy: torch.nn.Module) -> torch.Tensor:
        rewards = torch.zeros(len(self.agent.possible_actions))
        action_index = torch.randint(0, len(self.agent.possible_actions), (1,), device=self.device)
        rewards[action_index] = 1.0
        return action_index

class Greedy(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device) -> None:
        super().__init__(agent, world, device)
    
    def get_expected_rewards(self, policy: torch.nn.Module) -> torch.Tensor:
        rewards = torch.zeros(len(self.agent.possible_actions))
        visual_field = self.world.get_visual_field_at_agent(self.agent).flatten().unsqueeze(dim=0).to(self.device)
        inventory = self.agent.inventory.flatten().unsqueeze(dim=0).to(self.device)
        return policy(visual_field, inventory)

class EpsilonGreedy(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device, epsilon: float) -> None:
        super().__init__(agent, world, device)
        self.epsilon = epsilon
        
    def get_expected_rewards(self, policy: torch.nn.Module) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            rewards = torch.zeros(len(self.agent.possible_actions))
            action_index = torch.randint(0, len(self.agent.possible_actions), (1,), device=self.device)
            rewards[action_index] = 1.0
            return rewards
        
        visual_field = self.world.get_visual_field_at_agent(self.agent).flatten().unsqueeze(dim=0).to(self.device)
        inventory = self.agent.inventory.flatten().unsqueeze(dim=0).to(self.device)
        return policy(visual_field, inventory)
