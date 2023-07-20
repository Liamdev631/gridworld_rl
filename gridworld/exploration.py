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
    def choose_action(self, policy: torch.nn.Module) -> torch.Tensor:
        pass
    
class TrueRandom(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device) -> None:
        super().__init__(agent, world, device)
        
    def choose_action(self, policy: torch.nn.Module) -> torch.Tensor:
        action = torch.randint(0, len(self.agent.possible_actions), (1,), device=self.device)
        return action

class Greedy(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device) -> None:
        super().__init__(agent, world, device)
    
    def choose_action(self, policy: torch.nn.Module) -> torch.Tensor:
        visual_field = self.world.get_visual_field_at_agent(self.agent).flatten().unsqueeze(dim=0).to(self.device)
        inventory = self.agent.inventory.flatten().unsqueeze(dim=0).to(self.device)
        expected_rewards = policy(visual_field, inventory)
        action_probs = torch.softmax(expected_rewards, dim=1)
        action = torch.multinomial(action_probs, 1)
        return action

class EpsilonGreedy(ExplorationMethod):
    def __init__(self, agent: Agent, world: World, device, epsilon: float) -> None:
        super().__init__(agent, world, device)
        self.epsilon = epsilon
        self.random_exploration = TrueRandom(agent, world, device)
        self.greedy_exploration = Greedy(agent, world, device)
        
    def choose_action(self, policy: torch.nn.Module) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            return self.random_exploration.choose_action(policy)
        else:
            return self.greedy_exploration.choose_action(policy)
