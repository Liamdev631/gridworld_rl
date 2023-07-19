import torch
from torch.distributions import Categorical
from abc import abstractmethod

from gridworld.agent import Agent

def select_action_stochastically(expected_rewards):
    transformed_rewards = expected_rewards - torch.min(expected_rewards) # Shift rewards to be non-negative
    probabilities = torch.softmax(transformed_rewards, dim=0)
    action_dist = Categorical(probabilities)
    action = action_dist.sample()
    return action

class ExplorationMethod:
    def __init__(self, agent: Agent, device) -> None:
        self.agent = agent
        self.device = device
        pass

    @abstractmethod
    def select_action(self, policy: torch.nn.Module, visual_field: torch.Tensor) -> torch.Tensor:
        pass
    
class TrueRandom(ExplorationMethod):
    def __init__(self, agent: Agent, device) -> None:
        super().__init__(agent, device)
        
    def select_action(self, policy: torch.nn.Module, visual_field: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, len(self.agent.possible_actions), (1,), device=self.device)

class Greedy(ExplorationMethod):
    def __init__(self, agent: Agent, device) -> None:
        super().__init__(agent, device)
    
    def select_action(self, policy: torch.nn.Module, visual_field: torch.Tensor) -> torch.Tensor:
        visual_field = visual_field.unsqueeze(dim=0).to(self.device)
        rewards = torch.zeros(len(self.agent.possible_actions))
        for action_index in range(len(self.agent.possible_actions)):
            action_tensor = torch.tensor([action_index], device=self.device).unsqueeze(dim=0)
            rewards[action_index] = policy(visual_field, action_tensor)
        return select_action_stochastically(rewards)

class EpsilonGreedy(ExplorationMethod):
    def __init__(self, agent: Agent, device, epsilon: float) -> None:
        super().__init__(agent, device)
        self.epsilon = epsilon
        
    def select_action(self, policy: torch.nn.Module, visual_field: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, len(self.agent.possible_actions), (1,), device=self.device)
        else:
            visual_field = visual_field.unsqueeze(dim=0).to(self.device)
            rewards = torch.zeros(len(self.agent.possible_actions))
            for action in range(len(self.agent.possible_actions)):
                action_tensor = torch.tensor([action], device=self.device).unsqueeze(dim=0)
                rewards[action] = policy(visual_field, action_tensor)
            action = select_action_stochastically(rewards)
        return action
        
class EpsilonZGreedy(ExplorationMethod):
    def __init__(self, agent: Agent, device, epsilon: float, z: float) -> None:
        super().__init__(agent, device)
        self.device = device
        self.epsilon = epsilon
        self.z_duration = z
        self.z_action: torch.Tensor = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.z_timer = 0
        
    def select_action(self, policy: torch.nn.Module, visual_field: torch.Tensor, num_actions: int) -> torch.Tensor:
        if self.z_timer == 0:
            self.z_timer = self.z_duration
            if torch.rand(1) < self.epsilon:
                self.z_action = torch.randint(0, num_actions, (1,), device=self.device)
            else:
                visual_field = visual_field.unsqueeze(dim=0)
                rewards = torch.zeros(num_actions)
                for action_index in range(num_actions):
                    action_tensor = torch.tensor([action_index], device=self.device).unsqueeze(dim=0)
                    rewards[action_index] = policy(visual_field, action_tensor)
                self.z_action = select_action_stochastically(rewards)
        return self.z_action