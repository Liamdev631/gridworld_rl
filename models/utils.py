import torch
from gridworld.agent import Agent

agent: Agent = Agent()

class ReplayMemory:
    def __init__(self, device) -> None:
        self.device = device
        self.observations = torch.zeros((0, (agent.vision_range * 2 + 1)**2), dtype=torch.int32).to(self.device)
        self.actions = torch.zeros((0, 1), dtype=torch.int32).to(self.device)
        self.rewards = torch.zeros((0, 1), dtype=torch.float32).to(self.device)
        
    def add_experience(self, observation: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, discount_factor: float = 0) -> None:
        if discount_factor > 0:
            rewards = _compute_discounted_rewards(rewards, discount_factor)
        self.observations = torch.cat((self.observations, observation.to(self.device)))
        self.actions = torch.cat((self.actions, actions.to(self.device)))
        self.rewards = torch.cat((self.rewards, rewards.to(self.device)))
    
    def make_batch(self, batch_size: int):
        indices = torch.randint(0, len(self.observations), size=(batch_size,))
        memory = ReplayMemory(self.device)
        memory.add_experience(self.observations[indices], self.actions[indices], self.rewards[indices])
        return memory
    
    def get_all_batches(self, batch_size: int):
        num_batches = len(self.observations) // batch_size
        for i in range(num_batches):
            memory = ReplayMemory(self.device)
            memory.add_experience(
                self.observations   [i * batch_size : (i + 1) * batch_size],
                self.actions        [i * batch_size : (i + 1) * batch_size],
                self.rewards        [i * batch_size : (i + 1) * batch_size]
            )
            yield memory
    
def _compute_discounted_rewards(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
    num_steps = len(rewards)
    discounted_rewards = torch.zeros((num_steps, 1))
    cumulative_reward = 0

    for t in reversed(range(num_steps)):
        cumulative_reward = rewards[t] + discount_factor * cumulative_reward
        discounted_rewards[t] = cumulative_reward

    return discounted_rewards
