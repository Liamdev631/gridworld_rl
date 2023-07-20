import torch
from gridworld.agent import Agent

class ReplayMemory:
    def __init__(self, agent: Agent, device) -> None:
        self.device = device
        self.agent = agent
        self.observations = torch.zeros((0, (agent.vision_range * 2 + 1)**2), dtype=torch.int32)
        self.actions = torch.zeros((0, 1), dtype=torch.int32)
        self.inventories = torch.zeros((0, agent.inventory.shape[0]), dtype=torch.int32)
        self.rewards = torch.zeros((0, 1), dtype=torch.float32)
        
    def add_experience(self, observation: torch.Tensor, actions: torch.Tensor, inventories: torch.Tensor, rewards: torch.Tensor, discount_factor: float = 0) -> None:
        # Compute mean and standard deviation of undiscounted rewards
        mean_reward = rewards.mean()
        std_reward = rewards.std()

        # Perform reward normalization
        if std_reward != 0:  # To avoid division by zero
            rewards = (rewards - mean_reward) / std_reward
        if discount_factor > 0:
            rewards = _compute_discounted_rewards(rewards, discount_factor)

        self.observations = torch.cat([self.observations, observation])
        self.actions = torch.cat([self.actions, actions])
        self.inventories = torch.cat([self.inventories, inventories])
        self.rewards = torch.cat([self.rewards, rewards])
    
    def make_batch(self, batch_size: int):
        indices = torch.randint(0, len(self.observations), size=(batch_size,))
        return (
            self.observations[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.inventories[indices].to(self.device),
            self.rewards[indices].to(self.device)
        )
    
    def get_all_batches(self, batch_size: int):
        # Shuffle the data
        idx = torch.randperm(self.observations.shape[0])
        self.observations = self.observations[idx]
        self.actions = self.actions[idx]
        self.inventories = self.inventories[idx]
        self.rewards = self.rewards[idx]
        
        # Yield batches
        num_batches = len(self.observations) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            yield (
                self.observations[start_idx:end_idx].to(self.device),
                self.actions[start_idx:end_idx].to(self.device),
                self.inventories[start_idx:end_idx].to(self.device),
                self.rewards[start_idx:end_idx].to(self.device)
            )
    
def _compute_discounted_rewards(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
    num_steps = len(rewards)
    discounted_rewards = torch.zeros((num_steps, 1))
    cumulative_reward = 0

    for t in reversed(range(num_steps)):
        cumulative_reward = rewards[t] + discount_factor * cumulative_reward
        discounted_rewards[t] = cumulative_reward

    return discounted_rewards
