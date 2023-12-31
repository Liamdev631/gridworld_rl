from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gridworld.tile_types import TileType
from gridworld.agent import Agent

class DeepQNet(nn.Module):
    def __init__ (self, agent: Agent):
        super(DeepQNet, self).__init__()
        self.tile_embeddings_dim = 4
        self.tile_embeddings = nn.Embedding(num_embeddings=len(TileType), embedding_dim=self.tile_embeddings_dim)
        
        self.input_size = self.tile_embeddings_dim * (agent.vision_range * 2 + 1)**2 \
            + agent.inventory.shape[0]
        
        self.layer1 = nn.Linear(self.input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, len(agent.possible_actions))
        
    def forward(self, tiles, inventories) -> torch.Tensor:
        # print('tiles', tiles.shape)
        # print('actions', actions.shape)
        # print('inventories', inventories.shape)
        
        tiles_embedded = self.tile_embeddings(tiles).flatten(start_dim=1)
        
        x = torch.cat([tiles_embedded, inventories], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x
