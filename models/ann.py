from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNet(nn.Module):
    def __init__ (self, input_size, output_size):
        super(DeepQNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
