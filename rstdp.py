import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingNeuron(nn.Module):
    def __init__(self, in_features, out_features, tau=20):
        super(SpikingNeuron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau

        self.register_buffer('membrane_potential', torch.zeros(out_features))
        self.register_buffer('spike', torch.zeros(out_features))
        self.register_buffer('spike_count', torch.zeros(out_features))

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_spikes):
        psc = input_spikes.float() @ self.weight
        self.membrane_potential = (1.0 - 1.0 / self.tau) * self.membrane_potential + psc
        self.spike = self.membrane_potential > 1.0
        self.membrane_potential.masked_fill_(self.spike, 0.0)
        self.spike_count += self.spike.float()
        return self.spike
    
    def reset(self):
        self.membrane_potential = self.membrane_potential.zero_()
        self.spike = self.spike.zero_()
        self.spike_count = self.spike_count.zero_()

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = SpikingNeuron(input_size, hidden_size)
        self.hidden_layer = SpikingNeuron(hidden_size, output_size)

    def forward(self, input_spike):
        hidden_spike = self.input_layer(input_spike)
        output_spike = self.hidden_layer(hidden_spike)
        return output_spike

    def reset(self):
        self.input_layer.reset()
        self.hidden_layer.reset()

class RSTDPNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, reward_decay=0.99, spike_decay=0.01):
        super(RSTDPNetwork, self).__init__()
        self.lr = lr
        self.reward_decay = reward_decay
        self.spike_decay = spike_decay

        self.snn = SpikingNeuralNetwork(input_size, hidden_size, output_size)
        self.register_buffer('reward', torch.zeros(1))

    def forward(self, input_spike, reward):
        self.reward = self.reward_decay * self.reward + reward
        return self.snn(input_spike)

    def update_weights(self):
        torch.add(self.snn.input_layer.weight, self.lr * (self.reward * self.snn.input_layer.spike_count - self.spike_decay * self.snn.input_layer.weight))
        torch.add(self.snn.hidden_layer.weight, self.lr * (self.reward * self.snn.hidden_layer.spike_count - self.spike_decay * self.snn.hidden_layer.weight))

    def reset(self):
        self.snn.reset()
        self.reward = torch.zeros_like(self.reward)
