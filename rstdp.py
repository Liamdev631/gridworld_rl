import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingNeuron(nn.Module):
    def __init__(self, in_features, out_features, tau=10, tau_trace=25):
        super(SpikingNeuron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.tau_trace = tau_trace

        self.membrane_potential = torch.zeros(out_features)
        self.spike = torch.zeros(out_features)
        self.spike_count = torch.zeros(out_features)

        self.eligibility_trace: torch.Tensor = torch.zeros(in_features, out_features)
        self.weights: torch.Tensor = torch.zeros(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        self.weights *= 0.1

    def forward(self, input_spikes):
        psc = input_spikes.float() @ self.weights
        self.membrane_potential = (1.0 - 1.0 / self.tau) * self.membrane_potential + psc
        self.spike = self.membrane_potential > 1.0
        self.membrane_potential.masked_fill_(self.spike, 0.0)
        self.eligibility_trace = (1.0 - 1.0 / self.tau_trace) * self.eligibility_trace + self.spike.float()
        self.spike_count += self.spike.float()

        return self.spike
    
    def reset(self):
        self.membrane_potential = self.membrane_potential.zero_()
        self.spike = self.spike.zero_()
        self.spike_count = self.spike_count.zero_()
        
    def reset_eligibility_traces(self):
        self.eligibility_trace = self.eligibility_trace.zero_()

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer: SpikingNeuron = SpikingNeuron(input_size, hidden_size)
        self.hidden_layer: SpikingNeuron = SpikingNeuron(hidden_size, output_size)

    def forward(self, input_spike):
        hidden_spike = self.input_layer(input_spike)
        output_spike = self.hidden_layer(hidden_spike)
        return output_spike

    def reset(self):
        self.input_layer.reset()
        self.hidden_layer.reset()
        
    def reset_eligibility_traces(self):
        self.input_layer.reset_eligibility_traces()
        self.hidden_layer.reset_eligibility_traces()

class RSTDPNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, tau_reward=25):
        super(RSTDPNetwork, self).__init__()
        self.lr = lr
        self.tau_reward = tau_reward

        self.snn = SpikingNeuralNetwork(input_size, hidden_size, output_size)
        self.reward = torch.zeros(1)

    def forward(self, input_spike, reward) -> torch.Tensor:
        self.reward = (1 - 1 / self.tau_reward) * self.reward + reward
        self.spikes = self.snn.forward(input_spike)
        return self.spikes

    def update_weights(self):
        self.snn.input_layer.weights += self.lr * self.reward * (self.snn.input_layer.eligibility_trace - self.snn.input_layer.weights)
        self.snn.hidden_layer.weights += self.lr * self.reward * (self.snn.hidden_layer.eligibility_trace - self.snn.hidden_layer.weights)

    def reset(self):
        self.snn.reset()
        self.reward.zero_()
        
    def reset_eligibility_traces(self):
        self.snn.reset_eligibility_traces()
