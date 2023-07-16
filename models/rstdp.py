import torch
import torch.nn as nn

class SpikingNeuron(nn.Module):
    def __init__(self,
            num_neurons: int = 1,
            tau_membrane: float = 20
        ):
        super(SpikingNeuron, self).__init__()
        self.num_neurons = num_neurons
        self.tau_membrane = tau_membrane
        self.threshold = torch.ones(num_neurons)
        self.membrane_potential = torch.zeros(num_neurons)
        self.spikes = torch.zeros(num_neurons)

    def forward(self, inputs):
        self.membrane_potential += inputs
        self.spikes = self.membrane_potential >= self.threshold
        self.membrane_potential[self.spikes] = 0
        self.membrane_potential = self.membrane_potential * (1 - 1 / self.tau_membrane)
        
        return self.spikes

class SNNLayer(nn.Module):
    def __init__(self, num_neurons, num_presynaptic_neurons, lr=1e-2, tau_trace = 20):
        super(SNNLayer, self).__init__()
        self.num_neurons = num_neurons
        self.num_presynaptic_neurons = num_presynaptic_neurons
        self.lr = lr
        self.tau_trace = tau_trace
        self.weights = torch.randn(num_neurons, num_presynaptic_neurons)
        self.neurons = SpikingNeuron(num_neurons)
        self.spiking_trace = torch.zeros(num_neurons)
        self.post_synaptic_potential = torch.zeros(num_neurons)

    def forward(self, inputs, spiking_trace : torch.Tensor | None):
        weighted_inputs = torch.matmul(self.weights, inputs.float())
        self.post_synaptic_potential += weighted_inputs
        self.neurons(self.post_synaptic_potential)
        
        # STDP
        if spiking_trace is torch.Tensor:
            time_difference = spiking_trace - self.spiking_trace.T
            self.weights += time_difference * self.neurons.spikes * self.lr
        self.spiking_trace[self.neurons.spikes] = 1
        self.spiking_trace = self.spiking_trace * (1 - 1 / self.tau_trace)
        
        return self.neurons.spikes

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNN, self).__init__()
        self.hidden_layer = SNNLayer(hidden_size, input_size)
        self.output_layer = SNNLayer(output_size, hidden_size)

    def forward(self, inputs):
        hidden_spikes = self.hidden_layer(inputs, spiking_trace=None)
        output_spikes = self.output_layer(hidden_spikes, self.hidden_layer.spiking_trace)
        return output_spikes
