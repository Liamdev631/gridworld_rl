import torch
import torch.nn as nn

class SpikingFEASTLayer(nn.Module):
	def __init__(self, input_size, num_neurons, f_open: float = 1e-2, lr_thresh: float = 1e-2, lr_weights: float = 1e-2, lr: float = 1):
		super(SpikingFEASTLayer, self).__init__()
		self.input_size = input_size
		self.num_neurons = num_neurons
		self.f_open = f_open
		self.lr_thresh = lr_thresh
		self.lr_weights = lr_weights
		self.lr = lr
		
		# Initialize synaptic weights and thresholds
		self.weights = torch.randn(num_neurons, input_size)
		self.thresholds = torch.randn(num_neurons)

	def forward(self, x):
		# Calculate the Euclidean distance between input and weight vectors
		values = torch.norm(self.weights - x, dim=1)

		# Find the closest neuron based on the distances
		closest_neuron = torch.argmin(values)

		# Check eligibility for activation based on thresholds
		eligible_neurons = values <= self.thresholds
		has_eligible_neuron = torch.any(eligible_neurons)

		# If no neurons are eligible, expand the thresholds
		self.thresholds[~eligible_neurons] += self.f_open

		# Update weights and thresholds for the winning neuron
		delta_thresh = values[closest_neuron] - self.thresholds[closest_neuron]
		delta_weights = x - self.weights[closest_neuron]

		self.thresholds[closest_neuron] += delta_thresh * self.lr_thresh * self.lr
		self.weights[closest_neuron] += delta_weights * self.lr_weights * self.lr

		# Create a one-hot vector for the winning neuron
		closest_neuron = torch.argmin(values)
		activation = torch.zeros(self.num_neurons)
		activation[closest_neuron] = 1.0

		return activation

class SpikingFEASTNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, hidden_size2, output_size, lr: float = 1):
		super(SpikingFEASTNetwork, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.hidden_size2 = hidden_size2
		self.output_size = output_size
		self.lr = lr

		self.input_layer = SpikingFEASTLayer(input_size, hidden_size, lr=lr)
		self.hidden_layer = SpikingFEASTLayer(hidden_size, hidden_size2, lr=lr)

		self.fc1 = torch.nn.Linear(hidden_size2, output_size)
  
	def forward(self, x, reward):
		x = self.input_layer(x, reward)
		x = self.hidden_layer(x, reward)
		x = self.fc1(x)
		return x