import torch
import torch.nn as nn

class SpikingFEAST(nn.Module):
	def __init__(self, input_size, num_neurons, f_open=1e-2, lr_thresh = 1e-2, lr_weights = 1e-2, lr=1):
		super(SpikingFEAST, self).__init__()
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
		has_eligible_neuron = torch.any(eligible_neurons).float()

		# If no neurons are eligible, expand the thresholds
		self.thresholds += self.f_open * has_eligible_neuron

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
