import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import InputLayer

input_dim = 2
num_neurons = 10
theta_open = 1e-4
learning_rate = 1e-3
num_epochs = 1000

input_layer = InputLayer(num_neurons, input_dim, theta_open, learning_rate)
input_layer.weights *= 3

# Simulated dataset with three probability density functions (PDF)
dataset = np.random.randn(100, input_dim) * 0.2
dataset[:33] += 2
dataset[33:66] -= 2

# Create the figure and axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Adaptation of Input Layer Neurons')
ax.set_aspect('equal', adjustable='box')

# Initialize the scatter plot for weights and circles for thresholds
weights_scatter = ax.scatter([], [], c='r', marker='o')
dataset_scatter = ax.scatter([], [], c='b', marker='o', alpha=0.2)
circles = [plt.Circle((0, 0), 0, color='b', fill=False) for _ in range(num_neurons)]

# Function to update the scatter plot and circles
def update(frame):
    for observation in dataset:
        input_layer.adapt(observation)

    weights_scatter.set_offsets(input_layer.weights)
    dataset_scatter.set_offsets(dataset)
    for i in range(num_neurons):
        circles[i].center = input_layer.weights[i]
        circles[i].radius = input_layer.theta[i]
        ax.add_patch(circles[i])

# Create the animation
animation = FuncAnimation(fig, update, frames=range(num_epochs), interval=100, repeat=False)

# Display the animation
plt.show()
