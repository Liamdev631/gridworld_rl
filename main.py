# main.py
import numpy as np
from agent import Agent
from simulation import run_simulation_without_rendering, run_simulation_with_rendering
from world import GRID_SIZE
import matplotlib.pyplot as plt

num_epochs = 100

agent = Agent(GRID_SIZE)

epoch_performance= np.zeros(num_epochs)
for epoch in range(num_epochs):
    reward_history, frames = run_simulation_without_rendering(agent)
    epoch_performance[epoch] = frames

fig = plt.figure()
plt.plot(epoch_performance)

while True:
    run_simulation_with_rendering(agent)
