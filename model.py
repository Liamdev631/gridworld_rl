import numpy as np

class InputLayer:
    def __init__(self, num_neurons, input_dim, theta_open, learning_rate, exploration_prob, decay_rate, final_exploration_prob, gamma, td_eta, td_decay_rate):
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.theta = np.random.rand(num_neurons)
        self.weights = np.random.randn(num_neurons, input_dim)
        self.theta_open = theta_open
        self.learning_rate = learning_rate
        self.exploration_prob = exploration_prob
        self.decay_rate = decay_rate
        self.final_exploration_prob = final_exploration_prob
        self.gamma = gamma
        self.td_eta = td_eta
        self.td_decay_rate = td_decay_rate
        self.actor_weights = np.zeros((num_neurons, num_neurons))
        self.critic_weights = np.zeros(num_neurons)
        self.actor_trace = np.zeros((num_neurons, num_neurons))
        self.critic_trace = np.zeros(num_neurons)
    
    def calculate_values(self, input_vector):
        return np.linalg.norm(self.weights - input_vector, axis=1)
    
    def select_winner_neuron(self, values) -> int:
        eligible_neurons = values <= self.theta
        if np.any(eligible_neurons):
            min_activated_neuron = np.argmin(values[eligible_neurons])
            return int(min_activated_neuron)
        else:
            min_activated_neuron = np.argmin(values)
            return int(min_activated_neuron)
    
    def adapt_weights(self, winner_neuron, input_vector):
        delta_theta = self.calculate_values(input_vector) - self.theta
        delta_weights = input_vector - self.weights
        self.theta += self.learning_rate * delta_theta * (winner_neuron == np.arange(self.num_neurons))
        self.weights += self.learning_rate * delta_weights * (winner_neuron == np.arange(self.num_neurons)).reshape(-1, 1)
    
    def adapt_thresholds(self):
        self.theta += self.theta_open
    
    def select_action(self, input_vector) -> int:
        if np.random.rand() > self.exploration_prob:
            values = self.calculate_values(input_vector)
            winner_neuron = self.select_winner_neuron(values)
            return winner_neuron
        else:
            return np.random.choice(self.num_neurons)
    
    def adapt_actor_critic(self, action: int, td_error):
        self.actor_trace *= self.td_decay_rate
        self.actor_trace[action] += 1.0
        self.critic_trace *= self.td_decay_rate
        self.critic_trace += 1.0
        
        self.actor_weights += self.actor_trace * self.td_eta * td_error
        self.critic_weights += self.critic_trace * self.td_eta * td_error
        
        self.theta += self.td_eta * np.abs(td_error) * self.actor_trace[action]
        self.weights += self.td_eta * np.abs(td_error) * self.actor_weights @ self.critic_trace
    
    def adapt(self, action, input_vector, reward):
        values = self.calculate_values(input_vector)
        winner_neuron = self.select_winner_neuron(values)
        self.adapt_weights(winner_neuron, input_vector)
        if np.all(~(values <= self.theta)):
            self.adapt_thresholds()
        
        td_error = reward + self.gamma * self.critic_weights @ self.calculate_values(input_vector) - self.critic_weights @ self.critic_trace
        self.adapt_actor_critic(action, td_error)
        
        self.exploration_prob = max(self.exploration_prob * self.decay_rate, self.final_exploration_prob)
