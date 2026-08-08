import numpy as np

class EpsilonGreedyAgent:
    """
    A Multi-Armed Bandit agent that balances exploration and exploitation.
    Chooses the best action most of the time but also explores other actions occasionally.
    """
    
    def __init__(self, options=10, epsilon=0.1, random_seed=None):
        self.options = options
        self.epsilon = epsilon
        self.action_counts = np.zeros(options)
        self.q_values = np.zeros(options)
        if random_seed is not None:
            np.random.seed(random_seed)  # For reproducibility if a seed is provided

    def choose_action(self):
        """
        Selects the action with the highest Q-value.
        Ties must be broken randomly.
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.options)
        else:
            # Exploit: choose the best known action
            indices_of_max = np.where(self.q_values == np.max(self.q_values))[0]  # Get indices of all max Q-values
            chosen_action = np.random.choice(indices_of_max)  # Randomly select one of the indices
            return chosen_action
        
    def update(self, action, reward):
        """
        Updates the action count and Q-value estimate for the chosen action.
        """
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        
    def reset(self, random_seed=None):
        """
        Wipes the agent's memory for a new episode.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.q_values = np.zeros(self.options)
        self.action_counts = np.zeros(self.options)