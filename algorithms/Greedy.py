import numpy as np

class GreedyAgent:
    """
    A purely exploitative Multi-Armed Bandit agent.
    Always chooses the action with the highest estimated value.
    """
    
    def __init__(self, options=10, random_seed=None):
        self.options = options
        self.action_counts = np.zeros(options)
        self.q_values = np.zeros(options)
        if random_seed is not None:
            np.random.seed(random_seed)  # For reproducibility if a seed is provided

    def choose_action(self):
        """
        Selects the action with the highest Q-value.
        Ties must be broken randomly.
        """
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