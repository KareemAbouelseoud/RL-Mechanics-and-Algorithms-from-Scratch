import numpy as np

class GreedyAgent:
    """
    A purely exploitative Multi-Armed Bandit agent.
    Always chooses the action with the highest estimated value.
    """
    
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.action_counts = np.zeros(n_actions)
        self.q_values = np.zeros(n_actions)
        self.rng = np.random.default_rng()

    def choose_action(self):
        """
        Selects the action with the highest Q-value.
        Ties must be broken randomly.
        """
        indices_of_max = np.where(self.q_values == np.max(self.q_values))[0]  # Get indices of all max Q-values
        return indices_of_max[0] if len(indices_of_max) == 1 else self.rng.choice(indices_of_max)  # Break ties randomly
        
    def update(self, action, reward):
        """
        Updates the action count and Q-value estimate for the chosen action.
        """
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        
    def reset(self, rng=None):
        """
        Wipes the agent's memory for a new episode.
        """
        if rng is not None:
            self.rng = rng
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)