import numpy as np

class EpsilonGreedyAgent:
    """
    An advanced Multi-Armed Bandit agent with optional enhancements:
    Epsilon Decay, Optimistic Initialization, and SoftMax Action Selection.
    """
    
    def __init__(self, options=10, epsilon=0.1, random_seed=None,
                 use_decay=False, decay_rate=0.99,
                 use_optimistic_init=False, optimistic_value=5.0,
                 use_softmax=False):
        
        self.options = options
        self.initial_epsilon = epsilon  # Need to remember this for when we reset!
        self.epsilon = epsilon
        
        # Improvement Parameters
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        self.use_optimistic_init = use_optimistic_init
        self.optimistic_value = optimistic_value
        self.use_softmax = use_softmax
        
        if random_seed is not None:
            np.random.seed(random_seed)

        self.action_counts = np.zeros(options)
        
        # OPTIMISTIC INITIALIZATION
        if self.use_optimistic_init:
            # Fill the array with the high optimistic value (e.g., 5.0) instead of 0
            self.q_values = np.full(options, float(optimistic_value))
        else:
            self.q_values = np.zeros(options)

    def choose_action(self):
        """
        Selects an action using either SoftMax or standard Epsilon-Greedy.
        """
        if self.use_softmax:
            # Subtracting the max Q-value is a standard ML trick for numerical stability
            # so the e^x calculation doesn't overflow to infinity!
            exp_q = np.exp(self.q_values - np.max(self.q_values))
            probabilities = exp_q / np.sum(exp_q)
            
            # Pick an action directly based on these calculated percentages
            return np.random.choice(self.options, p=probabilities)

        # Standard Epsilon-Greedy Selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.options)
        else:
            indices_of_max = np.where(self.q_values == np.max(self.q_values))[0]
            return np.random.choice(indices_of_max)
        
    def update(self, action, reward):
        """
        Updates Q-values and applies epsilon decay if activated.
        """
        self.action_counts[action] += 1        
        if self.use_decay:
            # Multiply epsilon by the decay rate (e.g., 0.99) on every single step
            self.epsilon *= self.decay_rate
            
    def reset(self, random_seed=None):
        """
        Wipes the agent's memory and resets epsilon for a new episode.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.action_counts = np.zeros(self.options)
        
        # Must reset epsilon back to its starting value (e.g., 1.0)
        self.epsilon = self.initial_epsilon 
        
        # Must re-apply optimistic initialization if it's turned on
        if self.use_optimistic_init:
            self.q_values = np.full(self.options, float(self.optimistic_value))
        else:
            self.q_values = np.zeros(self.options)