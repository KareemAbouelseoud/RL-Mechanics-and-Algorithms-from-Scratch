import numpy as np

class EpsilonGreedyAgent:
    """
    An advanced Multi-Armed Bandit agent with optional enhancements:
    Epsilon Decay, Optimistic Initialization, and SoftMax Action Selection.
    Normally, we should inherit from a base class, but for the sake of readability and learning, 
    this is a standalone implementation.
    """
    
    def __init__(self, n_actions=10, epsilon=0.1,
                 use_decay=False, decay_rate=0.99,
                 use_optimistic_init=False, optimistic_value=5.0,
                 use_softmax=False, temperature=1.0, min_temperature=0.01):
        
        self.n_actions = n_actions
        self.initial_epsilon = epsilon  # Need to remember this for when we reset
        self.epsilon = epsilon
        
        # Improvement Parameters
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        self.use_optimistic_init = use_optimistic_init
        self.optimistic_value = optimistic_value
        self.use_softmax = use_softmax

        # Temperature: controls how peaked/flat the softmax exploration
        # distribution is. High temp -> near-uniform. Low temp -> near-argmax.
        self.initial_temperature = temperature
        self.temperature = temperature
        self.min_temperature = min_temperature  # floor, so we never divide by ~0
        
        self.rng = np.random.default_rng()

        self.action_counts = np.zeros(n_actions)
        
        # OPTIMISTIC INITIALIZATION
        if self.use_optimistic_init:
            # Fill the array with the high optimistic value (e.g., 5.0) instead of 0
            self.q_values = np.full(n_actions, float(optimistic_value))
        else:
            self.q_values = np.zeros(n_actions)

    def choose_action(self):
        """
        Epsilon controls how often we explore.
        When exploring, Softmax (if enabled) weights exploration toward
        less-bad actions instead of pulling uniformly at random.
        """
        if self.rng.random() < self.epsilon:
            if self.use_softmax:
                # Scale by temperature before exponentiating
                scaled_q = self.q_values / self.temperature

                # Subtracting the max is a standard numerical stability trick
                exp_q = np.exp(scaled_q - np.max(scaled_q))
                probabilities = exp_q / np.sum(exp_q)

                return self.rng.choice(self.n_actions, p=probabilities)
            return self.rng.integers(self.n_actions)
        else:
            indices_of_max = np.where(self.q_values == np.max(self.q_values))[0]
            return indices_of_max[0] if len(indices_of_max) == 1 else self.rng.choice(indices_of_max)
        
    def update(self, action, reward):
        """
        Updates Q-values and applies epsilon/temperature decay if activated.
        """
        self.action_counts[action] += 1        
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

        if self.use_decay:
            # Multiply epsilon by the decay rate (e.g., 0.99) on every single step
            self.epsilon *= self.decay_rate
            # Same idea for temperature: sharpen the softmax distribution
            # over time as Q-value estimates become more trustworthy.
            self.temperature = max(self.temperature * self.decay_rate, self.min_temperature)
            
    def reset(self, rng=None):
        """
        Wipes the agent's memory and resets epsilon/temperature for a new episode.
        """
        if rng is not None:
            self.rng = rng
            
        self.action_counts = np.zeros(self.n_actions)
        
        # Must reset epsilon and temperature back to their starting values
        self.epsilon = self.initial_epsilon 
        self.temperature = self.initial_temperature
        
        # Must reapply optimistic initialization if its turned on
        if self.use_optimistic_init:
            self.q_values = np.full(self.n_actions, float(self.optimistic_value))
        else:
            self.q_values = np.zeros(self.n_actions)