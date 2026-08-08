"""
Multi-Armed Bandit Environments

This module contains implementations of stationary and non-stationary 
multi-armed bandit problems. These environments serve as the foundational 
testbeds for evaluating purely evaluative (value-based) and policy-gradient 
Reinforcement Learning algorithms.
"""

import numpy as np

class MultiArmedBandit:
    """
    A Stationary k-Armed Bandit Environment.
    
    This environment simulates a casino with k slot machines. Upon initialization, 
    each machine is assigned a true expected reward (q-star) drawn from a standard 
    normal distribution. When an agent takes an action, the environment returns 
    a noisy reward drawn from a normal distribution centered at the action's 
    true expected value.

    Attributes:
        k_arms (int): The number of available actions (levers).
        _true_values (np.ndarray): Private array of true expected values for each arm.
        optimal_action (int): The index of the arm with the highest true expected value.
        optimal_value (float): The actual highest true expected value.
    """

    def __init__(self, k_arms=10, random_seed=None):
        """
        Initializes the Bandit environment.

        Args:
            k_arms (int): Number of arms (actions) in the bandit. Default is 10.
            random_seed (int, optional): Seed for reproducibility.
        """
        self.k_arms = k_arms
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self._true_values = np.random.normal(loc=0.0, scale=1.0, size=k_arms)
        
        # Pre-calculate the optimal values for benchmarking regret later
        self.optimal_action = np.argmax(self._true_values)
        self.optimal_value = np.max(self._true_values)

    def step(self, action):
        """
        Takes an action in the environment and returns a noisy reward.

        Args:
            action (int): The index of the arm chosen by the agent.

        Returns:
            float: The reward received, consisting of the true value plus Gaussian noise.
        """
        # The noise is centered at 0 with a standard deviation of 1
        noise = np.random.normal(loc=0.0, scale=1.0)
        
        # Reward = True Expected Value + Noise
        reward = self._true_values[action] + noise
        return reward

    def get_optimal(self):
        """
        Returns the optimal action and its true expected value for calculating regret.
        
        Returns:
            tuple: (optimal_action_index, optimal_true_value)
        """
        return self.optimal_action, self.optimal_value

    def reset(self,random_seed=None):
        """
        Resets the environment by generating a completely new set of slot machines.
        This is called at the end of an episode/run to prepare for the next seed.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self._true_values = np.random.normal(loc=0.0, scale=1.0, size=self.k_arms)
        self.optimal_action = np.argmax(self._true_values)
        self.optimal_value = np.max(self._true_values)