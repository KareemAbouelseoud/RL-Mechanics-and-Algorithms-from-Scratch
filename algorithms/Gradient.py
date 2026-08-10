import numpy as np

class GradientAgent:
    def __init__(self, n_actions=10, alpha=0.1,
                 use_decay=False, decay_rate=0.99,
                 use_baseline=True):

        self.n_actions = n_actions
        self.initial_alpha = alpha  # remembered for reset, same pattern as initial_epsilon
        self.alpha = alpha

        self.use_decay = use_decay
        self.decay_rate = decay_rate
        self.use_baseline = use_baseline

        self.rng = np.random.default_rng()

        self.H = np.zeros(n_actions)  # Preference values
        self.action_counts = np.zeros(n_actions)
        self.avg_reward = 0.0
        self.t = 0

    def _softmax(self):
        # Standard numerical stability trick: subtract max before exponentiating
        exp_H = np.exp(self.H - np.max(self.H))
        return exp_H / np.sum(exp_H)

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng

        self.H = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.avg_reward = 0.0
        self.t = 0

        # Must reset alpha back to its starting value
        self.alpha = self.initial_alpha

    def choose_action(self):
        probabilities = self._softmax()
        return self.rng.choice(self.n_actions, p=probabilities)

    def update(self, action, reward):
        self.action_counts[action] += 1
        probabilities = self._softmax()

        # Baseline: running average reward, used to reduce variance in the
        # update signal. Optional — with use_baseline=False this degrades to
        # the plain (no-baseline) gradient bandit update from the slides.
        if self.use_baseline:
            self.t += 1
            self.avg_reward += (reward - self.avg_reward) / self.t
            target = reward - self.avg_reward
        else:
            target = reward

        indicator = np.zeros(self.n_actions)
        indicator[action] = 1
        # Preference for the chosen action moves toward the reward signal;
        # every other action's preference moves the opposite way, weighted
        # by how likely it was under the current policy.
        self.H += self.alpha * target * (indicator - probabilities)

        if self.use_decay:
            self.alpha *= self.decay_rate