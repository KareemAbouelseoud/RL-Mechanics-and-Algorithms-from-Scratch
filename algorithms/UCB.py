import numpy as np

class UCBAgent:
    def __init__(self, n_actions, c=2.0, use_variance=False):
        self.n_actions = n_actions
        self.c = c
        self.use_variance = use_variance

        self.rng = np.random.default_rng()

        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)      # running mean per arm, Q(a)
        self.m2 = np.zeros(n_actions)           # running sum of squared deviations, for Welford's variance
        self.t = 0

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        self.counts = np.zeros(self.n_actions)
        self.values = np.zeros(self.n_actions)
        self.m2 = np.zeros(self.n_actions)
        self.t = 0

    def choose_action(self):
        # Must try every arm once before any confidence formula is well-defined
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return self.rng.choice(untried)

        if self.use_variance:
            # Sample variance per arm: m2 / (n - 1), guarded against n=1 (division by zero)
            variance = np.where(self.counts > 1, self.m2 / np.maximum(self.counts - 1, 1), 1.0)
            bonus = np.sqrt((variance / self.counts) * np.log(self.t))
        else:
            bonus = self.c * np.sqrt(np.log(self.t) / self.counts)

        ucb_scores = self.values + bonus
        indices_of_max = np.where(ucb_scores == np.max(ucb_scores))[0]
        return indices_of_max[0] if len(indices_of_max) == 1 else self.rng.choice(indices_of_max)

    def update(self, action, reward):
        self.t += 1
        self.counts[action] += 1
        n = self.counts[action]

        # Welford's incremental mean + variance, one pass so we don't have to store all rewards per action.
        delta = reward - self.values[action]
        self.values[action] += delta / n
        delta2 = reward - self.values[action]
        self.m2[action] += delta * delta2