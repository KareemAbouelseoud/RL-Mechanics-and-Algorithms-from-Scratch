"""
Gymnax-compliant 4x4 GridWorld Environment for JAX.

Supports both pure functional episodic rollouts and analytical extraction
of transition probability tensors P(s' | s, a) and reward tensors R(s, a)
for Dynamic Programming.
"""

from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Static parameters for the GridWorld environment."""
    grid_size: int = 4
    step_reward: float = -1.0
    terminal_reward: float = 0.0
    max_steps_in_episode: int = 100


@struct.dataclass
class EnvState(environment.EnvState):
    """Dynamic, immutable state of the GridWorld agent."""
    pos: jnp.ndarray  # Shape: (2,), [row, col]
    time: int


class GridWorld(environment.Environment[EnvState, EnvParams]):
    """
    Classic 4x4 GridWorld environment implemented in JAX/Gymnax.
    Terminal states are at (0, 0) and (3, 3).
    """

    def __init__(self, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self._num_actions = 4

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(grid_size=self.grid_size)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, float, bool, Dict[str, Any]]:
        """
        Pure functional step transition.
        Actions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        Not used by the DP algorithms (they solve analytically from P/R),
        but needed for rollouts, evaluation, and any learning-based agents.
        """
        action_deltas = jnp.array([
            [-1, 0],  # 0: UP
            [0, 1],   # 1: RIGHT
            [1, 0],   # 2: DOWN
            [0, -1],  # 3: LEFT
        ])
        delta = action_deltas[action]

        current_terminal = self.is_terminal(state, params)

        new_pos = jnp.clip(state.pos + delta, 0, params.grid_size - 1)
        new_pos = jax.lax.select(current_terminal, state.pos, new_pos)

        new_state = EnvState(pos=new_pos, time=state.time + 1)
        done = self.is_terminal(new_state, params)

        reward = jax.lax.select(current_terminal, params.terminal_reward, params.step_reward)
        obs = self.get_obs(new_state)

        return obs, new_state, reward, done, {}

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        """Resets the agent to a non-terminal state chosen randomly."""
        flat_idx = jax.random.randint(key, shape=(), minval=1, maxval=params.grid_size * params.grid_size - 1)
        row = flat_idx // params.grid_size
        col = flat_idx % params.grid_size

        state = EnvState(pos=jnp.array([row, col], dtype=jnp.int32), time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jnp.ndarray:
        """Returns the 1D discrete state index (row * grid_size + col)."""
        return state.pos[0] * self.grid_size + state.pos[1]

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Terminal states: (0, 0) and (grid_size - 1, grid_size - 1)."""
        is_top_left = jnp.all(state.pos == jnp.array([0, 0]))
        is_bottom_right = jnp.all(state.pos == jnp.array([params.grid_size - 1, params.grid_size - 1]))
        return jnp.logical_or(is_top_left, is_bottom_right)

    @property
    def name(self) -> str:
        return f"GridWorld-{self.grid_size}x{self.grid_size}-Gymnax"

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_states)

    def get_transition_dynamics(self, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes analytical MDP matrices for Dynamic Programming algorithms.

        Returns:
            P (jnp.ndarray): Shape (num_states, num_actions, num_states), P[s, a, s']
            R (jnp.ndarray): Shape (num_states, num_actions), expected rewards R[s, a]
        """
        N = self.num_states
        A = self._num_actions
        G = params.grid_size

        P = jnp.zeros((N, A, N))
        R = jnp.zeros((N, A))

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for s in range(N):
            r, c = divmod(s, G)
            is_term = (r == 0 and c == 0) or (r == G - 1 and c == G - 1)

            for a in range(A):
                if is_term:
                    P = P.at[s, a, s].set(1.0)
                    R = R.at[s, a].set(params.terminal_reward)
                else:
                    dr, dc = deltas[a]
                    nr, nc = max(0, min(G - 1, r + dr)), max(0, min(G - 1, c + dc))
                    next_s = nr * G + nc
                    P = P.at[s, a, next_s].set(1.0)
                    R = R.at[s, a].set(params.step_reward)

        return P, R