"""
Value Iteration agent for tabular MDPs, written as pure JAX functions
plus a thin stateful wrapper matching the benchmark engine's interface:

    agent.set_model(P, R)                              # called once, before any episodes
    agent.reset(rng=...)
    agent.choose_action(state, rng=...)
    agent.update(state, action, reward, next_state, done)

Note: Value Iteration is the eval_sweeps=1 special case of the Modified
Policy Iteration family in policy_iteration.py, but is implemented here via
its own fused max-in-one-sweep update, which is the more standard/efficient
way to express Value Iteration specifically.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

def _argmax_with_random_tiebreak(Q: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    argmax over axis=1, breaking ties uniformly at random instead of always
    picking the lowest action index. Q has shape (num_states, num_actions).
    """
    is_max = Q == jnp.max(Q, axis=1, keepdims=True)
    noise = jax.random.uniform(key, Q.shape)
    tiebreak_scores = jnp.where(is_max, noise, -jnp.inf)
    return jnp.argmax(tiebreak_scores, axis=1)


@partial(jax.jit, static_argnames=("gamma",))
def _vi_update_step(V: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """One sweep of Value Iteration: V(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V(s') ]"""
    expected_future = jnp.einsum('san,n->sa', P, V)
    Q = R + gamma * expected_future
    return jnp.max(Q, axis=1)


@partial(jax.jit, static_argnames=("gamma",))
def _extract_policy(V: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray, gamma: float,
                     key: jax.Array) -> jnp.ndarray:
    """Extracts the greedy policy from the optimal value function, with random tie-breaking."""
    expected_future = jnp.einsum('san,n->sa', P, V)
    Q = R + gamma * expected_future
    return _argmax_with_random_tiebreak(Q, key)


@partial(jax.jit, static_argnames=("num_states", "max_iters",'gamma'))
def _value_iteration_solve(P: jnp.ndarray, R: jnp.ndarray, num_states: int,
                            gamma: float, theta: float, max_iters: int,
                            key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Runs the entire Value Iteration algorithm to convergence on-device."""
    def cond_fn(carry):
        V, V_prev, i = carry
        delta = jnp.max(jnp.abs(V - V_prev))
        return jnp.logical_and(delta > theta, i < max_iters)

    def body_fn(carry):
        V, _, i = carry
        V_next = _vi_update_step(V, P, R, gamma)
        return (V_next, V, i + 1)

    V_init = jnp.zeros(num_states)
    init_carry = (V_init, V_init - (theta * 2 + 1.0), 0)
    V_final, _, final_iters = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    pi_final = _extract_policy(V_final, P, R, gamma, key)
    return V_final, pi_final, final_iters


class ValueIterationAgent:
    """
    Model-based planning agent for Value Iteration. Solving happens once,
    in set_model(), because Value Iteration is deterministic given fixed
    P/R — there is nothing to "learn" per episode or per seed.
    choose_action() is then just a lookup into the cached optimal policy,
    and update() is a no-op.
    """

    def __init__(self, num_states: int, num_actions: int, gamma: float = 0.99,
                 theta: float = 1e-5, max_iters: int = 10000, tiebreak_seed: int = 0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iters = max_iters
        self.key = jax.random.PRNGKey(tiebreak_seed)

        self.V = None
        self.pi = None
        self._solved = False

    def set_model(self, P: jnp.ndarray, R: jnp.ndarray) -> None:
        self.V, self.pi, final_iters = _value_iteration_solve(
            P, R, self.num_states, self.gamma, self.theta, self.max_iters, self.key,
        )
        self.V = np.asarray(self.V)
        self.pi = np.asarray(self.pi)
        self._solved = True
        print(f"Value Iteration solved entirely on-device in {final_iters} iterations.")

    def reset(self, rng=None):
        if not self._solved:
            raise RuntimeError("ValueIterationAgent.reset() called before set_model().")

    def choose_action(self, state, rng=None):
        if not self._solved:
            raise RuntimeError("ValueIterationAgent.choose_action() called before set_model().")
        return int(self.pi[int(state)])

    def update(self, state, action, reward, next_state, done):
        # No-op: agent already knows everything.
        pass