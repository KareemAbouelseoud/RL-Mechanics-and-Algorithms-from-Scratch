"""
Policy Iteration agent for tabular MDPs, written as pure JAX functions
plus a thin stateful wrapper matching the benchmark engine's interface:

    agent.set_model(P, R)                              # called once, before any episodes
    agent.reset(rng=...)
    agent.choose_action(state, rng=...)
    agent.update(state, action, reward, next_state, done)
"""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

@partial(jax.jit, static_argnames=("gamma",))
def _evaluate_step(V: jnp.ndarray, pi: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray,
                    gamma: float, state_indices: jnp.ndarray) -> jnp.ndarray:
    """One sweep of policy evaluation: V(s) = R(s,pi(s)) + gamma * sum_s' P(s'|s,pi(s)) V(s')"""
    P_pi = P[state_indices, pi, :]
    R_pi = R[state_indices, pi]
    return R_pi + gamma * jnp.dot(P_pi, V)


def _argmax_with_random_tiebreak(Q: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    argmax over axis=1, breaking ties uniformly at random instead of always
    picking the lowest action index. Q has shape (num_states, num_actions).

    Mechanism: among actions tied for the max, assign each a small random
    score and take argmax of THAT — this never changes which actions are
    "in the running" (only tied actions can win), it only decides which
    tied action wins.
    """
    is_max = Q == jnp.max(Q, axis=1, keepdims=True)
    noise = jax.random.uniform(key, Q.shape)
    tiebreak_scores = jnp.where(is_max, noise, -jnp.inf)
    return jnp.argmax(tiebreak_scores, axis=1)


@partial(jax.jit, static_argnames=("gamma",))
def _improve_step(V: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray, gamma: float,
                   key: jax.Array) -> jnp.ndarray:
    """Greedy policy improvement: pi(s) = argmax_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V(s') ]"""
    expected_future = jnp.einsum('san,n->sa', P, V)
    Q = R + gamma * expected_future
    return _argmax_with_random_tiebreak(Q, key)


@partial(jax.jit, static_argnames=("gamma", "k"))
def _policy_evaluation_k_sweeps(V: jnp.ndarray, pi: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray,
                                 gamma: float, k: int, state_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Runs exactly k sweeps of policy evaluation, fully on-device via
    fori_loop. k=1 makes the outer solve loop behave like Value Iteration;
    larger k approaches full policy evaluation (Modified Policy Iteration).
    """
    def body_fn(i, V):
        return _evaluate_step(V, pi, P, R, gamma, state_indices)
    return jax.lax.fori_loop(0, k, body_fn, V)


@partial(jax.jit, static_argnames=("gamma", "theta", "max_inner_iters"))
def _policy_evaluation_to_convergence(V_init: jnp.ndarray, pi: jnp.ndarray, P: jnp.ndarray,
                                       R: jnp.ndarray, gamma: float, theta: float,
                                       max_inner_iters: int, state_indices: jnp.ndarray) -> jnp.ndarray:
    """Runs policy evaluation sweeps to convergence (classic full Policy Iteration)."""
    def cond_fn(carry):
        V, V_prev, i = carry
        delta = jnp.max(jnp.abs(V - V_prev))
        return jnp.logical_and(delta > theta, i < max_inner_iters)

    def body_fn(carry):
        V, _, i = carry
        V_next = _evaluate_step(V, pi, P, R, gamma, state_indices)
        return (V_next, V, i + 1)

    init_carry = (V_init, V_init - (theta * 2 + 1.0), 0)
    V_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return V_final


def _policy_iteration_solve(P: jnp.ndarray, R: jnp.ndarray, num_states: int,
                             gamma: float, theta: float, max_outer_iters: int,
                             max_inner_iters: int, eval_sweeps: Optional[int],
                             key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full (Modified) Policy Iteration: alternates policy evaluation and
    greedy policy improvement until the policy stops changing.

    eval_sweeps:
        None  -> classic full Policy Iteration (evaluate to convergence
                 every outer iteration).
        int k -> Modified/Truncated Policy Iteration (evaluate for exactly
                 k sweeps every outer iteration). k=1 behaves like Value
                 Iteration; small k trades a bit of per-iteration accuracy
                 for far fewer total Bellman backups.

    The outer loop stays in Python: max_outer_iters is small (a handful of
    sweeps on a 4x4 grid), so the host-sync cost of the convergence check
    per outer iteration is negligible.
    """
    state_indices = jnp.arange(num_states)
    V = jnp.zeros(num_states)
    pi = jnp.zeros(num_states, dtype=jnp.int32)

    for outer_i in range(max_outer_iters):
        if eval_sweeps is None:
            V = _policy_evaluation_to_convergence(V, pi, P, R, gamma, theta, max_inner_iters, state_indices)
        else:
            V = _policy_evaluation_k_sweeps(V, pi, P, R, gamma, eval_sweeps, state_indices)

        key, subkey = jax.random.split(key)
        pi_next = _improve_step(V, P, R, gamma, subkey)

        if jnp.array_equal(pi, pi_next):
            print(f"Policy Iteration converged in {outer_i + 1} policy updates.")
            break
        pi = pi_next
    else:
        print(f"Policy Iteration did NOT converge within {max_outer_iters} outer iterations.")

    return V, pi


class PolicyIterationAgent:
    """
    Model-based planning agent. Solving happens once, in set_model(),
    because Policy Iteration is deterministic given fixed P/R — there is
    nothing to "learn" per episode or per seed. choose_action() is then
    just a lookup into the cached optimal policy, and update() is a no-op.

    eval_sweeps controls Modified Policy Iteration:
        None -> classic full Policy Iteration.
        k    -> truncate policy evaluation to k sweeps per outer iteration.
    """

    def __init__(self, num_states: int, num_actions: int, gamma: float = 0.99,
                 theta: float = 1e-5, max_outer_iters: int = 100,
                 max_inner_iters: int = 1000, eval_sweeps: Optional[int] = None,
                 tiebreak_seed: int = 0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.eval_sweeps = eval_sweeps
        self.key = jax.random.PRNGKey(tiebreak_seed)

        self.V = None
        self.pi = None
        self._solved = False

    def set_model(self, P: jnp.ndarray, R: jnp.ndarray) -> None:
        self.V, self.pi = _policy_iteration_solve(
            P, R, self.num_states, self.gamma, self.theta,
            self.max_outer_iters, self.max_inner_iters, self.eval_sweeps, self.key,
        )
        self.pi = np.asarray(self.pi)
        self._solved = True

    def reset(self, rng=None):
        if not self._solved:
            raise RuntimeError("PolicyIterationAgent.reset() called before set_model(); no policy available yet.")

    def choose_action(self, state, rng=None):
        if not self._solved:
            raise RuntimeError("PolicyIterationAgent.choose_action() called before set_model().")
        return int(self.pi[int(state)])

    def update(self, state, action, reward, next_state, done):
        # No-op: DP agents don't learn from experience, they were solved
        # analytically ahead of time from the model (P, R).
        pass