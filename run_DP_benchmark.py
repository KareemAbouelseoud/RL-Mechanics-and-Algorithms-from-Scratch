import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import jax

from algorithms import ValueIterationAgent, PolicyIterationAgent
from environments import GridWorld

AGENT_REGISTRY = {
    "value_iteration": ValueIterationAgent,
    "policy_iteration": PolicyIterationAgent,
}

ENV_REGISTRY = {
    "gridworld": GridWorld,
}


def get_class(registry, name, kind):
    if name not in registry:
        raise ValueError(f"Unknown {kind} '{name}'. Options: {list(registry)}")
    return registry[name]


def get_git_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Stateful MDP Benchmark Engine")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    exp_dir = os.path.join("results_mdp", experiment_name)

    if os.path.exists(exp_dir) and not args.force:
        raise FileExistsError(
            f"{exp_dir} already exists. Use --force to overwrite, "
            f"or rename the config to start a fresh run."
        )
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(exp_dir, "config.json"))

    EnvClass = get_class(ENV_REGISTRY, config["environment"]["name"], "environment")
    AgentClass = get_class(AGENT_REGISTRY, config["agent"]["name"], "agent")

    env = EnvClass(**config["environment"]["kwargs"])
    params = env.default_params  # Gymnax convention: params are passed explicitly, not stored on env
    agent = AgentClass(**config["agent"]["kwargs"])

    print(f"Started MDP Experiment: [{experiment_name}]")

    # ---------------------------------------------------------
    # THE PLANNING HOOK: Give the blueprint to Model-Based Agents
    # ---------------------------------------------------------
    # DP agents (Value/Policy Iteration) are deterministic given fixed P/R —
    # solving happens exactly ONCE here, not per-seed. Re-solving per seed
    # would recompute the identical policy every time, since nothing about
    # set_model() depends on any of the environment/agent randomness below.
    solve_time = None
    if hasattr(agent, "set_model") and hasattr(env, "get_transition_dynamics"):
        P, R = env.get_transition_dynamics(params)
        solve_start = time.time()
        agent.set_model(P, R)
        solve_time = time.time() - solve_start

        # Save the solver's actual output: value function + extracted policy.
        # This is the one thing that's impossible to recover after the run
        # ends without re-solving from scratch, so it's saved unconditionally
        # whenever the agent exposes V/pi.
        if hasattr(agent, "V") and hasattr(agent, "pi") and agent.V is not None:
            np.savez_compressed(
                os.path.join(exp_dir, "solution.npz"),
                V=np.asarray(agent.V),
                pi=np.asarray(agent.pi),
            )

    seeds = config["simulation"].get("seeds", 5)
    episodes = config["simulation"].get("episodes", 100)
    max_steps_per_ep = config["simulation"].get("max_steps", 50)
    master_seed = config["simulation"].get("master_seed", 0)

    # Which single (seed, episode) to record a full step-by-step trajectory
    # for, so it can be replayed / drawn as a path through the grid later.
    # Defaults to the very last episode of seed 0, on the assumption that's
    # representative of "settled" behavior for a fixed DP policy (there's no
    # learning curve for DP agents, but this keeps the convention consistent
    # with something you might reuse for learning-based agents later).
    trajectory_seed_idx = config["simulation"].get("trajectory_seed_idx", 0)
    trajectory_episode = config["simulation"].get("trajectory_episode", episodes - 1)

    # We track total return (sum of rewards) per episode
    returns_history = np.zeros((seeds, episodes))

    trajectory_states, trajectory_actions, trajectory_rewards = [], [], []

    # Gymnax needs jax.random.PRNGKey, not numpy Generators. We still use
    # SeedSequence to derive independent per-seed integer seeds (keeping the
    # same reproducible-seed-isolation pattern used in the bandit engine),
    # then convert each to a PRNGKey for the JAX-side calls, and split off a
    # separate stream for the agent (used for e.g. tie-breaking, if the
    # agent exposes any stochastic behavior via reset/choose_action).
    run_seed_seqs = np.random.SeedSequence(master_seed).spawn(seeds)

    for seed_idx, seed_seq in enumerate(run_seed_seqs):
        env_seed, agent_seed = seed_seq.generate_state(2)
        key = jax.random.PRNGKey(int(env_seed))
        rng_agent = np.random.default_rng(int(agent_seed))  # DP agents don't need JAX randomness at rollout time

        for ep in range(episodes):
            record_this_episode = (seed_idx == trajectory_seed_idx and ep == trajectory_episode)

            # 1. Reset Environment for a new episode (Gymnax: key, params -> obs, state)
            key, reset_key = jax.random.split(key)
            obs, env_state = env.reset(reset_key, params)
            agent.reset(rng=rng_agent)

            ep_return = 0.0

            for step in range(max_steps_per_ep):
                # 2. Agent needs current state (as a plain state index) to act
                action = agent.choose_action(int(obs), rng=rng_agent)

                # 3. Environment advances (Gymnax: key, state, action, params -> obs, state, reward, done, info)
                key, step_key = jax.random.split(key)
                next_obs, env_state, reward, done, _info = env.step(step_key, env_state, action, params)

                # 4. Agent learns (ignored by DP, used by Q-Learning later)
                agent.update(int(obs), action, float(reward), int(next_obs), bool(done))

                if record_this_episode:
                    trajectory_states.append(int(obs))
                    trajectory_actions.append(int(action))
                    trajectory_rewards.append(float(reward))

                ep_return += float(reward)
                obs = next_obs

                # 5. End episode if terminal state reached
                if bool(done):
                    if record_this_episode:
                        trajectory_states.append(int(obs))  # log the final terminal state too
                    break

            returns_history[seed_idx, ep] = ep_return

    print("Simulation complete. Saving results...")

    avg_returns = np.mean(returns_history, axis=0)
    sem_returns = np.std(returns_history, axis=0, ddof=1) / np.sqrt(seeds)

    save_data = np.column_stack((avg_returns, sem_returns))
    np.savetxt(os.path.join(exp_dir, "results.csv"), save_data, delimiter=",", header="Average_Return,SEM_Return", comments="")

    # Raw per-seed episode returns, so distributions/spread can be examined
    # later without rerunning (same role raw.npz plays in the bandit engine).
    np.savez_compressed(os.path.join(exp_dir, "raw.npz"), returns_history=returns_history)

    if trajectory_states:
        np.savez_compressed(
            os.path.join(exp_dir, "trajectory.npz"),
            states=np.array(trajectory_states),
            actions=np.array(trajectory_actions),
            rewards=np.array(trajectory_rewards),
        )
    else:
        print(f"Warning: no trajectory recorded — trajectory_seed_idx={trajectory_seed_idx} / "
              f"trajectory_episode={trajectory_episode} was never reached (check seeds/episodes bounds).")

    meta = {
        "git_hash": get_git_hash(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "master_seed": master_seed,
        "seeds": seeds,
        "episodes": episodes,
        "max_steps_per_ep": max_steps_per_ep,
        "solve_time_seconds": solve_time,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"All data saved to: {exp_dir}/")


if __name__ == "__main__":
    main()