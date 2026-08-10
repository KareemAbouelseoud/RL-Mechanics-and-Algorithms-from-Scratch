import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np

from algorithms import GreedyAgent, EpsilonGreedyAgent, GradientAgent
from environments import MultiArmedBandit

AGENT_REGISTRY = {
    "greedy": GreedyAgent,
    "epsilon_greedy": EpsilonGreedyAgent,
    "gradient": GradientAgent,

}

ENV_REGISTRY = {
    "bandit": MultiArmedBandit,
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
    parser = argparse.ArgumentParser(description="Universal RL Benchmark Engine")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config) as f:
        config = json.load(f)

    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    exp_dir = os.path.join("results", experiment_name)

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
    agent = AgentClass(**config["agent"]["kwargs"])

    print(f"Started Experiment: [{experiment_name}]")
    print(f"Agent: {agent.__class__.__name__}")
    print(f"Environment: {env.__class__.__name__}")

    seeds = config["simulation"].get("seeds", 200)
    steps = config["simulation"].get("steps", 1000)
    master_seed = config["simulation"].get("master_seed", 0)

    reward_history = np.zeros((seeds, steps))
    optimal_action_history = np.zeros((seeds, steps))
    optimal_values = np.zeros(seeds)  # best-arm true value per seed, for regret calc
    has_oracle = hasattr(env, "get_optimal")

    # One SeedSequence for the whole run, spawned into `seeds` independent
    # per-run seed sequences, each of which spawns exactly 2 children:
    # one for the environment, one for the agent. This guarantees no
    # stream in the entire run collides with any other, without
    # hand-rolled offsets.
    run_seed_seqs = np.random.SeedSequence(master_seed).spawn(seeds)

    for seed_idx, seed_seq in enumerate(run_seed_seqs):
        rng_env, rng_agent = [np.random.default_rng(s) for s in seed_seq.spawn(2)]

        env.reset(rng=rng_env)
        agent.reset(rng=rng_agent)

        optimal_action = None
        if has_oracle:
            optimal_action, optimal_value = env.get_optimal()
            optimal_values[seed_idx] = optimal_value

        for step in range(steps):
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)

            reward_history[seed_idx, step] = reward
            if has_oracle and action == optimal_action:
                optimal_action_history[seed_idx, step] = 1

    print("Simulation complete. Calculating averages...")

    avg_rewards = np.mean(reward_history, axis=0)
    sem_rewards = np.std(reward_history, axis=0, ddof=1) / np.sqrt(seeds)

    if has_oracle:
        avg_optimal = np.mean(optimal_action_history, axis=0) * 100
        save_data = np.column_stack((avg_rewards, sem_rewards, avg_optimal))
        header = "Average_Reward,SEM_Reward,Optimal_Action_Percentage"
    else:
        save_data = np.column_stack((avg_rewards, sem_rewards))
        header = "Average_Reward,SEM_Reward"

    np.savetxt(os.path.join(exp_dir, "results.csv"), save_data, delimiter=",", header=header, comments="")

    # Raw per-seed data too, so anything (SEM bands, regret, distributions)
    # can be recomputed later without rerunning the simulation.
    np.savez_compressed(
        os.path.join(exp_dir, "raw.npz"),
        reward_history=reward_history,
        optimal_action_history=optimal_action_history,
        optimal_values=optimal_values,
    )

    meta = {
        "git_hash": get_git_hash(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "master_seed": master_seed,
        "seeds": seeds,
        "steps": steps,
        "has_oracle": has_oracle,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"All data saved to: {exp_dir}/")


if __name__ == "__main__":
    main()