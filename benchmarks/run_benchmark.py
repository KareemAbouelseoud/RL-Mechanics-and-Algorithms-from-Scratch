import argparse
import json
import os
import shutil
import numpy as np

# --- 1. Import your API Gateways ---
from algorithms import GreedyAgent, EpsilonGreedyAgent
from environments import MultiArmedBandit

# --- 2. The Registries ---
AGENT_REGISTRY = {
    "greedy": GreedyAgent,
    "epsilon_greedy": EpsilonGreedyAgent,
}

ENV_REGISTRY = {
    "bandit": MultiArmedBandit
}

def main():
    parser = argparse.ArgumentParser(description="Universal RL Benchmark Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON experiment config file")
    args = parser.parse_args()
    
    # 1. Load the Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # 2. Setup the isolated Experiment Directory
    # This takes "configs/ucb_test_1.json" and creates a folder called "results/ucb_test_1/"
    experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    exp_dir = os.path.join("results", experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Copy the exact config into the results folder for 100% reproducibility!
    shutil.copy(args.config, os.path.join(exp_dir, "config.json"))
    
    # 3. Instantiate Blueprints using the Registry Pattern
    EnvClass = ENV_REGISTRY[config["environment"]["name"]]
    AgentClass = AGENT_REGISTRY[config["agent"]["name"]]
    
    env = EnvClass(**config["environment"]["kwargs"])
    agent = AgentClass(**config["agent"]["kwargs"])
    
    print(f"Started Experiment: [{experiment_name}]")
    print(f"Agent: {agent.__class__.__name__}")
    print(f"Environment: {env.__class__.__name__}")
    
    # 4. Extract Simulation Parameters
    seeds = config["simulation"].get("seeds", 200)
    steps = config["simulation"].get("steps", 1000)
    
    # Pre-allocate memory for speed
    reward_history = np.zeros((seeds, steps))
    optimal_action_history = np.zeros((seeds, steps)) 
    
    # Check if this environment has a known perfect answer
    has_oracle = hasattr(env, "get_optimal")

    # --- THE MACRO LOOP (EPISODES/SEEDS) ---
    for seed in range(seeds):
        # Pass the exact current seed to both the environment and the agent
        env.reset(random_seed=seed)
        agent.reset(random_seed=seed)
        
        if has_oracle:
            optimal_action, _ = env.get_optimal()
            
        # --- THE MICRO LOOP (STEPS) ---
        for step in range(steps):
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)
            
            reward_history[seed, step] = reward
            if has_oracle and action == optimal_action:
                optimal_action_history[seed, step] = 1

    # 5. Data Processing & Saving
    print("Simulation complete. Calculating averages...")
    
    avg_rewards = np.mean(reward_history, axis=0)
    
    if has_oracle:
        avg_optimal = np.mean(optimal_action_history, axis=0) * 100
        save_data = np.column_stack((avg_rewards, avg_optimal))
        header = "Average_Reward,Optimal_Action_Percentage"
    else:
        save_data = avg_rewards
        header = "Average_Reward"
    
    # Save the CSV directly into the isolated experiment folder
    results_path = os.path.join(exp_dir, "results.csv")
    np.savetxt(results_path, save_data, delimiter=",", header=header, comments="")
    
    print(f"All data securely saved to: {exp_dir}/")

if __name__ == "__main__":
    main()