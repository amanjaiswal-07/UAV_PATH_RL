import torch
import numpy as np
import sys

# Add project directories to the path to allow imports
sys.path.append("env")
sys.path.append("train")

from uav_env import UAVEnv
from dqnn_agent import DQNNAgent

def evaluate_agent(model_path, env, num_episodes=100):
    """
    Evaluates a trained DQNN agent.

    Args:
        model_path (str): The file path to the trained model's .pth file.
        env (gym.Env): The environment to test the agent on.
        num_episodes (int): The number of episodes to run for evaluation.
    """
    print(f"\n--- Evaluating model: {model_path} ---")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the agent with the same architecture as the one that was trained
    # Note: Hyperparameters like lr, gamma, etc., don't matter for evaluation.
    agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True)
    
    # Load the trained model weights from the specified file
    agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set the network to evaluation mode (disables things like dropout if used)
    agent.q_net.eval()

    # Lists to store the metrics from each evaluation episode
    all_rewards = []
    all_steps = []
    all_successes = []
    all_collisions = []
    all_coverage = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        
        total_reward = 0
        steps = 0
        collisions = 0
        
        while not done:
            # Select action greedily with no exploration
            action = agent.select_action(state, train_mode=False)
            
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1
            if info.get('collision', False):
                collisions += 1

        # Store the final metrics for this episode
        all_rewards.append(total_reward)
        all_steps.append(steps)
        if np.array_equal(env.uav_pos, env.goal):
             all_successes.append(1)
        else:
             all_successes.append(0)
        all_collisions.append(collisions)
        all_coverage.append(len(env.covered_area) / (env.grid_size[0] * env.grid_size[1]))

    # Calculate and print the average performance over all evaluation episodes
    avg_reward = np.mean(all_rewards)
    success_rate = (np.sum(all_successes) / num_episodes) * 100
    avg_steps = np.mean(all_steps)
    avg_collisions = np.mean(all_collisions)
    avg_coverage = np.mean(all_coverage)

    print("\n--- Evaluation Summary ---")
    print(f"Ran for {num_episodes} episodes.")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.2f}")
    print(f"Average Collisions per Episode: {avg_collisions:.2f}")
    print(f"Average Coverage Ratio: {avg_coverage:.2%}")
    print("--------------------------\n")

if __name__ == "__main__":
    # 1. Create the environment
    env = UAVEnv()
    
    # 2. IMPORTANT: Rename your best model file to "best_model.pth"
    #    or change the path below to match your file name.
    best_model_path = "dqnn_uav_model.pth"
    
    # 3. Run the evaluation function
    evaluate_agent(best_model_path, env)