# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# #change
# import sys
# sys.path.append(".") 
# #change
# from dqnn_agent import DQNNAgent
# #change
# from generate_data import generate_users, generate_obstacles, generate_snr_map
# #change

# def train_dqn(env, episodes=3000, max_steps=200):
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True, prioritized=False ,target_update_freq=1000)

#     # Define linear decay for epsilon
#     epsilon_start = agent.epsilon
#     epsilon_end = 0.01  # The minimum epsilon value
#     # Calculate the decay rate based on the number of episodes
#     epsilon_decay_rate = (epsilon_start - epsilon_end) / episodes

#     all_rewards = []
#     all_epsilons = []
#     all_steps = []
#     all_successes = []
#     all_coverage = []
#     all_collisions = []

#     for ep in range(episodes):
        
#         new_users = generate_users()
#         new_obstacles = generate_obstacles()
#         new_snr_map = generate_snr_map()
#         # Pass the new data to the reset function
#         state = env.reset(users=new_users, obstacles=new_obstacles, snr_map=new_snr_map)

#         # state = env.reset()
#         total_reward = 0
#         steps = 0
#         collisions = 0
#         goal_reached = False

#         for step in range(max_steps):
#             action = agent.select_action(state)
#             next_state, reward, done, info = env.step(action)
#             agent.store((state, action, reward, next_state, float(done)))
#             agent.train()
#             state = next_state
#             total_reward += reward
#             steps += 1

#             # ✅ Count collision (when UAV didn't move)
#             # if len(env.trajectory) > 1 and env.trajectory[-1] == env.trajectory[-2]:
#             #     collisions += 1

#             if info.get('collision', False):
#                 collisions += 1

#             if np.array_equal(env.uav_pos, env.goal):
#                 goal_reached = True

#             if done:
#                 break

#         all_rewards.append(total_reward)
#         all_epsilons.append(agent.epsilon)
#         all_steps.append(steps)
#         all_successes.append(1 if goal_reached else 0)
#         all_coverage.append(len(env.covered_area) / (env.grid_size[0] * env.grid_size[1]))
#         all_collisions.append(collisions)

#         print(f"Episode {ep+1}/{episodes} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {steps}, Success: {goal_reached}, Collisions: {collisions}")
#         agent.epsilon = max(epsilon_end, agent.epsilon - epsilon_decay_rate)

#         # print(f"Episode {ep+1}/{episodes} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, "
#         #       f"Steps: {steps}, Success: {success}, Collisions: {collisions}, Coverage: {coverage_ratio:.2f}")

#     # Save final model
#     torch.save(agent.q_net.state_dict(), "dqnn_uav_model_randomized_3.pth")
#     print("✅  Model trained on randomized environments saved.")

#     plt.figure(figsize=(15, 10))

#     plt.subplot(2, 3, 1)
#     plt.plot(all_rewards)
#     plt.title("Episode Reward")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")

#     plt.subplot(2, 3, 2)
#     plt.plot(all_epsilons)
#     plt.title("Epsilon Decay")
#     plt.xlabel("Episode")
#     plt.ylabel("Epsilon")

#     plt.subplot(2, 3, 3)
#     plt.plot(all_steps)
#     plt.title("Steps per Episode")
#     plt.xlabel("Episode")
#     plt.ylabel("Steps")

#     plt.subplot(2, 3, 4)
#     plt.plot(all_successes)
#     plt.title("Success (Reached Goal)")
#     plt.xlabel("Episode")
#     plt.ylabel("Success (0 or 1)")

#     plt.subplot(2, 3, 5)
#     plt.plot(all_collisions)
#     plt.title("Collisions per Episode")
#     plt.xlabel("Episode")
#     plt.ylabel("Collisions")

#     plt.subplot(2, 3, 6)
#     plt.plot(all_coverage)
#     plt.title("Coverage Ratio")
#     plt.xlabel("Episode")
#     plt.ylabel("Ratio [0-1]")

#     plt.tight_layout()
#     plt.savefig("training_metrics.png")
#     plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".") 

from dqnn_agent import DQNNAgent
from generate_data import generate_users, generate_obstacles, generate_snr_map
import pandas as pd

def train_dqn(env, episodes=3000, max_steps=200):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True, prioritized=False ,target_update_freq=1000)

    epsilon_start = agent.epsilon
    epsilon_end = 0.01
    epsilon_decay_rate = (epsilon_start - epsilon_end) / episodes

    all_rewards = []
    all_epsilons = []
    all_steps = []
    all_successes = []
    all_coverage = []
    all_collisions = []
    best_reward = float('-inf')

    for ep in range(episodes):
        try:
            new_users = generate_users()
            new_obstacles = generate_obstacles()
            new_snr_map = generate_snr_map()
            state = env.reset(users=new_users, obstacles=new_obstacles, snr_map=new_snr_map)
        except Exception as e:
            print(f"Episode {ep+1}: Failed to reset environment — {e}")
            continue

        total_reward = 0
        steps = 0
        collisions = 0
        goal_reached = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store((state, action, reward, next_state, float(done)))
            agent.train()
            state = next_state
            total_reward += reward
            steps += 1

            if info.get('collision', False):
                collisions += 1

            if np.array_equal(env.uav_pos, env.goal):
                goal_reached = True

            if done:
                break

        all_rewards.append(total_reward)
        all_epsilons.append(agent.epsilon)
        all_steps.append(steps)
        all_successes.append(1 if goal_reached else 0)
        all_coverage.append(len(env.covered_area) / (env.grid_size[0] * env.grid_size[1]))
        all_collisions.append(collisions)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_net.state_dict(), "best_model.pth")

        print(f"Episode {ep+1}/{episodes} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {steps}, Success: {goal_reached}, Collisions: {collisions}")
        agent.epsilon = max(epsilon_end, agent.epsilon - epsilon_decay_rate)

    torch.save(agent.q_net.state_dict(), "beta10.pth")
    print("✅ Model training complete. Final and best model saved.")

    df = pd.DataFrame({
        'reward': all_rewards,
        'epsilon': all_epsilons,
        'steps': all_steps,
        'success': all_successes,
        'coverage': all_coverage,
        'collisions': all_collisions
    })
    df['reward_avg'] = df['reward'].rolling(100).mean()
    df['success_avg'] = df['success'].rolling(100).mean()
    df['collisions_avg'] = df['collisions'].rolling(100).mean()

    # Save smoothed reward curve for comparison
    np.save('rewards_beta10.npy', df['reward_avg'].to_numpy())
    np.save('success_beta10.npy', df['success_avg'].to_numpy())
    np.save('collisions_beta10.npy', df['collisions_avg'].to_numpy())


    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(df['reward'], alpha=0.3)
    plt.plot(df['reward_avg'], label='Smoothed')
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(df['epsilon'])
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.subplot(2, 3, 3)
    plt.plot(df['steps'])
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.subplot(2, 3, 4)
    plt.plot(df['success'], alpha=0.3)
    plt.plot(df['success_avg'], label='Smoothed')
    plt.title("Success (Reached Goal)")
    plt.xlabel("Episode")
    plt.ylabel("Success (0 or 1)")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(df['collisions'], alpha=0.3)
    plt.plot(df['collisions_avg'], label='Smoothed')
    plt.title("Collisions per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Collisions")
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(df['coverage'])
    plt.title("Coverage Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio [0-1]")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
