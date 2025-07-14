import torch
import numpy as np
import matplotlib.pyplot as plt
from dqnn_agent import DQNNAgent

def train_dqn(env, episodes=500, max_steps=100):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True, prioritized=False)

    # Metric trackers
    # all_rewards = []
    # all_epsilons = []
    # all_steps = []
    # success_count = []
    # collision_count = []
    # coverage_ratio_list = []

    # for ep in range(episodes):
    #     state = env.reset()
    #     total_reward = 0
    #     steps = 0
    #     collisions = 0
    #     done = False

    #     for step in range(max_steps):
    #         action = agent.select_action(state)
    #         next_state, reward, done, _ = env.step(action)

    #         agent.store((state, action, reward, next_state, float(done)))
    #         agent.train()

    #         state = next_state
    #         total_reward += reward
    #         steps += 1

    #         # Check if UAV attempted a collision this step
    #         if env.collisions and tuple(env.trajectory[-1]) == env.collisions[-1]:
    #             collisions += 1

    #         if done:
    #             break

    #     success = 1 if (env.uav_pos == env.goal).all() else 0
    #     coverage_ratio = len(env.covered_area) / (env.grid_size[0] * env.grid_size[1])

    #     # Append metrics
    #     all_rewards.append(total_reward)
    #     all_epsilons.append(agent.epsilon)
    #     all_steps.append(steps)
    #     success_count.append(success)
    #     collision_count.append(collisions)
    #     coverage_ratio_list.append(coverage_ratio)
    all_rewards = []
    all_epsilons = []
    all_steps = []
    all_successes = []
    all_coverage = []
    all_collisions = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        collisions = 0
        goal_reached = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store((state, action, reward, next_state, float(done)))
            agent.train()
            state = next_state
            total_reward += reward
            steps += 1

            # ✅ Count collision (when UAV didn't move)
            if len(env.trajectory) > 1 and env.trajectory[-1] == env.trajectory[-2]:
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

        print(f"Episode {ep+1}/{episodes} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {steps}, Success: {goal_reached}, Collisions: {collisions}")


        # print(f"Episode {ep+1}/{episodes} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, "
        #       f"Steps: {steps}, Success: {success}, Collisions: {collisions}, Coverage: {coverage_ratio:.2f}")

    # Save final model
    torch.save(agent.q_net.state_dict(), "dqnn_uav_model.pth")
    print("✅ Model saved to dqnn_uav_model.pth")

    # === Plot all 6 metrics ===
    # fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # axs[0, 0].plot(all_rewards)
    # axs[0, 0].set_title("Episode Reward")
    # axs[0, 0].set_xlabel("Episode")
    # axs[0, 0].set_ylabel("Total Reward")

    # axs[0, 1].plot(all_epsilons)
    # axs[0, 1].set_title("Epsilon Decay")
    # axs[0, 1].set_xlabel("Episode")
    # axs[0, 1].set_ylabel("Epsilon")

    # axs[0, 2].plot(all_steps)
    # axs[0, 2].set_title("Steps per Episode")
    # axs[0, 2].set_xlabel("Episode")
    # axs[0, 2].set_ylabel("Steps")

    # axs[1, 0].plot(success_count)
    # axs[1, 0].set_title("Success (Reached Goal)")
    # axs[1, 0].set_xlabel("Episode")
    # axs[1, 0].set_ylabel("Success (0 or 1)")

    # axs[1, 1].plot(collision_count)
    # axs[1, 1].set_title("Collisions per Episode")
    # axs[1, 1].set_xlabel("Episode")
    # axs[1, 1].set_ylabel("Collisions")

    # axs[1, 2].plot(coverage_ratio_list)
    # axs[1, 2].set_title("Coverage Ratio")
    # axs[1, 2].set_xlabel("Episode")
    # axs[1, 2].set_ylabel("Ratio [0–1]")

    # plt.tight_layout()
    # plt.savefig("training_metrics_detailed.png")
    # plt.show()
    # Plotting
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(all_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(2, 3, 2)
    plt.plot(all_epsilons)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.subplot(2, 3, 3)
    plt.plot(all_steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.subplot(2, 3, 4)
    plt.plot(all_successes)
    plt.title("Success (Reached Goal)")
    plt.xlabel("Episode")
    plt.ylabel("Success (0 or 1)")

    plt.subplot(2, 3, 5)
    plt.plot(all_collisions)
    plt.title("Collisions per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Collisions")

    plt.subplot(2, 3, 6)
    plt.plot(all_coverage)
    plt.title("Coverage Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio [0-1]")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
