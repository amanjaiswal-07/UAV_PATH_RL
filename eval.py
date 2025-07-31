# import torch
# import os
# import numpy as np
# import imageio
# import glob
# from env.uav_env import UAVEnv
# from train.dqnn_agent import DQNNAgent
# from train.q_networks import QNetwork, DuelingQNetwork


# def evaluate_agent(model_path, data_dir, episodes=10, gif_dir="eval_gifs"):
#     env = UAVEnv(data_dir=data_dir)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     # Load trained agent
#     agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True)
#     agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     agent.q_net.eval()

#     os.makedirs(gif_dir, exist_ok=True)

#     # Stats
#     all_rewards, all_steps, all_collisions, all_coverage = [], [], [], []
#     successes = 0

#     for ep in range(1, episodes + 1):
#         state = env.reset()
#         done = False
#         step_count, total_reward, collisions = 0, 0, 0
#         goal_reached = False
#         frames = []

#         while not done:
#             action = agent.select_action(state, train_mode=False)
#             next_state, reward, done, info = env.step(action)
#             state = next_state
#             total_reward += reward
#             step_count += 1

#             if info.get("collision", False):
#                 collisions += 1

#             if done and np.array_equal(env.uav_pos, env.goal):
#                 goal_reached = True

#             # Render frame
#             env.render_2d(save=True, frame_id=step_count)
#             frame_path = f"frames/frame_{step_count:03d}.png"
#             frames.append(imageio.imread(frame_path))

#             if step_count > 200:
#                 print(f"Episode {ep} exceeded 200 steps.")
#                 break

#         # Save GIF
#         gif_path = os.path.join(gif_dir, f"episode_{ep:02d}.gif")
#         imageio.mimsave(gif_path, frames, fps=5)

#         # Cleanup
#         for file in glob.glob("frames/*.png"):
#             os.remove(file)

#         print(f"🎞️ Saved: {gif_path} | Steps: {step_count}, Reward: {total_reward:.2f}, Success: {goal_reached}")

#         # Record stats
#         if goal_reached: successes += 1
#         all_rewards.append(total_reward)
#         all_steps.append(step_count)
#         all_collisions.append(collisions)
#         all_coverage.append(len(env.covered_area) / (env.grid_size[0] * env.grid_size[1]))

#     # Summary
#     print("\n📊 --- Evaluation Summary ---")
#     print(f"Total Episodes:         {episodes}")
#     print(f"Success Rate:           {100.0 * successes / episodes:.1f}%")
#     print(f"Avg Reward:             {np.mean(all_rewards):.2f}")
#     print(f"Avg Steps:              {np.mean(all_steps):.1f}")
#     print(f"Avg Collisions:         {np.mean(all_collisions):.1f}")
#     print(f"Avg Coverage Ratio:     {np.mean(all_coverage):.3f}")
#     print("-------------------------------")

# if __name__ == "__main__":
#     evaluate_agent(
#         model_path="best_model.pth",
#         data_dir="data_unseen",
#         episodes=1,
#         gif_dir="eval_gifs2"
#     )
import torch
import os
import numpy as np
import imageio
import glob
from env.uav_env import UAVEnv
from train.dqnn_agent import DQNNAgent

def evaluate_agent(model_path, base_env_dir="data_eval_difficulty", episodes=10, gif_dir="eval_gifs_difficulty"):
    os.makedirs(gif_dir, exist_ok=True)

    all_rewards, all_steps, all_collisions, all_coverage = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
        env_dir = os.path.join(base_env_dir, f"env_{ep:02d}")
        env = UAVEnv(data_dir=env_dir)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True)
        agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.q_net.eval()

        state = env.reset()
        done = False
        step_count, total_reward, collisions = 0, 0, 0
        goal_reached = False
        frames = []

        while not done:
            action = agent.select_action(state, train_mode=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            step_count += 1

            if info.get("collision", False):
                collisions += 1

            if done and np.array_equal(env.uav_pos, env.goal):
                goal_reached = True

            env.render_2d(save=True, frame_id=step_count)
            frame_path = f"frames/frame_{step_count:03d}.png"
            frames.append(imageio.imread(frame_path))

            if step_count > 200:
                print(f"Episode {ep} exceeded 200 steps.")
                break

        gif_path = os.path.join(gif_dir, f"episode_{ep:02d}.gif")
        imageio.mimsave(gif_path, frames, fps=5)

        for file in glob.glob("frames/*.png"):
            os.remove(file)

        # print(f"🎞️ Saved: {gif_path} | Steps: {step_count}, Reward: {total_reward:.2f}, Success: {goal_reached}")
        print(f"🎞️ Saved: {gif_path}")
        print(f"Ep {ep} — Obstacles: {len(env.obstacles)}, Reward: {total_reward:.2f}, "f"Steps: {step_count}, Collisions: {collisions}, "f"Coverage: {len(env.covered_area)/(env.grid_size[0]*env.grid_size[1]):.3f}, "f"Success: {goal_reached}")

        if goal_reached:
            successes += 1
        all_rewards.append(total_reward)
        all_steps.append(step_count)
        all_collisions.append(collisions)
        all_coverage.append(len(env.covered_area) / (env.grid_size[0] * env.grid_size[1]))

    print("\n📊 --- Evaluation Summary ---")
    print(f"Total Episodes:         {episodes}")
    print(f"Success Rate:           {100.0 * successes / episodes:.1f}%")
    print(f"Avg Reward:             {np.mean(all_rewards):.2f}")
    print(f"Avg Steps:              {np.mean(all_steps):.1f}")
    print(f"Avg Collisions:         {np.mean(all_collisions):.1f}")
    print(f"Avg Coverage Ratio:     {np.mean(all_coverage):.3f}")
    print(f"Avg Collisions/Step:     {np.mean(all_collisions)/np.mean(all_steps):.3f}")
    print("-------------------------------")

if __name__ == "__main__":
    evaluate_agent(
        model_path="best_model.pth",
        base_env_dir="data_eval_difficulty",
        episodes=20,
        gif_dir="eval_gifs_difficulty10"
    )
