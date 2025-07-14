# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from env.uav_env import UAVEnv
# import time

# env = UAVEnv()
# state = env.reset()
# done = False

# for step in range(100):
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     print(f"Action: {action}, Reward: {reward:.3f}, Battery: {state[3]:.2f}")
#     env.render_2d(save=True, frame_id=step)
#     time.sleep(0.2)
#     if done:
#         print("Episode finished.\n")
#         break

# # env.render(use_3d=True)
# Phase 3
import os
import sys
import torch
import numpy as np

sys.path.append("env")
sys.path.append("train")

from uav_env import UAVEnv
from dqnn_agent import DQNNAgent

# Load environment
env = UAVEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load agent and model
agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True, prioritized=False)
agent.q_net.load_state_dict(torch.load("dqnn_uav_model.pth", map_location=torch.device('cpu')))
agent.q_net.eval()

# Run 1 episode
state = env.reset()
done = False
while not done:
    action = agent.select_action(state, train_mode=False)  # No exploration
    state, reward, done, _ = env.step(action)

# Show final trajectory
env.render_2d()
env.render(use_3d=True)
