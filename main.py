# import os
# import sys
# sys.path.append(os.path.abspath("env"))
# sys.path.append(os.path.abspath("scripts"))

# from uav_env import UAVEnv
# import generate_data  # Data generator module

# # === STEP 1: Generate Data ===
# print("Generating data...")
# generate_data.generate_users()
# generate_data.generate_obstacles()
# generate_data.generate_snr_map()
# print("Data generated.")

# # === STEP 2: Initialize Environment ===
# print("Initializing environment...")
# env = UAVEnv()
# state = env.reset()
# print("Environment ready.")

# # === STEP 3: Run simulation silently ===
# steps = 100
# for step in range(steps):
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break

# # === STEP 4: Show both 2D and 3D maps ===
# print("Showing 2D view...")
# env.render_2d(save=False)

# print("Showing 3D view...")
# env.render(use_3d=True)

# print("Simulation complete. Both views shown.")

import os
import sys

# === Extend path to access env/, scripts/, train/ ===
sys.path.append(os.path.abspath("env"))
sys.path.append(os.path.abspath("scripts"))
sys.path.append(os.path.abspath("train"))

from uav_env import UAVEnv
import generate_data
from train_dqn import train_dqn

if __name__ == "__main__":
    print("📡 Generating synthetic data...")
    generate_data.generate_users()
    generate_data.generate_obstacles()
    generate_data.generate_snr_map()
    print("✅ Data generation complete.\n")

    print("🛰️ Initializing UAV environment and RL agent...")
    env = UAVEnv()
    print("✅ Environment and agent ready.\n")

    print("🎮 Starting DQNN training...\n")
    train_dqn(env)
    print("✅ Training complete. Model saved.\n")

    print("📊 Showing final learned UAV path (2D and 3D)...")
    env.render_2d()
    env.render(use_3d=True)

