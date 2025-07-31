import os
import numpy as np

# Configuration
grid_size = (20, 20, 5)
base_data_dir = "./data_eval_difficulty"

# Ensure the base directory exists
os.makedirs(base_data_dir, exist_ok=True)

def generate_users(n_users=100):
    return np.random.randint(low=0, high=grid_size[0:2], size=(n_users, 2))

def generate_obstacles(n_obs=10):
    obstacles = []
    while len(obstacles) < n_obs:
        x1, y1, z1 = np.random.randint(0, 15, size=3)
        dx, dy, dz = np.random.randint(1, 5, size=3)
        x2, y2, z2 = np.clip([x1+dx, y1+dy, z1+dz], [0,0,0], [g-1 for g in grid_size])
        obstacles.append(((x1, y1, z1), (x2, y2, z2)))
    return obstacles

def generate_snr_map(n_sources=3):
    snr_map = np.zeros(grid_size, dtype=np.float32)
    for _ in range(n_sources):
        cx, cy, cz = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]), np.random.randint(2, grid_size[2])
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    dist = np.linalg.norm([x-cx, y-cy, z-cz])
                    snr_map[x, y, z] += np.exp(-dist**2 / (2*15))
    return snr_map / np.max(snr_map)

# Generate 10 environments with increasing difficulty
for ep in range(20):
    env_dir = os.path.join(base_data_dir, f"env_{ep+1:02d}")
    os.makedirs(env_dir, exist_ok=True)
    np.save(os.path.join(env_dir, "users.npy"), generate_users())
    np.save(os.path.join(env_dir, "obstacles.npy"), generate_obstacles(n_obs=10 + ep), allow_pickle=True)
    np.save(os.path.join(env_dir, "channel_map.npy"), generate_snr_map())

env_dirs = sorted(os.listdir(base_data_dir))
env_dirs = [os.path.join(base_data_dir, d) for d in env_dirs if os.path.isdir(os.path.join(base_data_dir, d))]
env_dirs[:5]  # Show first 5 generated env paths for confirmation
