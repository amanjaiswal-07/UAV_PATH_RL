
import numpy as np
import os

grid_size = (20, 20, 5)
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

def generate_users(n_users=100):
    users = np.random.randint(low=0, high=grid_size[0:2], size=(n_users, 2))
    np.save(os.path.join(data_dir, "users.npy"), users)

def generate_obstacles(n_obs=10):
    obstacles = []
    for _ in range(n_obs):
        x1, y1, z1 = np.random.randint(0, 15, size=3)
        dx, dy, dz = np.random.randint(1, 5, size=3)
        x2, y2, z2 = np.clip([x1+dx, y1+dy, z1+dz], [0,0,0], grid_size)
        obstacles.append(((x1, y1, z1), (x2, y2, z2)))
    np.save(os.path.join(data_dir, "obstacles.npy"), obstacles, allow_pickle=True)

def generate_snr_map(n_sources=3):
    snr_map = np.zeros(grid_size, dtype=np.float32)
    for _ in range(n_sources):
        cx, cy, cz = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]), np.random.randint(2, 5)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    dist = np.linalg.norm([x-cx, y-cy, z-cz])
                    snr_map[x, y, z] += np.exp(-dist**2 / (2*15))
    snr_map = snr_map / np.max(snr_map)
    np.save(os.path.join(data_dir, "channel_map.npy"), snr_map)

if __name__ == "__main__":
    generate_users()
    generate_obstacles()
    generate_snr_map()
    print("✅ Synthetic data generated.")
