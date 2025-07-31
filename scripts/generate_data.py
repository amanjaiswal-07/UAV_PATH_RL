
# import numpy as np
# import os

# grid_size = (20, 20, 5)
# data_dir = "./data_unseen"#chnage
# os.makedirs(data_dir, exist_ok=True)

# start_pos = (0, 0, 1)
# goal_pos = (19, 19, 1)

# def is_blocking(pos, box):
#     (x1, y1, z1), (x2, y2, z2) = box
#     return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2 and z1 <= pos[2] <= z2

# def generate_users(n_users=100):
#     users = np.random.randint(low=0, high=grid_size[0:2], size=(n_users, 2))
#     np.save(os.path.join(data_dir, "users.npy"), users)

# def generate_obstacles(n_obs=10):
#     obstacles = []
#     max_attempts = 1000  # prevent infinite loop in rare cases
#     attempts = 0
#     while len(obstacles) < n_obs and attempts < max_attempts:
#         x1, y1, z1 = np.random.randint(0, 15, size=3)
#         dx, dy, dz = np.random.randint(1, 5, size=3)
#         x2, y2, z2 = np.clip([x1+dx, y1+dy, z1+dz], [0,0,0], [g-1 for g in grid_size])
#         box = ((x1, y1, z1), (x2, y2, z2))
        
#         if not is_blocking(start_pos, box) and not is_blocking(goal_pos, box):
#             obstacles.append(box)
        
#         attempts += 1

#     np.save(os.path.join(data_dir, "obstacles.npy"), obstacles, allow_pickle=True)

# def generate_snr_map(n_sources=3):
#     snr_map = np.zeros(grid_size, dtype=np.float32)
#     for _ in range(n_sources):
#         cx, cy, cz = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]), np.random.randint(2, 5)
#         for x in range(grid_size[0]):
#             for y in range(grid_size[1]):
#                 for z in range(grid_size[2]):
#                     dist = np.linalg.norm([x-cx, y-cy, z-cz])
#                     snr_map[x, y, z] += np.exp(-dist**2 / (2*15))
#     snr_map = snr_map / np.max(snr_map)
#     np.save(os.path.join(data_dir, "channel_map.npy"), snr_map)

# if __name__ == "__main__":
#     generate_users()
#     generate_obstacles(n_obs=20)#chnage
#     generate_snr_map()
#     print("✅ Synthetic data generated.")

import numpy as np

# Grid size is defined here to be used by all generation functions
grid_size = (20, 20, 5)

def generate_users(n_users=100):
    """Generates and returns a random set of user locations."""
    users = np.random.randint(low=0, high=grid_size[0:2], size=(n_users, 2))
    return users

def generate_obstacles():
    """Generates and returns a random set of obstacles with a variable count."""
    obstacles = []
    # Randomize the number of obstacles for each environment
    num_obstacles = np.random.randint(8, 16) # Generate between 8 and 15 obstacles

    start_pos = (0, 0, 1)
    goal_pos = (19, 19, 1)

    while len(obstacles) < num_obstacles:
        x1, y1, z1 = np.random.randint(0, 15, size=3)
        dx, dy, dz = np.random.randint(1, 5, size=3)
        x2, y2, z2 = np.clip([x1+dx, y1+dy, z1+dz], [0,0,0], [g-1 for g in grid_size])

        is_start_blocked = (x1 <= start_pos[0] <= x2 and y1 <= start_pos[1] <= y2 and z1 <= start_pos[2] <= z2)
        is_goal_blocked = (x1 <= goal_pos[0] <= x2 and y1 <= goal_pos[1] <= y2 and z1 <= goal_pos[2] <= z2)

        if not is_start_blocked and not is_goal_blocked:
            obstacles.append(((x1, y1, z1), (x2, y2, z2)))
    
    return obstacles

def generate_snr_map(n_sources=3):
    """Generates and returns a 3D SNR map using a faster vectorized method."""
    x, y, z = np.arange(grid_size[0]), np.arange(grid_size[1]), np.arange(grid_size[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    snr_map = np.zeros(grid_size, dtype=np.float32)
    
    for _ in range(n_sources):
        cx, cy, cz = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]), np.random.randint(2, grid_size[2])
        dist_sq = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
        snr_map += np.exp(-dist_sq / (2 * 15**2))

    if np.max(snr_map) > 0:
        snr_map /= np.max(snr_map)

    return snr_map