import numpy as np
import gym
import os
from gym import spaces
import matplotlib.pyplot as plt

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        self.grid_size = np.array([20, 20, 5])
        self.battery = 1.0

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([
                self.grid_size[0]-1,
                self.grid_size[1]-1,
                self.grid_size[2]-1,
                1.0,
                1.0,
                1.0,
                np.linalg.norm(self.grid_size),
                1.0
            ]),
            dtype=np.float32
        )

        self.snr_threshold = 0.5
        self.alpha = 1.0
        self.beta = 2.0
        self.gamma = 5.0
        self.delta = 1.0

        self.reset()

    def reset(self):
        self.uav_pos = np.array([0, 0, 1])
        self.goal = np.array([19, 19, 1])
        self.battery = 1.0
        self.covered_area = set()
        self.covered_area.add(tuple(self.uav_pos[:2]))

        self.users = np.load("data/users.npy")
        self.obstacles = np.load("data/obstacles.npy", allow_pickle=True)
        self.snr_map = np.load("data/channel_map.npy")
        self.trajectory = [tuple(self.uav_pos)]
        self.collisions = []  # Track attempted blocked moves
        return self.get_state()

    def get_state(self):
        x, y, z = self.uav_pos
        snr = self.snr_map[x, y, z]
        coverage_ratio = len(self.covered_area) / (self.grid_size[0] * self.grid_size[1])
        distance = np.linalg.norm(self.goal - self.uav_pos)
        obstacle_risk = self.check_obstacle(self.uav_pos)
        return np.array([x, y, z, self.battery, snr, coverage_ratio, distance, obstacle_risk], dtype=np.float32)

    def step(self, action):
        move_map = {
            0: [0, 0, 1],
            1: [0, 0, -1],
            2: [-1, 0, 0],
            3: [1, 0, 0],
            4: [1, 1, 0],
            5: [1, -1, 0],
            6: [0, 0, 0],
        }

        move = move_map[action]
        new_pos = self.uav_pos + np.array(move)
        new_pos = np.clip(new_pos, [0, 0, 0], self.grid_size - 1)

        collision = self.check_obstacle(new_pos)
        energy_used = 0.01 if action != 6 else 0.005
        self.battery -= energy_used
        self.battery = max(self.battery, 0)

        if not collision:
            self.uav_pos = new_pos
            self.trajectory.append(tuple(self.uav_pos))  # Only track when move happens
        else:
            self.collisions.append(tuple(new_pos))       # Track attempted blocked move

        is_new_area = tuple(self.uav_pos[:2]) not in self.covered_area
        if is_new_area:
            self.covered_area.add(tuple(self.uav_pos[:2]))

        snr = self.snr_map[tuple(self.uav_pos)]
        reward = self.compute_reward(snr, is_new_area, collision, energy_used)
        done = np.array_equal(self.uav_pos, self.goal) or self.battery <= 0

        return self.get_state(), reward, done, {}

    def compute_reward(self, snr, is_new_area, collision, energy_used):
        reward = 0.0
        if snr > self.snr_threshold:
            reward += self.alpha
        if is_new_area:
            reward += self.beta
        if collision:
            reward -= self.gamma
        reward -= self.delta * energy_used
        return reward

    def check_obstacle(self, pos):
        for box in self.obstacles:
            (x1, y1, z1), (x2, y2, z2) = box
            if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2 and z1 <= pos[2] <= z2:
                return 1.0
        return 0.0

    def render_2d(self, save=False, frame_id=0):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_title("UAV Top-Down View (2D)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        for user in self.users:
            ax.plot(user[0], user[1], 'bo', markersize=4)

        for box in self.obstacles:
            (x1, y1, _), (x2, y2, _) = box
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, color='black', alpha=0.3)
            ax.add_patch(rect)

        traj_x = [pos[0] for pos in self.trajectory]
        traj_y = [pos[1] for pos in self.trajectory]
        ax.plot(traj_x, traj_y, 'y--', linewidth=1, label='Trajectory')

        # Plot attempted collisions as red X
        for (x, y, _) in self.collisions:
            ax.plot(x, y, 'rx', markersize=6, label='Collision' if 'Collision' not in ax.get_legend_handles_labels()[1] else "")

        x, y, _ = self.uav_pos
        ax.plot(x, y, 'go', markersize=8, label="UAV")

        gx, gy, _ = self.goal
        ax.plot(gx, gy, 'ro', markersize=8, label="Goal")

        ax.legend()

        if save:
            os.makedirs("frames", exist_ok=True)
            plt.savefig(f"frames/frame_{frame_id:03d}.png")

        plt.close(fig) if save else plt.show()


    from mpl_toolkits.mplot3d import Axes3D  # Add this at the top

    def render(self, use_3d=False):
        if not use_3d:
            return self.render_2d()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("UAV 3D View")
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_zlim(0, self.grid_size[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Users
        for user in self.users:
            ax.scatter(user[0], user[1], 0, c='blue', s=10)

        # Obstacles (cuboids)
        for box in self.obstacles:
            (x1, y1, z1), (x2, y2, z2) = box
            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            ax.bar3d(x1, y1, z1, dx, dy, dz, color='gray', alpha=0.3)

        # Covered area (top layer)
        for (x, y) in self.covered_area:
            ax.scatter(x, y, 1, c='yellow', s=5)

        # Trajectory
        traj_x = [pos[0] for pos in self.trajectory]
        traj_y = [pos[1] for pos in self.trajectory]
        traj_z = [pos[2] for pos in self.trajectory]
        ax.plot(traj_x, traj_y, traj_z, 'y--', label="Trajectory")

        # Collisions
        for (x, y, z) in self.collisions:
            ax.scatter(x, y, z, c='red', marker='x', s=40, label="Collision" if 'Collision' not in ax.get_legend_handles_labels()[1] else "")

        # UAV position
        x, y, z = self.uav_pos
        ax.scatter(x, y, z, c='green', s=50, label="UAV")

        # Goal
        gx, gy, gz = self.goal
        ax.scatter(gx, gy, gz, c='red', s=50, label="Goal")

        ax.legend()
        plt.show()

