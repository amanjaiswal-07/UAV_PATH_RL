import numpy as np
import matplotlib.pyplot as plt

# Load saved reward curves
beta0 = np.load('collisions_beta0.npy')
beta6 = np.load('collisions_beta6.npy')
beta4 = np.load('collisions_dueling_double_dqn.npy')
beta10 = np.load('collisions_beta10.npy')

episodes = np.arange(len(beta0))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(episodes, beta0, label='self.beta = 0.0')
plt.plot(episodes, beta6, label='self.beta = 6.0')
plt.plot(episodes, beta4, label='self.beta = 4.0')
plt.plot(episodes, beta10, label='self.beta = 10.0')


plt.xlabel("Episode")
plt.ylabel("collisions")
plt.title("Training Curves Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("collisions_comparison_ablation.png")
plt.show()
