# Phase 3 - Step 3: Prioritized Experience Replay
import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / sum(scaled_priorities)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()

        return samples, indices, weights
    
    def update_priorities(self, batch_indices, td_errors):
        """
        Update the priorities of the sampled experiences after a training step.
        """
        # Ensure errors are positive and add a small epsilon to avoid zero priority
        priorities = np.abs(td_errors) + self.epsilon
        
        for idx, priority in zip(batch_indices, priorities):
            # Ensure index is within the valid range
            if idx < len(self.priorities):
                self.priorities[idx] = priority
