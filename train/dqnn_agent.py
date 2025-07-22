# Phase 3 - Step 2: DQNNAgent class with Dueling & Double DQN support
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from q_networks import QNetwork, DuelingQNetwork
from prioritized_replay import PrioritizedReplayBuffer

class DQNNAgent:
    def __init__(self, state_dim, action_dim, dueling=True, double=True, prioritized=False, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.replay_capacity = 10000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.double = double
        self.target_update_freq = target_update_freq
        self.step_count = 0

        net_class = DuelingQNetwork if dueling else QNetwork
        self.q_net = net_class(state_dim, action_dim)
        self.target_net = net_class(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        if prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_capacity)
            self.use_per = True
        else:
            # self.replay_buffer = []
            # self.use_per = False
            # Use deque for an efficient standard buffer
            self.replay_buffer = deque(maxlen=self.replay_capacity)
            self.use_per = False

    # def select_action(self, state):
    #     if np.random.rand() < self.epsilon:
    #         return random.randint(0, self.action_dim - 1)
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         q_values = self.q_net(state_tensor)
    #     return q_values.argmax().item()

    def select_action(self, state, train_mode=True):
        if train_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def store(self, transition):
        if self.use_per:
            self.replay_buffer.add(*transition)
        else:
            # if len(self.replay_buffer) >= self.replay_capacity:
            #     self.replay_buffer.pop(0)
            self.replay_buffer.append(transition)

    def train(self):
        if self.use_per:
            if len(self.replay_buffer.buffer) < self.batch_size:
                return
            transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            if len(self.replay_buffer) < self.batch_size:
                return
            transitions = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        curr_Q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_Q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_Q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_Q = rewards + self.gamma * (1 - dones) * next_Q

        # loss = self.loss_fn(curr_Q, target_Q)
        if self.use_per:
            # Calculate TD errors (the difference between target and current Q values)
            td_errors = target_Q - curr_Q

            # Calculate the weighted MSE loss
            # The loss for each sample is squared error, multiplied by its importance weight
            loss = (td_errors.pow(2) * weights).mean()

            # Update the priorities in the replay buffer with the new errors
            # We detach errors from the graph before converting to numpy
            self.replay_buffer.update_priorities(indices, td_errors.detach().squeeze().cpu().numpy())
        else:
            # Original loss calculation for the standard replay buffer
            loss = self.loss_fn(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    