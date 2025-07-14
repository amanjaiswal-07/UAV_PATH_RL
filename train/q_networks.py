# Phase 3 - Step 1: Q-network and Dueling Q-network
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Dueling DQN

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()

        # Shared feature extraction layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Value stream: outputs a single scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream: outputs A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)# Shared layer
        values = self.value_stream(features) # V(s): shape [batch, 1]
        advantages = self.advantage_stream(features)# A(s,a): shape [batch, action_dim]

        # Combine V(s) and A(s,a) into Q(s,a)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
