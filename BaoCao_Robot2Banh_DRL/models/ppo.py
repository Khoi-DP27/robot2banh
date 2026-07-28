"""
===========================================================
  PPONetwork — Mạng PPO (Actor-Critic với Clipped Objective)
===========================================================
Kiến trúc:
  Input(4) ──> Shared Trunk (128 -> 128)
                     ├──> Actor Head (Linear -> Softmax)
                     └──> Critic Head (Linear)
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    def __init__(self, state_size=4, action_size=5, hidden_size=256):
        super(PPONetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)

        return probs, value

    def get_action_and_value(self, state, action=None, device="cpu"):
        state_t = torch.FloatTensor(state).to(device)
        probs, value = self.forward(state_t)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value
