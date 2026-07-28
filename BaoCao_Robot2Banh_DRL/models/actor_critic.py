"""
===========================================================
  ActorCriticNetwork — Mạng 2 đầu ra (Two-headed Network)
===========================================================
Kiến trúc:
                     ┌──> Actor Head (Linear -> Softmax) ──> π(a|s) [5 actions]
  Input(4) ──> Shared Trunk (128 -> 128)
                     └──> Critic Head (Linear) ───────────> V(s) [1 scalar]

Ưu điểm của thiết kế Shared Trunk:
  - Học biểu diễn đặc trưng dùng chung (shared representation) cho cả Policy và Value.
  - Tiết kiệm tham số và tăng tốc độ hội tụ.

Tham khảo:
  - Mnih et al. (2016) — "Asynchronous Methods for DRL" (A3C/A2C)
  - "Deep Reinforcement Learning in Action" (2020) — Chương 5 (Actor-Critic)
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Mạng Neural 2 đầu ra cho thuật toán Advantage Actor-Critic (A2C).

    Args:
        state_size (int): Số chiều trạng thái (= 4)
        action_size (int): Số lượng hành động (= 5)
        hidden_size (int): Số neuron lớp ẩn (= 128)
    """

    def __init__(self, state_size=4, action_size=5, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()

        # --- Shared Body (Thân dùng chung) ---
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # --- Actor Head (Đầu ra Policy: Xác suất action) ---
        self.actor_head = nn.Linear(hidden_size, action_size)

        # --- Critic Head (Đầu ra Value: Giá trị trạng thái V(s)) ---
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Forward pass tính toán đồng thời cả Policy distribution π(a|s) và State Value V(s).

        Args:
            state (torch.Tensor): State tensor (batch_size, 4) hoặc (4,)

        Returns:
            action_probs (torch.Tensor): Xác suất các action (batch_size, 5)
            state_value (torch.Tensor): Giá trị trạng thái V(s) (batch_size, 1)
        """
        # Shared trunk
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        # Actor head: Logits -> Softmax
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic head: Linear (không dùng activation)
        state_value = self.critic_head(x)

        return action_probs, state_value

    def evaluate_action(self, state, action, device):
        """
        Tính toán log_prob, entropy và state value cho trạng thái/hành động cho trước.
        """
        state_t = torch.FloatTensor(state).to(device)
        action_probs, state_value = self.forward(state_t)

        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, state_value
