"""
===========================================================
  DQNetwork — Mạng Deep Q-Network bằng PyTorch
===========================================================
Kiến trúc mạng Feedforward:

  Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(5)
     ↑                                                      ↑
  [x, ẋ, θ, θ̇]                                    Q(s,a) cho mỗi action

Mạng nhận trạng thái s và xuất Q-value cho TẤT CẢ action cùng lúc.
Action tối ưu = argmax Q(s, a)

Hai mạng được dùng (Double DQN concept):
  1. Policy Network (online): Chọn action, được train mỗi step
  2. Target Network: Tính target Q-value, chỉ update định kỳ
     → Giúp ổn định training, tránh "moving target" problem

Tham khảo:
  - Mnih et al. (2015) — DQN gốc (Nature paper)
  - Van Hasselt et al. (2016) — Double DQN
  - "Deep Reinforcement Learning in Action" (2020) — Chương 3, 4
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    """
    Feedforward Neural Network cho Deep Q-Learning.

    Kiến trúc:
      Input(state_size) → FC1(hidden) → ReLU → FC2(hidden) → ReLU → FC3(action_size)

    Args:
        state_size (int): Số chiều của state vector (= 4)
        action_size (int): Số lượng action rời rạc (= 5)
        hidden_size (int): Số neuron mỗi hidden layer (= 128)
    """

    def __init__(self, state_size=4, action_size=5, hidden_size=128):
        super(DQNetwork, self).__init__()

        # ============================================================
        # KIẾN TRÚC MẠNG
        # ============================================================
        # Layer 1: Input → Hidden (4 → 128)
        #   Chuyển vector trạng thái thô thành biểu diễn ẩn
        self.fc1 = nn.Linear(state_size, hidden_size)

        # Layer 2: Hidden → Hidden (128 → 128)
        #   Học các đặc trưng phi tuyến phức tạp hơn
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Layer 3: Hidden → Output (128 → 5)
        #   Xuất Q-value cho mỗi action
        #   KHÔNG có activation function ở output (Q-value là số thực)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Forward pass: tính Q-values cho tất cả actions.

        Args:
            state (torch.Tensor): Batch trạng thái — shape (batch_size, 4)

        Returns:
            q_values (torch.Tensor): Q-values — shape (batch_size, 5)
                q_values[i][j] = Q(state_i, action_j)
                = "Giá trị kỳ vọng của tổng reward tương lai
                   nếu ở state_i và chọn action_j"

        Ví dụ:
            state = [0.01, -0.02, 0.03, -0.01]  (gần cân bằng)
            q_values = [-12.3, -5.1, 15.2, -4.8, -11.9]
            → Action tốt nhất = 2 (không tác dụng lực, vì đã gần cân bằng)
        """
        x = F.relu(self.fc1(state))   # (batch, 4) → (batch, 128) + ReLU
        x = F.relu(self.fc2(x))       # (batch, 128) → (batch, 128) + ReLU
        q_values = self.fc3(x)        # (batch, 128) → (batch, 5), NO activation
        return q_values
