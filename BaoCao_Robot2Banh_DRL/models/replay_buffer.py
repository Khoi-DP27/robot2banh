"""
===========================================================
  ReplayBuffer — Bộ nhớ Experience Replay cho DQN
===========================================================
Experience Replay là kỹ thuật cốt lõi giúp DQN học ổn định:

Vấn đề nếu KHÔNG có Replay:
  - Các mẫu (s, a, r, s') liên tiếp có tương quan cao
  - Mạng neural bị "catastrophic forgetting" — quên kinh nghiệm cũ
  - Gradient thiên lệch, mô hình dao động không hội tụ

Giải pháp Experience Replay:
  - Lưu tất cả transition (s, a, r, s', done) vào buffer
  - Khi train, lấy NGẪU NHIÊN 1 batch từ buffer
  - Phá vỡ tương quan thời gian → gradient ổn định hơn

Tham khảo:
  - Mnih et al. (2015) — "Human-level control through DRL"
  - "Deep Reinforcement Learning in Action" (2020) — Chương 3
===========================================================
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Circular buffer lưu trữ các transition (s, a, r, s', done).
    
    Khi buffer đầy, các transition cũ nhất sẽ bị ghi đè (FIFO).
    Hỗ trợ lấy mẫu ngẫu nhiên (uniform sampling) cho training.

    Args:
        capacity (int): Số lượng transition tối đa lưu trữ
        seed (int): Random seed (để reproducible)
    """

    def __init__(self, capacity=10000, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """
        Lưu 1 transition vào buffer.

        Args:
            state (np.ndarray): Trạng thái hiện tại s(t) — shape (4,)
            action (int): Hành động đã chọn a(t) — [0, 1, 2, 3, 4]
            reward (float): Phần thưởng r(t)
            next_state (np.ndarray): Trạng thái tiếp theo s(t+1) — shape (4,)
            done (bool): Episode kết thúc? (terminated hoặc truncated)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Lấy ngẫu nhiên 1 batch từ buffer.

        Args:
            batch_size (int): Kích thước batch

        Returns:
            states (np.ndarray): Mảng trạng thái — shape (batch_size, 4)
            actions (np.ndarray): Mảng action — shape (batch_size,)
            rewards (np.ndarray): Mảng reward — shape (batch_size,)
            next_states (np.ndarray): Mảng trạng thái tiếp — shape (batch_size, 4)
            dones (np.ndarray): Mảng done flags — shape (batch_size,)
        """
        batch = random.sample(self.buffer, batch_size)

        # Tách batch thành các thành phần riêng biệt
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """Số lượng transition hiện tại trong buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Kiểm tra buffer đã đủ mẫu để train chưa."""
        return len(self.buffer) >= batch_size
