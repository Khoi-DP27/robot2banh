# 🤖 Robot 2 Bánh Tự Cân Bằng & Di Chuyển Bám Target
> **Nghiên cứu, Triển khai và Đánh giá So sánh các Thuật toán Học Tăng Cường Sâu (DRL) với Bộ Điều Khiển PID trong Bài Toán Cân Bằng & Bám Target cho Robot 2 Bánh**

---

## 📌 Tổng Quan Dự Án

Dự án xây dựng môi trường mô phỏng động lực học vật lý (Gymnasium) cho **Robot 2 bánh tự cân bằng** dựa trên phương trình động lực học con lắc ngược Euler-Lagrange (Inverted Pendulum on Cart). 

Nhiệm vụ cốt lõi của Robot bao gồm **2 mục tiêu đồng thời**:
1. **Giữ cân bằng đứng thẳng:** Duy trì góc nghiêng $\theta \approx 0^\circ$ (không đổ quá $\pm 20^\circ$).
2. **Chủ động di chuyển tới điểm đích (Target Tracking):** Điều khiển vị trí $x$ chạy tiến/lùi tới điểm mục tiêu $x_{target}$ ngẫu nhiên trên đường đi.

---

## 🚀 Các Phương Pháp Điều Khiển Được Triển Khai

Dự án triển khai và so sánh **5 phương pháp điều khiển**:

| STT | Phương pháp | Phân loại | Mô tả |
| :---: | :--- | :--- | :--- |
| 1 | **PID Controller** | Baseline truyền thống | Bộ điều khiển PID kép (Dual-loop Cascaded PID) |
| 2 | **DQN** | Value-based DRL | Deep Q-Network với Replay Buffer & Target Network |
| 3 | **REINFORCE** | Policy-based DRL | Monte-Carlo Policy Gradient |
| 4 | **A2C** | Actor-Critic | Synchronous Advantage Actor-Critic |
| 5 | **PPO** | Advanced Policy DRL | Proximal Policy Optimization (Clipped Surrogate Objective) |

---

## 🏆 Bảng Kết Quả Đánh Giá Tổng Hợp

*(Được xuất từ script [compare_all.py](compare_all.py) trên 100 tập đánh giá độc lập)*

| Thuật toán | Phân loại | Reward Eval | Số bước sống sót (/500) | Tỷ lệ sống sót |
| :--- | :--- | :---: | :---: | :---: |
| **PID** | Baseline | $+72.1 \pm 34.8$ | $123.1$ / $500$ | $0.0\%$ |
| **DQN** | Value-based DRL | $+468.9 \pm 1.2$ | $500.0$ / $500$ | **$100.0\%$** |
| **REINFORCE** | Policy Gradient | $+321.8 \pm 149.5$ | $451.1$ / $500$ | $65.0\%$ |
| **A2C** | Actor-Critic | $+461.8 \pm 19.9$ | $500.0$ / $500$ | **$100.0\%$** |
| **PPO** | Proximal Policy Opt | $+452.9 \pm 2.9$ | $500.0$ / $500$ | **$100.0\%$** |

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
F:\ROBOT 2 Banh\
├── controllers/
│   └── pid_controller.py        # Bộ điều khiển PID kép (Dual-loop)
├── envs/
│   ├── balance_bot_env.py       # Môi trường Gymnasium cơ bản
│   ├── balance_bot_target_env.py# Môi trường bám điểm Target (Cốt lõi)
│   └── balance_bot_hard_env.py  # Môi trường Hard Mode (nhiễu, dốc, lực xô)
├── models/
│   ├── dqn_model.py             # Mạng DQN
│   ├── replay_buffer.py         # Replay Buffer cho DQN
│   ├── policy_gradient.py      # Mạng REINFORCE
│   ├── actor_critic.py         # Mạng A2C
│   └── ppo.py                  # Mạng PPO
├── train_target_tracking.py     # Huấn luyện PPO bài toán bám Target
├── demo_target_tracking.py      # Demo chạy xe bám Target thực tế
├── record_target_video.py       # Ghi hình Video HD 720p & GIF mô phỏng Target
├── compare_all.py               # Script vẽ đồ thị so sánh 4 DRL vs PID
├── benchmark_hard_mode.py       # Script kiểm thử độ bền vững (Robustness)
├── record_videos.py             # Script xuất video tổng hợp các mô hình
├── run_pid.py                   # Script kiểm thử PID Controller
└── requirements.txt             # Khai báo các thư viện Python
```

---

## 💻 Hướng Dẫn Sử Dụng & Chạy Demo

### 1. Cài đặt môi trường
Yêu cầu Python 3.8+ và các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. CHẠY TỰ ĐỘNG TOÀN BỘ DỰ ÁN TRONG 1 CÂU LỆNH (Recommeded)
```bash
python run_all.py
```
*(Script sẽ tự động train lần lượt 4 thuật toán, vẽ biểu đồ so sánh, xuất video HD 720p & GIF và chạy demo kết quả)*

### 3. Chạy từng câu lệnh riêng lẻ
```bash
python train_ppo.py              # Train PPO Hard Mode
python train_target_tracking.py   # Train PPO Bám Target
python compare_all.py             # Xuất bảng so sánh & đồ thị hội tụ
python record_target_video.py    # Xuất Video HD & GIF 🚩
python demo_target_tracking.py    # Chạy demo thực tế
```

---

## 🎬 Trực Quan Hóa Thực Nghiệm

- 🎥 **Video Demo Bám Target:** `saved_videos/target_tracking_ppo.mp4` & `saved_videos/target_tracking_ppo.gif`
- 📈 **Đồ thị Tiến trình Hội tụ:** `saved_plots/overall_convergence_comparison.png`
- 🎯 **Đồ thị Huấn luyện Target Tracking:** `saved_plots/target_tracking_training.png`
- 🛡️ **Đồ thị Đánh giá Độ bền vững (Hard Mode):** `saved_plots/hard_mode_benchmark.png`

---

## 📜 Giấy Phép & Tác Quyền
Được phát triển phục vụ Bài tập lớn môn học Robot & Học Tăng Cường Sâu.
