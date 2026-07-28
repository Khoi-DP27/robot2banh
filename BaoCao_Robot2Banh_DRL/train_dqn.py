"""
===========================================================
  train_dqn.py — DQN Station-Keeping (Tự dừng >= 90%)
===========================================================
"""
import os, sys, time, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from envs.balance_bot_target_env import BalanceBotTargetEnv
from models.dqn_model import DQNetwork
from models.replay_buffer import ReplayBuffer

MODELS_DIR = "saved_models"
PLOTS_DIR  = "saved_plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Hyperparameters ──
LR           = 1e-3
GAMMA        = 0.99
BUFFER_SIZE  = 50_000
BATCH_SIZE   = 128
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.995
TARGET_UPDATE= 200       # steps
NUM_ACTIONS  = 11
MAX_TIME_SEC = 900.0     # 15 phút
PRINT_EVERY  = 50
TARGET_RATE  = 90.0
MIN_EP_CHECK = 1200
# ─────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("=" * 60)
    print("  DQN: ROBOT 2 BÁNH — SINGLE ZONE STATION-KEEPING")
    print(f"  Device: {device} | Tối đa: {MAX_TIME_SEC/60:.0f} phút")
    print("=" * 60, flush=True)

    env     = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    obs_dim = env.observation_space.shape[0]

    policy_net = DQNetwork(state_size=obs_dim, action_size=NUM_ACTIONS).to(device)
    target_net = DQNetwork(state_size=obs_dim, action_size=NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    buffer     = ReplayBuffer(BUFFER_SIZE)
    criterion  = nn.SmoothL1Loss()

    eps            = EPS_START
    rewards_hist   = []
    victories_hist = []
    best_reward    = -np.inf
    total_steps    = 0
    start_time     = time.time()
    ep             = 0

    while (time.time() - start_time) < MAX_TIME_SEC:
        ep += 1
        env.set_episode(ep)
        state, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0

        while not (done or trunc):
            total_steps += 1

            # Epsilon-greedy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q.argmax().item()

            nstate, r, done, trunc, info = env.step(action)
            buffer.push(state, action, r, nstate, float(done))
            state = nstate
            ep_reward += r

            # Train
            if len(buffer) >= BATCH_SIZE:
                s, a, rew, ns, d = buffer.sample(BATCH_SIZE)
                s   = torch.FloatTensor(s).to(device)
                a   = torch.LongTensor(a).to(device)
                rew = torch.FloatTensor(rew).to(device)
                ns  = torch.FloatTensor(ns).to(device)
                d   = torch.FloatTensor(d).to(device)

                q_vals  = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns).max(1)[0]
                    target = rew + GAMMA * next_q * (1 - d)

                loss = criterion(q_vals, target)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        eps = max(EPS_END, eps * EPS_DECAY)

        rewards_hist.append(ep_reward)
        victories_hist.append(1 if info.get("victory", False) else 0)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({"model_state_dict": policy_net.state_dict()},
                       os.path.join(MODELS_DIR, "dqn_target_model.pth"))

        win  = min(100, len(victories_hist))
        rate = np.mean(victories_hist[-win:]) * 100.0

        if ep % PRINT_EVERY == 0 or ep == 1:
            t_min = (time.time() - start_time) / 60.0
            avg_r = np.mean(rewards_hist[-win:])
            print(f"Ep {ep:5d} | {t_min:4.1f}m | Rew MA{win}: {avg_r:+8.1f} | "
                  f"Victory: {rate:5.1f}% | ε={eps:.3f} | "
                  f"Hover: {info.get('hover_counter',0):2d}/{env.HOVER_STEPS}",
                  flush=True)

        if ep >= MIN_EP_CHECK and rate >= TARGET_RATE:
            t_min = (time.time() - start_time) / 60.0
            print(f"\n🎯 DQN ĐẠT {rate:.1f}% tại Ep {ep} ({t_min:.1f} phút)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ DQN xong sau {elapsed/60:.1f} phút | {ep} episodes")
    _plot(rewards_hist, victories_hist, "DQN")


def _plot(rewards, victories, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{name} — Single Zone Station-Keeping", fontweight="bold")
    eps = np.arange(1, len(rewards)+1); w = 50
    ax1.plot(eps, rewards, alpha=0.2, color="#FF9800")
    if len(rewards) >= w:
        ax1.plot(range(w, len(rewards)+1),
                 np.convolve(rewards, np.ones(w)/w, "valid"),
                 color="#E65100", lw=2, label=f"MA-{w}")
    ax1.set_ylabel("Total Reward"); ax1.legend(); ax1.grid(alpha=0.3)
    v = np.array(victories, dtype=float)
    ax2.plot(eps, v*100, alpha=0.2, color="#9C27B0")
    if len(v) >= w:
        ax2.plot(range(w, len(v)+1),
                 np.convolve(v, np.ones(w)/w, "valid")*100,
                 color="#4A148C", lw=2, label=f"Victory% MA-{w}")
    ax2.axhline(90, color="red", ls="--", lw=1.5, label="90% Target")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Victory Rate (%)"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, f"{name.lower()}_target_training.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"✓ Đã lưu biểu đồ: {p}")


if __name__ == "__main__":
    train()
