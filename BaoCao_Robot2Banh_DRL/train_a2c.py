"""
===========================================================
  train_a2c.py — A2C Station-Keeping & Navigation với GAE
===========================================================
"""
import os, sys, time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from envs.balance_bot_target_env import BalanceBotTargetEnv
from models.actor_critic import ActorCriticNetwork

MODELS_DIR = "saved_models"
PLOTS_DIR  = "saved_plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Hyperparameters ──
LR           = 3e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95      # Generalized Advantage Estimation
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
NUM_ACTIONS  = 11
MAX_TIME_SEC = 1800.0    # 30 phút
PRINT_EVERY  = 50
TARGET_RATE  = 90.0
MIN_EP_CHECK = 1000
# ─────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("=" * 60)
    print("  A2C (với GAE): ROBOT 2 BÁNH — STATION-KEEPING & NAVIGATION")
    print(f"  Device: {device} | Tối đa: {MAX_TIME_SEC/60:.0f} phút")
    print("=" * 60, flush=True)

    env     = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    obs_dim = env.observation_space.shape[0]

    model     = ActorCriticNetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    rewards_hist   = []
    victories_hist = []
    best_reward    = -np.inf
    start_time     = time.time()
    ep             = 0

    while (time.time() - start_time) < MAX_TIME_SEC:
        ep += 1
        env.set_episode(ep)
        state, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0

        log_probs, values, rewards, masks, entropies = [], [], [], [], []

        while not (done or trunc):
            state_t  = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, val = model(state_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            nstate, r, done, trunc, info = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(val.squeeze())
            rewards.append(r)
            masks.append(1.0 - float(done))
            entropies.append(dist.entropy())

            state = nstate
            ep_reward += r

        # Compute GAE Advantages & Returns
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        vals_np = torch.stack(values).detach().cpu().numpy().reshape(-1)
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0
                next_non_terminal = 0.0
            else:
                next_val = float(vals_np[t + 1])
                next_non_terminal = float(masks[t])
            delta = float(rewards[t] + GAMMA * next_val * next_non_terminal - vals_np[t])
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae

        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t    = advantages_t + torch.stack(values).reshape(-1)

        # Standardize advantages
        advantages_norm = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs).reshape(-1)
        entropies_t = torch.stack(entropies).reshape(-1)

        actor_loss   = -(log_probs_t * advantages_norm).mean()
        critic_loss  = F.mse_loss(torch.stack(values).reshape(-1), returns_t.detach())
        entropy_loss = -entropies_t.mean()

        loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        rewards_hist.append(ep_reward)
        victories_hist.append(1 if info.get("victory", False) else 0)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({"model_state_dict": model.state_dict()},
                       os.path.join(MODELS_DIR, "a2c_target_model.pth"))

        win  = min(100, len(victories_hist))
        rate = np.mean(victories_hist[-win:]) * 100.0

        if ep % PRINT_EVERY == 0 or ep == 1:
            t_min = (time.time() - start_time) / 60.0
            avg_r = np.mean(rewards_hist[-win:])
            print(f"Ep {ep:5d} | {t_min:4.1f}m | Rew MA{win}: {avg_r:+8.1f} | "
                  f"Victory: {rate:5.1f}% | "
                  f"Hover: {info.get('hover_counter',0):2d}/{env.HOVER_STEPS}",
                  flush=True)

        if ep >= MIN_EP_CHECK and rate >= TARGET_RATE:
            t_min = (time.time() - start_time) / 60.0
            print(f"\n🎯 A2C ĐẠT {rate:.1f}% tại Ep {ep} ({t_min:.1f} phút)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ A2C xong sau {elapsed/60:.1f} phút | {ep} episodes")
    _plot(rewards_hist, victories_hist, "A2C")


def _plot(rewards, victories, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{name} (với GAE) — Single Zone Station-Keeping", fontweight="bold")
    eps = np.arange(1, len(rewards)+1); w = 50
    ax1.plot(eps, rewards, alpha=0.2, color="#4CAF50")
    if len(rewards) >= w:
        ax1.plot(range(w, len(rewards)+1),
                 np.convolve(rewards, np.ones(w)/w, "valid"),
                 color="#1B5E20", lw=2, label=f"MA-{w}")
    ax1.set_ylabel("Total Reward"); ax1.legend(); ax1.grid(alpha=0.3)
    v = np.array(victories, dtype=float)
    ax2.plot(eps, v*100, alpha=0.2, color="#2196F3")
    if len(v) >= w:
        ax2.plot(range(w, len(v)+1),
                 np.convolve(v, np.ones(w)/w, "valid")*100,
                 color="#0D47A1", lw=2, label=f"Victory% MA-{w}")
    ax2.axhline(90, color="red", ls="--", lw=1.5, label="90% Target")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Victory Rate (%)"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, f"{name.lower()}_target_training.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"✓ Đã lưu biểu đồ: {p}")


if __name__ == "__main__":
    train()
