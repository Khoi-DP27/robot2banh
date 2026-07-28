"""
===========================================================
  train_target_tracking.py — PPO 15 phút (Tự dừng >= 90%)
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
from models.ppo import PPONetwork

MODELS_DIR = "saved_models"
PLOTS_DIR  = "saved_plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Hyperparameters ──
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
LR           = 3e-4
ENTROPY_COEF = 0.015
VALUE_COEF   = 0.5
K_EPOCHS     = 6
BATCH_SIZE   = 128
NUM_ACTIONS  = 11
MAX_TIME_SEC = 900.0   # 15 phut
PRINT_EVERY  = 50
TARGET_RATE  = 90.0
MIN_EP_START = 300
# ─────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gae(rewards, values, dones, next_val):
    adv, gae = [], 0
    vals = values + [next_val]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * vals[t+1] * (1 - dones[t]) - vals[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        adv.insert(0, gae)
    return torch.tensor(adv, dtype=torch.float32)


def train():
    print("=" * 65)
    print("  PPO ROBOT 2 BÁNH — MULTI-STEP STATION-KEEPING")
    print(f"  Device: {device} | Tối đa: {MAX_TIME_SEC/60:.0f} phút")
    print(f"  Tự dừng khi Victory Rate >= {TARGET_RATE}%")
    print("=" * 65, flush=True)

    env     = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    obs_dim = env.observation_space.shape[0]

    # obs_dim tự động từ env (7 chiều: x, xdot, theta, thetadot, dist, cp, hover_norm)
    model     = PPONetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)
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

        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        while not (done or trunc):
            act_t, lp_t, _, val_t = model.get_action_and_value(state, device=device)
            act = act_t.item()
            nstate, r, done, trunc, info = env.step(act)

            states.append(state); actions.append(act)
            log_probs.append(lp_t.detach()); rewards.append(r)
            dones.append(float(done)); values.append(val_t.detach().item())

            state = nstate
            ep_reward += r

        # PPO update
        with torch.no_grad():
            _, nv = model(torch.FloatTensor(nstate).to(device))
        adv     = compute_gae(rewards, values, dones, nv.item()).to(device)
        returns = adv + torch.tensor(values, dtype=torch.float32).to(device)

        st_t  = torch.FloatTensor(np.array(states)).to(device)
        ac_t  = torch.LongTensor(np.array(actions)).to(device)
        lp_t  = torch.stack(log_probs).to(device)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(states)
        for _ in range(K_EPOCHS):
            idx = torch.randperm(n)
            for s in range(0, n, BATCH_SIZE):
                b = idx[s:s+BATCH_SIZE]
                if len(b) == 0: continue
                probs, vals_b = model(st_t[b])
                dist  = torch.distributions.Categorical(probs)
                nlps  = dist.log_prob(ac_t[b])
                ent   = dist.entropy().mean()
                ratio = torch.exp(nlps - lp_t[b])
                s1    = ratio * adv_n[b]
                s2    = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_n[b]
                pl    = -torch.min(s1, s2).mean()
                vl    = F.mse_loss(vals_b.squeeze(-1), returns[b])
                loss  = pl + VALUE_COEF * vl - ENTROPY_COEF * ent
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        rewards_hist.append(ep_reward)
        victories_hist.append(1 if info.get("victory", False) else 0)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({"model_state_dict": model.state_dict()},
                       os.path.join(MODELS_DIR, "ppo_target_model.pth"))

        win = min(100, len(victories_hist))
        rate = np.mean(victories_hist[-win:]) * 100.0

        if ep % PRINT_EVERY == 0 or ep == 1:
            t_min = (time.time() - start_time) / 60.0
            avg_r = np.mean(rewards_hist[-win:])
            print(
                f"Ep {ep:5d} | {t_min:4.1f}m | "
                f"Rew MA{win}: {avg_r:+8.1f} | "
                f"Victory: {rate:5.1f}% | "
                f"Hover: {info.get('hover_counter',0):2d}/{env.HOVER_STEPS} | "
                f"CP#{info.get('active_checkpoint',1)}",
                flush=True
            )

        if ep >= MIN_EP_START and rate >= TARGET_RATE:
            t_min = (time.time() - start_time) / 60.0
            print(f"\n🎯 THÀNH CÔNG! Victory Rate {rate:.1f}% tại Ep {ep} ({t_min:.1f} phút)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ Kết thúc sau {elapsed/60:.1f} phút | {ep} episodes")
    _plot(rewards_hist, victories_hist)


def _plot(rewards, victories):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("PPO — Multi-Step Station-Keeping 8m Track", fontweight="bold")

    eps = np.arange(1, len(rewards)+1)
    w   = 50

    ax1.plot(eps, rewards, alpha=0.2, color="#2196F3")
    if len(rewards) >= w:
        ax1.plot(range(w, len(rewards)+1),
                 np.convolve(rewards, np.ones(w)/w, "valid"),
                 color="#0D47A1", lw=2, label=f"MA-{w}")
    ax1.set_ylabel("Total Reward"); ax1.legend(); ax1.grid(alpha=0.3)

    v = np.array(victories, dtype=float)
    ax2.plot(eps, v * 100, alpha=0.2, color="#4CAF50")
    if len(v) >= w:
        ax2.plot(range(w, len(v)+1),
                 np.convolve(v, np.ones(w)/w, "valid") * 100,
                 color="#1B5E20", lw=2, label=f"Victory% MA-{w}")
    ax2.axhline(90, color="red", ls="--", lw=1.5, label="Mục tiêu 90%")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Victory Rate (%)"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "target_tracking_training.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"✓ Đã lưu biểu đồ: {p}")


if __name__ == "__main__":
    train()
