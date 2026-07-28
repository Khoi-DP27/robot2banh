"""
===========================================================
  record_combined_video.py — Ghép 3 thuật toán (DQN, PPO, A2C)
  vào 1 Video So Sánh Khung Hình 3 Tầng (3-Panel Split Screen)
===========================================================
"""
import os, sys, math
import numpy as np
import cv2
import torch
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from envs.balance_bot_target_env import BalanceBotTargetEnv
from models.dqn_model import DQNetwork
from models.ppo import PPONetwork
from models.actor_critic import ActorCriticNetwork

MODELS_DIR = "saved_models"
VIDEO_DIR  = "saved_videos"
ARTIFACTS_DIR = r"C:\Users\phuck\.gemini\antigravity-ide\brain\61c51315-aace-4226-beac-fdb454903ab5"
NUM_ACTIONS = 11
os.makedirs(VIDEO_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_SINGLE, H_SINGLE = 1280, 260
W_FINAL, H_FINAL = 1280, 780   # 3 tầng x 260px


def world2px(x, start_px=80, scale=140.0):
    return int(start_px + x * scale)


def draw_panel(env, x_raw, th_raw, xd, force, step, time_sec, algo_name, badge_color):
    th_deg  = math.degrees(th_raw)
    in_zone = env.zone_min <= x_raw <= env.zone_max
    hov     = env.hover_counter

    img = np.full((H_SINGLE, W_SINGLE, 3), 20, dtype=np.uint8)
    GY  = int(H_SINGLE * 0.72)
    SPX, SCL = 80, 140.0

    # Ground
    cv2.line(img, (SPX, GY), (SPX + 1120, GY), (100, 100, 100), 2)
    for m in range(9):
        px = world2px(m)
        cv2.line(img, (px, GY), (px, GY+6), (130, 130, 130), 1)

    # Zone
    za, zb = world2px(env.zone_min), world2px(env.zone_max)
    zc_px  = world2px(env.zone_center)
    zcol   = (0, 230, 120) if in_zone else (0, 160, 255)
    ov = img.copy()
    cv2.rectangle(ov, (za, GY-140), (zb, GY), zcol, -1)
    cv2.addWeighted(ov, 0.22, img, 0.78, 0, img)
    cv2.rectangle(img, (za, GY-140), (zb, GY), zcol, 2)

    # Flag
    cv2.line(img, (zc_px, GY), (zc_px, GY-100), (255, 255, 255), 2)
    pts = np.array([[zc_px, GY-100], [zc_px+28, GY-86], [zc_px, GY-72]], np.int32)
    cv2.fillPoly(img, [pts], zcol)

    # Cart
    cx = world2px(x_raw)
    cy = GY - 20
    cw, ch, wr = 55, 22, 10
    for wx in [cx - cw//3, cx + cw//3]:
        cv2.circle(img, (wx, GY - wr), wr, (40, 40, 40), -1)
        cv2.circle(img, (wx, GY - wr), wr, (180, 180, 180), 2)
    cv2.rectangle(img, (cx-cw//2, cy-ch), (cx+cw//2, cy), (180, 100, 0), -1)
    cv2.rectangle(img, (cx-cw//2, cy-ch), (cx+cw//2, cy), (255, 255, 255), 2)

    # Pole
    pole_len = 100
    px2 = int(cx + pole_len * math.sin(th_raw))
    py2 = int((cy - ch) - pole_len * math.cos(th_raw))
    pc  = (0, 220, 80) if abs(th_deg) < 10 else (0, 160, 255) if abs(th_deg) < 20 else (60, 60, 220)
    cv2.line(img, (cx, cy-ch), (px2, py2), pc, 5)
    cv2.circle(img, (px2, py2), 7, (255, 255, 255), -1)

    # Force arrow
    if abs(force) > 0.5:
        arlen = int(abs(force) * 3.0)
        adir  = 1 if force > 0 else -1
        ac    = (0, 140, 255) if force > 0 else (255, 100, 0)
        cv2.arrowedLine(img, (cx, cy-ch//2), (cx+adir*arlen, cy-ch//2), ac, 2, tipLength=0.3)

    # Algo Badge (Góc trái)
    cv2.rectangle(img, (15, 12), (320, 48), badge_color, -1)
    cv2.putText(img, algo_name, (25, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Hover progress bar (Trung tâm trên)
    bar_x, bar_y, bar_w = 340, 18, 300
    pct = min(1.0, hov / env.HOVER_STEPS)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+16), (50, 50, 50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+int(bar_w*pct), bar_y+16),
                  (0, 230, 120) if in_zone else (100, 100, 100), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+16), (200, 200, 200), 1)
    cv2.putText(img, f"HOVER: {hov}/{env.HOVER_STEPS} ({pct*100:.0f}%)",
                (bar_x+bar_w+15, bar_y+13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Telemetry HUD (Góc phải)
    status_txt = "IN ZONE ✓" if in_zone else "NAVIGATING"
    status_col = (0, 230, 120) if in_zone else (0, 180, 255)
    cv2.putText(img, f"x={x_raw:.2f}m  θ={th_deg:+.1f}°  v={xd:+.2f}m/s  F={force:+.1f}N",
                (W_SINGLE-450, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
    cv2.putText(img, status_txt, (W_SINGLE-140, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 2)

    # Separator line bottom
    cv2.line(img, (0, H_SINGLE-1), (W_SINGLE, H_SINGLE-1), (80, 80, 80), 2)
    return img


def record_combined():
    print("=" * 65)
    print("  GHÉP VIDEO SO SÁNH 3 THUẬT TOÁN (DQN | PPO | A2C)")
    print("=" * 65)

    env_dqn = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    env_ppo = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    env_a2c = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)

    obs_dim = env_dqn.observation_space.shape[0]

    # Load DQN
    model_dqn = DQNetwork(state_size=obs_dim, action_size=NUM_ACTIONS).to(device)
    ckpt = torch.load(os.path.join(MODELS_DIR, "dqn_target_model.pth"), map_location=device, weights_only=False)
    model_dqn.load_state_dict(ckpt["model_state_dict"])
    model_dqn.eval()

    # Load PPO
    model_ppo = PPONetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)
    ckpt = torch.load(os.path.join(MODELS_DIR, "ppo_target_model.pth"), map_location=device, weights_only=False)
    model_ppo.load_state_dict(ckpt["model_state_dict"])
    model_ppo.eval()

    # Load A2C
    model_a2c = ActorCriticNetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)
    ckpt = torch.load(os.path.join(MODELS_DIR, "a2c_target_model.pth"), map_location=device, weights_only=False)
    model_a2c.load_state_dict(ckpt["model_state_dict"])
    model_a2c.eval()

    fps = 50
    all_combined_frames = []

    # Reset cả 3 env ở cùng xuất phát x=3.2m
    for env in [env_dqn, env_ppo, env_a2c]:
        env.set_episode(5000)
        env.reset()
        env.raw_state[0] = 3.2

    obs_dqn = env_dqn._obs()
    obs_ppo = env_ppo._obs()
    obs_a2c = env_a2c._obs()

    done_dqn = done_ppo = done_a2c = False
    hover_dqn = hover_ppo = hover_a2c = 0

    print("  Đang render video ghép 3 tầng HD 1280x780...")

    for step in range(1, 350):
        # --- 1. DQN step ---
        if not done_dqn:
            ot = torch.FloatTensor(obs_dqn).unsqueeze(0).to(device)
            with torch.no_grad():
                a_dqn = model_dqn(ot).argmax().item()
            f_dqn = env_dqn.force_values[a_dqn]
            obs_dqn, _, d, t, info_dqn = env_dqn.step(a_dqn)
            if info_dqn.get("in_zone"): hover_dqn += 1
            if d or t or hover_dqn >= 200: done_dqn = True
        else:
            f_dqn = 0.0

        # --- 2. PPO step ---
        if not done_ppo:
            ot = torch.FloatTensor(obs_ppo).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = model_ppo(ot)
                a_ppo = torch.argmax(probs, dim=-1).item()
            f_ppo = env_ppo.force_values[a_ppo]
            obs_ppo, _, d, t, info_ppo = env_ppo.step(a_ppo)
            if info_ppo.get("in_zone"): hover_ppo += 1
            if d or t or hover_ppo >= 200: done_ppo = True
        else:
            f_ppo = 0.0

        # --- 3. A2C step ---
        if not done_a2c:
            ot = torch.FloatTensor(obs_a2c).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = model_a2c(ot)
                a_a2c = torch.argmax(probs, dim=-1).item()
            f_a2c = env_a2c.force_values[a_a2c]
            obs_a2c, _, d, t, info_a2c = env_a2c.step(a_a2c)
            if info_a2c.get("in_zone"): hover_a2c += 1
            if d or t or hover_a2c >= 200: done_a2c = True
        else:
            f_a2c = 0.0

        # Draw 3 panels
        p1 = draw_panel(env_dqn, (obs_dqn[0]+1)*4, obs_dqn[2]*env_dqn.theta_max, obs_dqn[1]*4, f_dqn, step, step/fps, "1. DQN (100% Win - 0.2m)", (0, 120, 215))
        p2 = draw_panel(env_ppo, (obs_ppo[0]+1)*4, obs_ppo[2]*env_ppo.theta_max, obs_ppo[1]*4, f_ppo, step, step/fps, "2. PPO (90% Win - 2.5m)", (180, 0, 180))
        p3 = draw_panel(env_a2c, (obs_a2c[0]+1)*4, obs_a2c[2]*env_a2c.theta_max, obs_a2c[1]*4, f_a2c, step, step/fps, "3. A2C (90% Win - 11.0m)", (0, 150, 0))

        # Stack vertically
        combined_frame = np.vstack([p1, p2, p3])
        all_combined_frames.append(combined_frame)

        if done_dqn and done_ppo and done_a2c:
            for _ in range(fps * 2): # Freeze 2 seconds
                all_combined_frames.append(combined_frame)
            break

    print(f"  ✓ Tổng cộng: {len(all_combined_frames)} frames")

    mp4_path = os.path.join(VIDEO_DIR, "target_tracking_combined.mp4")
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_FINAL, H_FINAL))
    for f in all_combined_frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  ✓ MP4 ghép 3 thuật toán: {mp4_path}")

    gif_path = os.path.join(VIDEO_DIR, "target_tracking_combined.gif")
    pil = [Image.fromarray(f).resize((720, 438), Image.Resampling.LANCZOS)
           for f in all_combined_frames[::2]]
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=40, loop=0)
    print(f"  ✓ GIF ghép 3 thuật toán: {gif_path}")

    # Copy to artifacts directory
    artifact_gif = os.path.join(ARTIFACTS_DIR, "target_tracking_combined.gif")
    try:
        import shutil
        shutil.copy(gif_path, artifact_gif)
        print(f"  ✓ Copied artifact: {artifact_gif}")
    except Exception as e:
        print(f"  ⚠ Lỗi copy artifact: {e}")


if __name__ == "__main__":
    record_combined()
