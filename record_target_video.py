"""
===========================================================
  record_target_video.py — Export Videos cho 3 Thuật Toán
  (DQN, PPO, A2C)
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
W, H = 1280, 720


def world2px(x, start_px=80, scale=140.0):
    return int(start_px + x * scale)


def draw_frame(env, x_raw, th_raw, xd, force, step, time_sec, algo_name="DQN"):
    th_deg  = math.degrees(th_raw)
    in_zone = env.zone_min <= x_raw <= env.zone_max
    hov     = env.hover_counter

    img = np.full((H, W, 3), 22, dtype=np.uint8)
    GY  = int(H * 0.72)
    SPX, SCL = 80, 140.0

    # Ground
    cv2.line(img, (SPX, GY), (SPX + 1120, GY), (120, 120, 120), 3)
    for m in range(9):
        px = world2px(m)
        cv2.line(img, (px, GY), (px, GY+10), (150, 150, 150), 2)
        cv2.putText(img, f"{m}m", (px-12, GY+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Zone
    za, zb = world2px(env.zone_min), world2px(env.zone_max)
    zc_px  = world2px(env.zone_center)
    zcol   = (0, 230, 120) if in_zone else (0, 160, 255)
    ov = img.copy()
    cv2.rectangle(ov, (za, GY-200), (zb, GY), zcol, -1)
    cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
    cv2.rectangle(img, (za, GY-200), (zb, GY), zcol, 2)

    # Zone label
    cv2.putText(img, f"TARGET ZONE", (za+5, GY-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zcol, 2)
    cv2.putText(img, f"{env.zone_min:.1f}m — {env.zone_max:.1f}m",
                (za+5, GY-38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zcol, 1)

    # Flag
    cv2.line(img, (zc_px, GY), (zc_px, GY-140), (255, 255, 255), 3)
    pts = np.array([[zc_px, GY-140], [zc_px+36, GY-122], [zc_px, GY-104]], np.int32)
    cv2.fillPoly(img, [pts], zcol)
    cv2.putText(img, "GOAL", (zc_px-20, GY-150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zcol, 2)

    # Cart
    cx = world2px(x_raw)
    cy = GY - 28
    cw, ch, wr = 65, 28, 13
    for wx in [cx - cw//3, cx + cw//3]:
        cv2.circle(img, (wx, GY - wr), wr, (55, 55, 55), -1)
        cv2.circle(img, (wx, GY - wr), wr, (200, 200, 200), 2)
    cv2.rectangle(img, (cx-cw//2, cy-ch), (cx+cw//2, cy), (180, 100, 0), -1)
    cv2.rectangle(img, (cx-cw//2, cy-ch), (cx+cw//2, cy), (255, 255, 255), 2)

    # Pole
    pole_len = 140
    px2 = int(cx + pole_len * math.sin(th_raw))
    py2 = int((cy - ch) - pole_len * math.cos(th_raw))
    pc  = (0, 220, 80) if abs(th_deg) < 10 else (0, 160, 255) if abs(th_deg) < 20 else (60, 60, 220)
    cv2.line(img, (cx, cy-ch), (px2, py2), pc, 7)
    cv2.circle(img, (px2, py2), 9, (255, 255, 255), -1)

    # Force arrow
    if abs(force) > 0.5:
        arlen = int(abs(force) * 3.5)
        adir  = 1 if force > 0 else -1
        ac    = (0, 140, 255) if force > 0 else (255, 100, 0)
        cv2.arrowedLine(img, (cx, cy-ch//2), (cx+adir*arlen, cy-ch//2), ac, 3, tipLength=0.3)

    # Hover bar
    bar_x, bar_y, bar_w = SPX, GY+55, 400
    pct = min(1.0, hov / env.HOVER_STEPS)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (50, 50, 50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+int(bar_w*pct), bar_y+20),
                  (0, 230, 120) if in_zone else (100, 100, 100), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (200, 200, 200), 1)
    cv2.putText(img, f"HOVER: {hov}/{env.HOVER_STEPS} steps ({pct*100:.0f}%)",
                (bar_x, bar_y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Status badge
    if in_zone:
        badge_col = (0, 200, 80)
        badge_txt = "IN ZONE ✓"
    else:
        badge_col = (200, 80, 0)
        badge_txt = "NAVIGATING"
    cv2.rectangle(img, (W-220, GY+45), (W-30, GY+78), badge_col, -1)
    cv2.putText(img, badge_txt, (W-210, GY+69),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # HUD
    cv2.rectangle(img, (25, 12), (W-25, 88), (35, 35, 35), -1)
    cv2.rectangle(img, (25, 12), (W-25, 88), (90, 90, 90), 2)
    cv2.putText(img, f"{algo_name} ROBOT 2 BANH — STATION-KEEPING DEMO",
                (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
    cv2.putText(img,
                f"t={time_sec:.2f}s | x={x_raw:.2f}m | θ={th_deg:+.1f}° | v={xd:+.2f}m/s | InZone:{in_zone}",
                (40, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
    return img


def record_single_algo(algo_tag, model_file, algo_display_name, model_type):
    model_path = os.path.join(MODELS_DIR, model_file)
    print(f"\n▶ GHI HÌNH THUẬT TOÁN: {algo_display_name}")

    env = BalanceBotTargetEnv(num_actions=NUM_ACTIONS)
    obs_dim = env.observation_space.shape[0]

    if not os.path.exists(model_path):
        print(f"⚠ Không tìm thấy {model_path}, bỏ qua.")
        return

    if model_type == "dqn":
        model = DQNetwork(state_size=obs_dim, action_size=NUM_ACTIONS).to(device)
    elif model_type == "ppo":
        model = PPONetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)
    elif model_type == "a2c":
        model = ActorCriticNetwork(state_size=obs_dim, action_size=NUM_ACTIONS, hidden_size=256).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fps = 50
    all_frames = []

    # Episode test
    env.set_episode(5000)
    obs, _ = env.reset()
    env.raw_state[0] = 3.2
    obs = env._obs()

    ep_frames = []
    hover_steps = 0

    for step in range(1, 800):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if model_type == "dqn":
                q_vals = model(obs_t)
                action = q_vals.argmax().item()
            else:
                probs, _ = model(obs_t)
                action = torch.argmax(probs, dim=-1).item()

        x_raw  = (obs[0] + 1.0) * 4.0
        th_raw = obs[2] * env.theta_max
        force  = env.force_values[action]

        frame = draw_frame(env, x_raw, th_raw, obs[1]*4.0, force, len(ep_frames)+1, (len(ep_frames)+1)/fps, algo_display_name)
        ep_frames.append(frame)

        obs, _, done, trunc, info = env.step(action)
        if info.get("in_zone", False):
            hover_steps += 1

        if done or trunc or hover_steps >= 200:
            for _ in range(fps * 2): # Freeze 2 seconds
                ep_frames.append(frame)
            break

    all_frames.extend(ep_frames)

    mp4_filename = f"target_tracking_{algo_tag}.mp4"
    gif_filename = f"target_tracking_{algo_tag}.gif"

    mp4_path = os.path.join(VIDEO_DIR, mp4_filename)
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in all_frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  ✓ MP4: {mp4_path}")

    gif_path = os.path.join(VIDEO_DIR, gif_filename)
    pil = [Image.fromarray(f).resize((640, 360), Image.Resampling.LANCZOS)
           for f in all_frames[::2]]
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=40, loop=0)
    print(f"  ✓ GIF: {gif_path}")

    # Copy to artifacts directory
    artifact_gif = os.path.join(ARTIFACTS_DIR, gif_filename)
    try:
        import shutil
        shutil.copy(gif_path, artifact_gif)
        print(f"  ✓ Copied artifact: {artifact_gif}")
    except Exception as e:
        print(f"  ⚠ Lỗi copy artifact: {e}")


def record_all():
    print("=" * 60)
    print("  GHI HÌNH ĐỦ 3 THUẬT TOÁN (DQN, PPO, A2C)")
    print("=" * 60)

    algos = [
        ("dqn", "dqn_target_model.pth", "DQN (100% Win)", "dqn"),
        ("ppo", "ppo_target_model.pth", "PPO (90% Win)", "ppo"),
        ("a2c", "a2c_target_model.pth", "A2C (90% Win)", "a2c"),
    ]

    for tag, mfile, name, mtype in algos:
        record_single_algo(tag, mfile, name, mtype)

    print("=" * 60)
    print("✓ ĐÃ HOÀN THÀNH GHI HÌNH TẤT CẢ 3 THUẬT TOÁN!")


if __name__ == "__main__":
    record_all()
