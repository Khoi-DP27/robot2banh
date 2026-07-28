"""
===========================================================
  run_all.py -- Chay va so sanh 3 thuat toan RL
  PPO | DQN | A2C tren bai toan Station-Keeping
===========================================================
"""
import subprocess, sys, time, os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


ALGOS = [
    ("PPO", "train_target_tracking.py"),
    ("DQN", "train_dqn.py"),
    ("A2C", "train_a2c.py"),
]

def run_all():
    print("=" * 65)
    print("  SO SÁNH 3 THUẬT TOÁN RL: PPO | DQN | A2C")
    print("  Bài toán: Single Zone Station-Keeping (8m Track)")
    print("=" * 65)
    results = {}

    for name, script in ALGOS:
        print(f"\n{'='*65}")
        print(f"  ▶ Bắt đầu train {name}...")
        print(f"{'='*65}")
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-u", script], capture_output=False)
        elapsed = time.time() - t0
        results[name] = {"time": elapsed, "ok": ret.returncode == 0}
        print(f"\n  ✓ {name} hoàn tất sau {elapsed/60:.1f} phút")

    print("\n" + "="*65)
    print("  KẾT QUẢ SO SÁNH:")
    print("="*65)
    for name, r in results.items():
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} {name}: {r['time']/60:.1f} phút")
    print("="*65)
    print("\n  Xem biểu đồ tại: saved_plots/")
    print("  Xem model tại:   saved_models/")


if __name__ == "__main__":
    run_all()
