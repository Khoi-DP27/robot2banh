"""
===========================================================
  envs/balance_bot_target_env.py — Single Zone Environment
===========================================================
  Zone: [5.25m, 6.75m] — 1.5m rộng, tâm 6.0m
  Hover: 20 steps (0.4s)
  Theta: 35°
  Reward: Gaussian potential field kéo về tâm zone
  Spawn: 50% trong zone (hoc hover), 50% ngoai zone (hoc navigation)
===========================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class BalanceBotTargetEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, num_actions=11):
        super().__init__()

        # Physics
        self.gravity          = 9.8
        self.mass_cart        = 1.0
        self.mass_pole        = 0.1
        self.total_mass       = self.mass_cart + self.mass_pole
        self.length           = 0.5
        self.pole_mass_length = self.mass_pole * self.length
        self.friction         = 0.005
        self.tau              = 0.02
        self.max_steps        = 800

        # Track
        self.track_min = 0.0
        self.track_max = 8.0

        # Zone duy nhất: 1.5m rộng, tâm 6.0m
        self.zone_min    = 5.25
        self.zone_max    = 6.75
        self.zone_center = 6.0

        # Hover: 20 steps (0.4s)
        self.HOVER_STEPS = 20

        # Ngưỡng ngã: 35°
        self.theta_max = math.radians(35.0)

        # 11 actions
        self.num_actions  = num_actions
        self.force_values = np.linspace(-20.0, 20.0, num_actions)
        self.action_space = spaces.Discrete(num_actions)

        # Obs: [x_norm, xdot_norm, theta_norm, thetadot_norm, dist_signed_norm, hover_norm]
        self.observation_space = spaces.Box(
            low= np.array([-1, -1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([ 1,  1,  1,  1,  1, 1], dtype=np.float32),
        )

        self.raw_state       = None
        self.hover_counter   = 0
        self.steps_done      = 0
        self.current_episode = 0

    def set_episode(self, ep):
        self.current_episode = ep

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 50% spawn trong zone (hoc hover), 50% spawn tu [3.0 - 5.0m] (hoc navigate)
        if self.np_random.random() < 0.5:
            x_init = float(self.np_random.uniform(5.3, 6.7))
        else:
            x_init = float(self.np_random.uniform(3.0, 5.0))

        theta_init = float(self.np_random.uniform(-0.005, 0.005))
        self.raw_state     = np.array([x_init, 0.0, theta_init, 0.0], dtype=np.float32)
        self.hover_counter = 0
        self.steps_done    = 0
        return self._obs(), {}

    def _obs(self):
        x, xd, th, thd = self.raw_state
        return np.array([
            (x / 4.0) - 1.0,
            np.clip(xd / 4.0, -1, 1),
            np.clip(th / self.theta_max, -1, 1),
            np.clip(thd / 6.0, -1, 1),
            np.clip((x - self.zone_center) / 4.0, -1, 1),   # signed dist
            self.hover_counter / self.HOVER_STEPS,
        ], dtype=np.float32)

    def step(self, action):
        x, xd, th, thd = self.raw_state
        F      = self.force_values[action]
        cos_th = math.cos(th)
        sin_th = math.sin(th)
        fric   = self.friction * (1 if xd > 0 else -1 if xd < 0 else 0)

        tmp    = (F + self.pole_mass_length * thd**2 * sin_th - fric) / self.total_mass
        th_acc = (self.gravity * sin_th - cos_th * tmp) / \
                 (self.length * (4/3 - self.mass_pole * cos_th**2 / self.total_mass))
        x_acc  = tmp - self.pole_mass_length * th_acc * cos_th / self.total_mass

        x   += self.tau * xd
        xd  += self.tau * x_acc
        th  += self.tau * thd
        thd += self.tau * th_acc

        self.raw_state = np.array([x, xd, th, thd], dtype=np.float32)
        self.steps_done += 1

        fallen  = abs(th) > self.theta_max
        oob     = (x < self.track_min) or (x > self.track_max)
        failed  = fallen or oob
        in_zone = (self.zone_min <= x <= self.zone_max) and not failed

        self.hover_counter = (self.hover_counter + 1) if in_zone else 0
        victory = self.hover_counter >= self.HOVER_STEPS
        done    = failed or victory
        trunc   = self.steps_done >= self.max_steps

        # Reward
        dist = abs(x - self.zone_center)
        if victory:
            r = 1000.0
        elif failed:
            r = -20.0
        else:
            potential  = 25.0 * math.exp(-1.2 * dist)
            zone_bonus = 20.0 if in_zone else 0.0
            tilt_pen   = -5.0 * th**2
            r = 0.5 + potential + zone_bonus + tilt_pen

        info = {
            "x": x,
            "hover_counter": self.hover_counter,
            "victory": victory,
            "failed": failed,
            "in_zone": in_zone,
            "zone_min": self.zone_min,
            "zone_max": self.zone_max,
            "target_center": self.zone_center,
            "active_checkpoint": 1,
            "checkpoint_completed": False,
        }
        return self._obs(), r, done, trunc, info
