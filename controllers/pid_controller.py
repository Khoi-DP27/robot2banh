"""
===========================================================
  PIDController — Bộ điều khiển Dual-Loop Cascaded PID
===========================================================
"""

import numpy as np


class PIDController:
    """
    Bộ điều khiển Dual-Loop Cascaded PID giữ xe vừa đứng thẳng vừa nằm ở trung tâm (x=0).
    """

    def __init__(self, Kp=30.0, Ki=2.0, Kd=15.0, Kpx=0.05, Kdx=0.1, dt=0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.Kpx = Kpx
        self.Kdx = Kdx

        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = 5.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def get_action(self, target_angle, current_angle, force_values, current_x=0.0, current_x_dot=0.0):
        # 1. Outer Loop: Vị trí x -> theta_ref (nếu lệch x > 0 thì cần tạo độ nghiêng hồi đáp)
        theta_ref = (self.Kpx * current_x + self.Kdx * current_x_dot)
        theta_ref = np.clip(theta_ref, -0.10, 0.10)

        # 2. Inner Loop: Góc theta -> Lực u(t)
        error = target_angle + theta_ref - current_angle

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u_t = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        action_idx = np.argmin(np.abs(force_values - u_t))
        raw_force = u_t

        return action_idx, raw_force
