"""
drone_sim/core/estimation.py
=============================
Sensor simulation + state estimation (sensor fusion)

Instead of adding white noise straight onto the true state, this module
simulates the sensors a real flight controller has and fuses them:

  IMU   gyro + accelerometer @ 100 Hz  (white noise + slowly drifting bias)
  GPS   position + velocity   @ 10 Hz  (white noise)
  Baro  altitude              @ 50 Hz  (white noise)
  AHRS  attitude reference    @ 100 Hz (noisy roll/pitch/yaw)

Estimator
  * Position / velocity : per-axis linear Kalman filter. The accelerometer
    (rotated with the *estimated* attitude) drives the prediction; GPS and
    baro correct it.
  * Attitude            : complementary filter. Gyro integration is the
    high-pass path, the AHRS attitude reference the low-pass path.

Note: this is a linear Kalman filter on position/velocity plus a
complementary attitude filter, not a full 15-state EKF.

Author : Haikal Hakim Baiqunni
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dynamics import DroneParams, euler_rate_matrix, rotation_matrix


@dataclass
class EstimatorParams:
    gps_pos_std: float = 0.25       # m
    gps_vel_std: float = 0.10       # m/s
    gps_rate: float = 10.0          # Hz
    baro_std: float = 0.10          # m
    baro_rate: float = 50.0         # Hz
    accel_std: float = 0.15         # m/s²
    accel_bias_walk: float = 0.002  # m/s² per √s
    gyro_std: float = 0.01          # rad/s
    gyro_bias_walk: float = 0.0005  # rad/s per √s
    att_ref_std: float = 0.02       # rad
    att_cutoff: float = 2.0         # rad/s complementary-filter crossover


class StateEstimator:
    """Simulates the sensor suite and returns an estimated 12-state vector."""

    def __init__(self, dp: DroneParams, params: EstimatorParams = None,
                 dt: float = 0.01, seed: Optional[int] = None):
        self.dp = dp
        self.p = params or EstimatorParams()
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, x0: Optional[np.ndarray] = None):
        x0 = np.zeros(12) if x0 is None else np.asarray(x0, dtype=float)
        # per-axis [pos, vel] estimate and covariance
        self.pos = x0[0:3].copy()
        self.vel = x0[3:6].copy()
        self.P = np.array([np.diag([1.0, 1.0]) for _ in range(3)])
        self.euler = x0[6:9].copy()
        self.omega = x0[9:12].copy()
        self._gyro_bias = np.zeros(3)
        self._acc_bias = np.zeros(3)
        self._t = 0.0
        self._t_gps = 0.0
        self._t_baro = 0.0
        self.last_error = np.zeros(3)

    # ------------------------------------------------------------------
    def step(self, true_state: np.ndarray, true_accel: np.ndarray,
             dt: Optional[float] = None) -> np.ndarray:
        """
        Args:
            true_state: ground-truth 12-vector at the current step
            true_accel: true inertial acceleration [m/s²] (incl. wind), used
                        only to synthesise the accelerometer reading
        Returns:
            estimated 12-vector handed to the controller
        """
        p, rng = self.p, self._rng
        dt = dt or self.dt
        self._t += dt
        g = np.array([0.0, 0.0, self.dp.g])

        # ── Sensor synthesis ───────────────────────────────────────────────
        self._gyro_bias += rng.standard_normal(3) * p.gyro_bias_walk * np.sqrt(dt)
        self._acc_bias += rng.standard_normal(3) * p.accel_bias_walk * np.sqrt(dt)
        gyro = true_state[9:12] + self._gyro_bias + rng.standard_normal(3) * p.gyro_std

        R_true = rotation_matrix(*true_state[6:9])
        f_body = R_true.T @ (true_accel + g)            # specific force
        accel_b = f_body + self._acc_bias + rng.standard_normal(3) * p.accel_std

        att_ref = true_state[6:9] + rng.standard_normal(3) * p.att_ref_std

        # ── Attitude: complementary filter ─────────────────────────────────
        self.omega = gyro
        eul_dot = euler_rate_matrix(self.euler[0], self.euler[1]) @ gyro
        err = att_ref - self.euler
        err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
        self.euler = self.euler + dt * (eul_dot + p.att_cutoff * err)
        self.euler[2] = (self.euler[2] + np.pi) % (2 * np.pi) - np.pi

        # ── Position / velocity: Kalman filter ─────────────────────────────
        R_hat = rotation_matrix(*self.euler)
        a_world = R_hat @ accel_b - g
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([0.5 * dt * dt, dt])
        q_a = p.accel_std ** 2 + 0.05           # model mismatch inflation
        Q = q_a * np.outer(G, G)

        do_gps = (self._t - self._t_gps) >= 1.0 / p.gps_rate - 1e-9
        do_baro = (self._t - self._t_baro) >= 1.0 / p.baro_rate - 1e-9
        if do_gps:
            self._t_gps = self._t
        if do_baro:
            self._t_baro = self._t

        for i in range(3):
            x = np.array([self.pos[i], self.vel[i]])
            x = F @ x + G * a_world[i]
            P = F @ self.P[i] @ F.T + Q

            meas = []
            if do_gps:
                meas.append((np.array([1.0, 0.0]), true_state[i] + rng.standard_normal() * p.gps_pos_std, p.gps_pos_std ** 2))
                meas.append((np.array([0.0, 1.0]), true_state[3 + i] + rng.standard_normal() * p.gps_vel_std, p.gps_vel_std ** 2))
            if do_baro and i == 2:
                meas.append((np.array([1.0, 0.0]), true_state[2] + rng.standard_normal() * p.baro_std, p.baro_std ** 2))
            for H, z, r in meas:
                S = H @ P @ H + r
                K = (P @ H) / S
                x = x + K * (z - H @ x)
                P = (np.eye(2) - np.outer(K, H)) @ P
            self.pos[i], self.vel[i] = x
            self.P[i] = P

        est = np.concatenate([self.pos, self.vel, self.euler, self.omega])
        self.last_error = est[0:3] - true_state[0:3]
        return est
