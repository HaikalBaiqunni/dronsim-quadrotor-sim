"""
drone_sim/core/research_controllers.py
=======================================
Modern control methods added in v2.0

  1. ADRC — Active Disturbance Rejection Control
       Linear extended state observers (ESO) estimate the "total disturbance"
       (wind, drag, unmodelled dynamics, actuator loss) on every axis and
       cancel it in the control law.

  2. GEO  — Geometric tracking control on SO(3)
       Attitude error is computed directly on the rotation group, so the
       controller has no Euler-angle singularity and works for large tilts.

References:
  [1] Han, J. (2009). From PID to Active Disturbance Rejection Control.
      IEEE Trans. Industrial Electronics 56(3).
  [2] Gao, Z. (2003). Scaling and bandwidth-parameterization based
      controller tuning. ACC 2003.
  [3] Lee, T., Leok, M., McClamroch, N. H. (2010). Geometric tracking
      control of a quadrotor UAV on SE(3). IEEE CDC.

Author : Haikal Hakim Baiqunni
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .controllers import BaseController
from .dynamics import DroneParams, rotation_matrix


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# ADRC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ADRCParams:
    """ADRC bandwidth parameterisation (Gao, 2003)."""
    wc_pos: float = 2.2       # position controller bandwidth   [rad/s]
    wo_pos: float = 9.0       # position observer bandwidth     [rad/s]
    wc_att: float = 14.0      # roll / pitch controller bandwidth
    wo_att: float = 40.0      # roll / pitch observer bandwidth
    wc_yaw: float = 3.0       # yaw controller bandwidth
    wo_yaw: float = 12.0      # yaw observer bandwidth
    max_tilt_deg: float = 35.0
    dist_limit: float = 12.0  # clamp on |disturbance estimate|  [m/s²]


class _ESO:
    """Third-order linear extended state observer for one axis.

    Plant:  ÿ = f + b0·u          (f = lumped "total disturbance")
    States: z1 → y,  z2 → ẏ,  z3 → f
    """

    SUBSTEPS = 4

    def __init__(self, wo: float, b0: float):
        self.b0 = b0
        self.set_bandwidth(wo)
        self.z = np.zeros(3)

    def set_bandwidth(self, wo: float):
        self.b1, self.b2, self.b3 = 3 * wo, 3 * wo ** 2, wo ** 3

    def init(self, y: float, ydot: float = 0.0):
        self.z[:] = (y, ydot, 0.0)

    def update(self, y: float, u: float, dt: float, f_limit: float):
        h = dt / self.SUBSTEPS
        z1, z2, z3 = self.z
        for _ in range(self.SUBSTEPS):
            e = y - z1
            z1 += h * (z2 + self.b1 * e)
            z2 += h * (z3 + self.b2 * e + self.b0 * u)
            z3 += h * (self.b3 * e)
            z3 = max(-f_limit, min(f_limit, z3))
        self.z[:] = (z1, z2, z3)


class ADRCController(BaseController):
    """Cascaded ADRC: position ESOs → desired tilt → attitude ESOs → torques."""

    def __init__(self, drone_params: DroneParams, params: ADRCParams = None):
        self.params = params or ADRCParams()
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "ADRC"

    @property
    def position_gains(self):
        return self.params.wc_pos ** 2, 2 * self.params.wc_pos

    def reset(self):
        p = self.params
        dp = self.dp
        self._pos_eso = [_ESO(p.wo_pos, 1.0) for _ in range(3)]
        self._att_eso = [_ESO(p.wo_att, 1.0 / dp.Ixx),
                         _ESO(p.wo_att, 1.0 / dp.Iyy),
                         _ESO(p.wo_yaw, 1.0 / dp.Izz)]
        self._a_cmd = np.zeros(3)
        self._tau_cmd = np.zeros(3)
        self._yaw_meas = 0.0
        self._initialized = False

    @property
    def disturbance_estimate(self) -> np.ndarray:
        """Estimated lumped disturbance acceleration [m/s²] (x, y, z)."""
        return np.array([e.z[2] for e in self._pos_eso])

    def compute(self, state: np.ndarray, setpoint: np.ndarray,
                dt: float) -> np.ndarray:
        p, dp = self.params, self.dp
        pos, vel = state[0:3], state[3:6]
        phi, theta, psi = state[6], state[7], state[8]
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0

        if not self._initialized:
            for i in range(3):
                self._pos_eso[i].init(pos[i], vel[i])
            self._att_eso[0].init(phi)
            self._att_eso[1].init(theta)
            self._att_eso[2].init(psi)
            self._yaw_meas = psi
            self._initialized = True

        # ── Position loop ───────────────────────────────────────────────────
        a_cmd = np.zeros(3)
        for i in range(3):
            eso = self._pos_eso[i]
            eso.update(pos[i], self._a_cmd[i], dt, p.dist_limit)
            u0 = p.wc_pos ** 2 * (setpoint[i] - eso.z[0]) - 2 * p.wc_pos * eso.z[1]
            a_cmd[i] = u0 - eso.z[2]

        g = dp.g
        a_h_max = g * math.tan(math.radians(p.max_tilt_deg))
        a_h = math.hypot(a_cmd[0], a_cmd[1])
        if a_h > a_h_max:
            a_cmd[0:2] *= a_h_max / a_h
        a_cmd[2] = max(-0.8 * g, min(a_cmd[2], 0.8 * g))
        self._a_cmd = a_cmd

        T, phi_d, theta_d = self.attitude_setpoint_from_position(
            a_cmd[0], a_cmd[1], a_cmd[2], psi_d, dp.mass, g)

        # ── Attitude loop (yaw angle is unwrapped for the observer) ─────────
        self._yaw_meas += _wrap(psi - self._yaw_meas)
        y_att = (phi, theta, self._yaw_meas)
        r_att = (phi_d, theta_d, self._yaw_meas + _wrap(psi_d - self._yaw_meas))
        wc = (p.wc_att, p.wc_att, p.wc_yaw)
        inertia = (dp.Ixx, dp.Iyy, dp.Izz)

        tau = np.zeros(3)
        for i in range(3):
            eso = self._att_eso[i]
            eso.update(y_att[i], self._tau_cmd[i], dt, 400.0)
            u0 = wc[i] ** 2 * (r_att[i] - eso.z[0]) - 2 * wc[i] * eso.z[1]
            tau[i] = inertia[i] * (u0 - eso.z[2])
        self._tau_cmd = tau

        return self.thrust_to_rotors(T, tau)

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        pos_err = state[0:3] - setpoint[0:3]
        att_err = state[6:9]
        return float(np.dot(pos_err, pos_err) + 0.1 * np.dot(att_err, att_err))


# ─────────────────────────────────────────────────────────────────────────────
# Geometric control on SO(3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoParams:
    """Geometric controller gains expressed as natural frequencies."""
    wn_pos: float = 2.2       # position natural frequency  [rad/s]
    zeta_pos: float = 0.9     # position damping ratio
    wn_att: float = 14.0      # attitude natural frequency  [rad/s]
    zeta_att: float = 1.0     # attitude damping ratio
    ki_pos: float = 0.0       # optional integral action    [1/s²]
    max_tilt_deg: float = 40.0


def _hat_inv(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class GeometricController(BaseController):
    """Lee-Leok-McClamroch geometric controller (position + attitude on SE(3))."""

    def __init__(self, drone_params: DroneParams, params: GeoParams = None):
        self.params = params or GeoParams()
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "Geometric SE(3)"

    @property
    def position_gains(self):
        return self.params.wn_pos ** 2, 2 * self.params.zeta_pos * self.params.wn_pos

    def reset(self):
        self._int = np.zeros(3)
        self.last_att_error = 0.0

    def compute(self, state: np.ndarray, setpoint: np.ndarray,
                dt: float) -> np.ndarray:
        p, dp = self.params, self.dp
        m, g = dp.mass, dp.g
        pos, vel = state[0:3], state[3:6]
        omega = state[9:12]
        R = rotation_matrix(state[6], state[7], state[8])
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0

        # ── Desired force vector ────────────────────────────────────────────
        e_x = pos - setpoint[0:3]
        self._int = np.clip(self._int + e_x * dt, -2.0, 2.0)
        kx, kv = p.wn_pos ** 2, 2 * p.zeta_pos * p.wn_pos
        a_des = -kx * e_x - kv * vel - p.ki_pos * self._int

        a_h_max = (g + max(a_des[2], -0.8 * g)) * math.tan(math.radians(p.max_tilt_deg))
        a_h = math.hypot(a_des[0], a_des[1])
        if a_h > a_h_max:
            a_des[0:2] *= a_h_max / a_h
        a_des[2] = max(-0.8 * g, a_des[2])

        F_des = m * (a_des + np.array([0.0, 0.0, g]))
        b3 = R[:, 2]
        T = float(np.dot(F_des, b3))

        # ── Desired attitude R_d ────────────────────────────────────────────
        b3d = F_des / max(np.linalg.norm(F_des), 1e-6)
        b1_ref = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])
        b2d = np.cross(b3d, b1_ref)
        n2 = np.linalg.norm(b2d)
        if n2 < 1e-6:                       # yaw reference parallel to thrust
            b2d = np.cross(b3d, np.array([1.0, 0.0, 0.0]))
            n2 = max(np.linalg.norm(b2d), 1e-6)
        b2d /= n2
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack((b1d, b2d, b3d))

        # ── Attitude error on SO(3) and torque ──────────────────────────────
        e_R = 0.5 * _hat_inv(Rd.T @ R - R.T @ Rd)
        e_W = omega
        self.last_att_error = float(np.linalg.norm(e_R))

        I = np.diag([dp.Ixx, dp.Iyy, dp.Izz])
        kR = np.diag(I) * p.wn_att ** 2
        kW = np.diag(I) * 2 * p.zeta_att * p.wn_att
        tau = -kR * e_R - kW * e_W + np.cross(omega, I @ omega)

        return self.thrust_to_rotors(max(T, 0.0), tau)

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        pos_err = state[0:3] - setpoint[0:3]
        att_err = state[6:9]
        return float(np.dot(pos_err, pos_err) + 0.1 * np.dot(att_err, att_err))
