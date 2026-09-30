"""
drone_sim/core/controllers.py
==============================
Flight Control Algorithms for Quadrotor UAV

Implements a cascaded control architecture common in industrial UAV systems:
    Outer loop: Position controller → desired attitude setpoints
    Inner loop: Attitude controller → motor commands

Supported controllers:
    1. PID (Proportional-Integral-Derivative)
    2. SMC (Sliding Mode Control) with boundary layer
    3. Cascaded PID + SMC hybrid

References:
    [1] Ziegler, J. G., & Nichols, N. B. (1942). Optimum settings for
        automatic controllers. Transactions of the ASME, 64(11), 759-765.
    [2] Utkin, V. I. (1992). Sliding Modes in Control and Optimization.
        Springer-Verlag.
    [3] Slotine, J. J. E., & Li, W. (1991). Applied Nonlinear Control.
        Prentice Hall.
    [4] Luukkonen, T. (2011). Modelling and control of quadcopter.
        Aalto University School of Science.
    [5] Argentim, L. M., et al. (2013). PID, LQR and LQR-PID on a
        quadrotor platform. ICIEV.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .dynamics import DroneParams, DroneState, euler_rate_matrix


# ---------------------------------------------------------------------------
# Data classes for controller parameters
# ---------------------------------------------------------------------------

@dataclass
class PIDGains:
    """PID gains for a single axis."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    i_limit: float = 10.0    # anti-windup integral clamp

    def to_dict(self) -> Dict[str, float]:
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}


@dataclass
class PIDParams:
    """Full PID parameter set for quadrotor cascaded control."""
    # Position loop (outer)
    pos_x:  PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.0, kd=0.8))
    pos_y:  PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.0, kd=0.8))
    pos_z:  PIDGains = field(default_factory=lambda: PIDGains(kp=2.0, ki=0.5, kd=1.0))

    # Attitude loop (inner)
    roll:   PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.2, kd=0.8))
    pitch:  PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.2, kd=0.8))
    yaw:    PIDGains = field(default_factory=lambda: PIDGains(kp=3.0, ki=0.1, kd=0.3))


@dataclass
class SMCParams:
    """Sliding Mode Control parameters."""
    # Sliding surface slopes (λ > 0)
    lambda_pos:  float = 1.5    # position sliding surface
    lambda_att:  float = 3.0    # attitude sliding surface

    # Reaching law gains (η > 0)
    eta_pos:    float = 2.0
    eta_att:    float = 4.0

    # Boundary layer thickness (prevents chattering)
    phi_pos:    float = 0.5     # m
    phi_att:    float = 0.3     # rad/s

    # Disturbance bound estimate
    Dmax_pos:   float = 1.0     # N/kg
    Dmax_att:   float = 0.5     # N·m/kg·m²


# ---------------------------------------------------------------------------
# Controller base class
# ---------------------------------------------------------------------------

class BaseController(ABC):
    """Abstract base class for all drone controllers."""

    def __init__(self, drone_params: DroneParams):
        self.dp = drone_params
        self.dt = 0.01
        self.reset()

    @abstractmethod
    def compute(
        self,
        state: np.ndarray,
        setpoint: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute rotor speed commands.

        Args:
            state: current 12-DOF state vector
            setpoint: [x_d, y_d, z_d, ψ_d] desired position + yaw
            dt: time step (s)

        Returns:
            rotor_speeds: [Ω1..4] rad/s
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset controller internal state (integrators, etc.)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def position_gains(self):
        """Effective (stiffness, damping) of the horizontal position loop.

        The CBF safety filter turns an acceleration correction into a setpoint
        shift of correction / stiffness, so it needs to know how stiff the
        controller really is. Controllers override this; 1.5 / 2.2 is a
        middle-of-the-road default.
        """
        return 1.5, 2.2

    def thrust_to_rotors(self, T: float, tau: np.ndarray) -> np.ndarray:
        """Convert total thrust + torques → individual rotor speeds.

        Inverts the wrench allocation matrix B:
            [T, τ_φ, τ_θ, τ_ψ] = B · [Ω1², Ω2², Ω3², Ω4²]

        B is defined as:
            T     = kT*(Ω1²+Ω2²+Ω3²+Ω4²)
            τ_φ   = kT*l*(-Ω2²+Ω4²)
            τ_θ   = kT*l*(Ω1²-Ω3²)
            τ_ψ   = kD*(-Ω1²+Ω2²-Ω3²+Ω4²)

        Solves: Ω²_vec = B⁻¹ · [T, τ_φ, τ_θ, τ_ψ]
        """
        p = self.dp
        l = p.arm_length
        kT = p.kT
        kD = p.kD

        # Allocation matrix B maps [Ω1², Ω2², Ω3², Ω4²] → [T, τφ, τθ, τψ]
        # X-frame layout (rotors at 45°, numbered 1=FR, 2=FL, 3=RL, 4=RR)
        # Spin directions: 1=CW(-), 2=CCW(+), 3=CW(-), 4=CCW(+)
        B = np.array([
            [ kT,    kT,    kT,    kT  ],   # Thrust
            [ 0.0,  -kT*l,  0.0,   kT*l],  # Roll  τ_φ
            [ kT*l,  0.0,  -kT*l,  0.0 ],  # Pitch τ_θ
            [-kD,    kD,   -kD,    kD  ],   # Yaw   τ_ψ
        ])

        # Desired wrench vector, clip thrust to physical range
        T_max = 4.0 * kT * p.omega_max**2
        wrench = np.array([
            np.clip(T, 0.0, T_max),
            tau[0], tau[1], tau[2]
        ])

        # Solve for Ω²_i = B⁻¹ · wrench
        B_inv = np.linalg.pinv(B)
        Omega2 = B_inv @ wrench

        # Clamp and convert to Ω_i
        Omega2 = np.clip(Omega2, 0.0, p.omega_max**2)
        Omega_i = np.sqrt(Omega2)
        Omega_i = np.clip(Omega_i, p.omega_min, p.omega_max)

        return Omega_i

    def attitude_setpoint_from_position(
        self, ax: float, ay: float, az: float,
        yaw_d: float, mass: float, g: float
    ):
        """
        Compute desired roll/pitch from desired acceleration (position loop output).

        From the equations of motion:
            ax = (T/m)(cos(ψ)sin(θ)cos(φ) + sin(ψ)sin(φ))
            ay = (T/m)(sin(ψ)sin(θ)cos(φ) - cos(ψ)sin(φ))
            az = (T/m)(cos(θ)cos(φ)) - g

        Simplified (small angle + decoupled):
            T = m * sqrt(ax² + ay² + (az+g)²)
            φ_d = arcsin((ax*sin(ψ) - ay*cos(ψ)) * m/T)
            θ_d = arctan((ax*cos(ψ) + ay*sin(ψ)) / (az+g))
        """
        T = mass * np.sqrt(ax**2 + ay**2 + (az + g)**2)
        T = max(T, 1e-6)
        phi_d   = np.arcsin(np.clip((ax*np.sin(yaw_d) - ay*np.cos(yaw_d)) * mass/T, -1, 1))
        theta_d = np.arctan2(ax*np.cos(yaw_d) + ay*np.sin(yaw_d), az + g)
        return T, phi_d, theta_d


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------

class PIDController(BaseController):
    """
    Cascaded PID controller for quadrotor position + attitude.

    Architecture:
        [x,y,z setpoint] → Position PID → [ax,ay,az desired] →
        [φ,θ,ψ setpoint] → Attitude PID → [T, τφ, τθ, τψ] →
        Motor mixer → [Ω1..4]

    Anti-windup: Conditional integration (integrator clamp).
    Derivative: Applied to state (not error) to avoid derivative kick.
    """

    def __init__(self, drone_params: DroneParams, pid_params: PIDParams = None):
        self.params = pid_params or PIDParams()
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "PID"

    @property
    def position_gains(self):
        return self.params.pos_x.kp, self.params.pos_x.kd

    def reset(self):
        self._int_x = self._int_y = self._int_z = 0.0
        self._int_phi = self._int_theta = self._int_psi = 0.0
        self._prev_x = self._prev_y = self._prev_z = 0.0
        self._prev_phi = self._prev_theta = self._prev_psi = 0.0
        self._prev_vx = self._prev_vy = self._prev_vz = 0.0
        self._initialized = False

    def _pid_step(
        self, error: float, error_dot: float, integral: float,
        gains: PIDGains, dt: float
    ) -> tuple:
        """Single PID axis computation with anti-windup."""
        integral += error * dt
        integral = np.clip(integral, -gains.i_limit, gains.i_limit)
        output = gains.kp * error + gains.ki * integral + gains.kd * error_dot
        return output, integral

    def compute(
        self, state: np.ndarray, setpoint: np.ndarray, dt: float
    ) -> np.ndarray:
        s = DroneState(state)
        x_d, y_d, z_d = setpoint[0], setpoint[1], setpoint[2]
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0

        if not self._initialized:
            self._prev_x, self._prev_y, self._prev_z = s.x, s.y, s.z
            self._prev_phi, self._prev_theta, self._prev_psi = s.phi, s.theta, s.psi
            self._initialized = True

        g = self.dp.g
        m = self.dp.mass
        p = self.params

        # ── Position loop ──────────────────────────────────────────────────
        ex = x_d - s.x
        ey = y_d - s.y
        ez = z_d - s.z

        ex_dot = (ex - (x_d - self._prev_x)) / dt if dt > 1e-9 else 0.0
        ey_dot = (ey - (y_d - self._prev_y)) / dt if dt > 1e-9 else 0.0
        ez_dot = -s.vz   # use velocity measurement

        ax, self._int_x = self._pid_step(ex, -s.vx,  self._int_x,  p.pos_x, dt)
        ay, self._int_y = self._pid_step(ey, -s.vy,  self._int_y,  p.pos_y, dt)
        az, self._int_z = self._pid_step(ez,  ez_dot, self._int_z,  p.pos_z, dt)

        # Desired attitude from desired acceleration
        T, phi_d, theta_d = self.attitude_setpoint_from_position(ax, ay, az, psi_d, m, g)

        # ── Attitude loop ──────────────────────────────────────────────────
        e_phi   = phi_d   - s.phi
        e_theta = theta_d - s.theta
        e_psi   = psi_d   - s.psi
        e_psi   = (e_psi + np.pi) % (2*np.pi) - np.pi   # wrap ±π

        tau_phi,   self._int_phi   = self._pid_step(e_phi,   -s.p, self._int_phi,   p.roll,  dt)
        tau_theta, self._int_theta = self._pid_step(e_theta, -s.q, self._int_theta, p.pitch, dt)
        tau_psi,   self._int_psi   = self._pid_step(e_psi,   -s.r, self._int_psi,   p.yaw,   dt)

        self._prev_x, self._prev_y, self._prev_z = s.x, s.y, s.z

        return self.thrust_to_rotors(T, np.array([tau_phi, tau_theta, tau_psi]))

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        """Integral of squared tracking error (LQR-style cost)."""
        pos_err = state[0:3] - setpoint[0:3]
        att_err = state[6:9]
        return float(np.dot(pos_err, pos_err) + 0.1*np.dot(att_err, att_err))


# ---------------------------------------------------------------------------
# Sliding Mode Controller
# ---------------------------------------------------------------------------

class SMController(BaseController):
    """
    Sliding Mode Controller (SMC) for quadrotor.

    Uses a cascaded structure with sliding surfaces defined on:
        Position: s_pos = ė + λ_pos·e  (first-order surface)
        Attitude:  s_att = ė + λ_att·e

    Reaching law (boundary layer - continuous approximation):
        u_sw = η · sat(s/φ)

    The boundary layer replaces the discontinuous sign(s) function,
    eliminating chattering while maintaining robustness to disturbances.

    Stability: Lyapunov function V = (1/2)s²
        V̇ = s·ṡ = s·(-λs + u_eq + u_sw - d)
        Choosing |u_sw| > |d|: V̇ ≤ -η|s| < 0  (globally attractive)
    """

    def __init__(self, drone_params: DroneParams, smc_params: SMCParams = None):
        self.params = smc_params or SMCParams()
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "SMC"

    def reset(self):
        self._prev_e_pos  = np.zeros(3)
        self._prev_e_att  = np.zeros(3)
        self._prev_pos_d  = np.zeros(3)
        self._initialized = False

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        """Saturation function (boundary layer)."""
        return np.clip(x / max(phi, 1e-6), -1.0, 1.0)

    def compute(
        self, state: np.ndarray, setpoint: np.ndarray, dt: float
    ) -> np.ndarray:
        s = DroneState(state)
        pos_d = np.array([setpoint[0], setpoint[1], setpoint[2]])
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0

        p = self.params
        m = self.dp.mass
        g = self.dp.g

        if not self._initialized:
            self._prev_e_pos = pos_d - s.pos
            self._prev_e_att = np.zeros(3)
            self._prev_pos_d = pos_d
            self._initialized = True

        # ── Position sliding surface ───────────────────────────────────────
        e_pos = pos_d - s.pos
        # Error derivative from the measured velocity (finite-differencing the
        # noisy position measurement amplifies sensor noise by 1/dt).
        v_d = (pos_d - self._prev_pos_d) / max(dt, 1e-9)
        e_pos_dot = v_d - s.vel

        sigma_pos = e_pos_dot + p.lambda_pos * e_pos  # sliding variable

        # Equivalent control (nominal dynamics)
        u_eq_pos = p.lambda_pos * e_pos_dot  # tracks desired trajectory

        # Switching control (robustness term)
        u_sw_pos = np.array([
            (p.eta_pos + p.Dmax_pos) * self._sat(sigma_pos[i], p.phi_pos)
            for i in range(3)
        ])

        accel_cmd = u_eq_pos + u_sw_pos  # desired acceleration
        ax, ay, az = accel_cmd

        # Map to thrust + desired attitude
        T, phi_d, theta_d = self.attitude_setpoint_from_position(ax, ay, az, psi_d, m, g)

        # ── Attitude sliding surface ───────────────────────────────────────
        att_d = np.array([phi_d, theta_d, psi_d])
        e_att = att_d - s.euler
        e_att[2] = (e_att[2] + np.pi) % (2*np.pi) - np.pi  # yaw wrap

        # Measured Euler rates (gyro) instead of a finite difference.
        e_att_dot = -(euler_rate_matrix(s.phi, s.theta) @ s.omega_body)
        sigma_att = e_att_dot + p.lambda_att * e_att

        u_eq_att = p.lambda_att * e_att_dot
        u_sw_att = np.array([
            (p.eta_att + p.Dmax_att) * self._sat(sigma_att[i], p.phi_att)
            for i in range(3)
        ])

        tau_cmd = u_eq_att + u_sw_att  # [τφ, τθ, τψ]

        # Scale torques by inertia
        I = np.array([self.dp.Ixx, self.dp.Iyy, self.dp.Izz])
        tau = I * tau_cmd

        self._prev_e_pos = e_pos
        self._prev_e_att = e_att
        self._prev_pos_d = pos_d

        return self.thrust_to_rotors(T, tau)

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        """Sliding surface magnitude as cost metric."""
        e_pos = np.array(setpoint[0:3]) - state[0:3]
        e_vel = -state[3:6]
        sigma = e_vel + self.params.lambda_pos * e_pos
        return float(np.dot(sigma, sigma))


# ---------------------------------------------------------------------------
# Hybrid: PID position + SMC attitude
# ---------------------------------------------------------------------------

class HybridPIDSMC(BaseController):
    """
    Hybrid controller: PID outer-loop (position) + SMC inner-loop (attitude).

    Combines the simplicity of PID for slow position dynamics with the
    robustness of SMC for fast attitude dynamics that are more susceptible
    to disturbances and model uncertainty.
    """

    def __init__(
        self, drone_params: DroneParams,
        pid_params: PIDParams = None,
        smc_params: SMCParams = None
    ):
        self._pid = PIDController(drone_params, pid_params)
        self._smc = SMController(drone_params, smc_params)
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "PID+SMC Hybrid"

    def reset(self):
        if hasattr(self, '_pid'):
            self._pid.reset()
            self._smc.reset()

    def compute(
        self, state: np.ndarray, setpoint: np.ndarray, dt: float
    ) -> np.ndarray:
        s = DroneState(state)
        m = self.dp.mass
        g = self.dp.g
        p_pid = self._pid.params
        p_smc = self._smc.params

        x_d, y_d, z_d = setpoint[0], setpoint[1], setpoint[2]
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0

        # ── PID position loop ──────────────────────────────────────────────
        ex = x_d - s.x
        ey = y_d - s.y
        ez = z_d - s.z

        ax, self._pid._int_x = self._pid._pid_step(ex, -s.vx, self._pid._int_x, p_pid.pos_x, dt)
        ay, self._pid._int_y = self._pid._pid_step(ey, -s.vy, self._pid._int_y, p_pid.pos_y, dt)
        az, self._pid._int_z = self._pid._pid_step(ez, -s.vz, self._pid._int_z, p_pid.pos_z, dt)

        T, phi_d, theta_d = self.attitude_setpoint_from_position(ax, ay, az, psi_d, m, g)

        # ── SMC attitude loop ──────────────────────────────────────────────
        att_d = np.array([phi_d, theta_d, psi_d])
        e_att = att_d - s.euler
        e_att[2] = (e_att[2] + np.pi) % (2*np.pi) - np.pi

        if not self._smc._initialized:
            self._smc._prev_e_att = e_att.copy()
            self._smc._initialized = True

        e_att_dot = -(euler_rate_matrix(s.phi, s.theta) @ s.omega_body)
        sigma_att = e_att_dot + p_smc.lambda_att * e_att

        u_sw_att = np.array([
            (p_smc.eta_att + p_smc.Dmax_att) * SMController._sat(sigma_att[i], p_smc.phi_att)
            for i in range(3)
        ])
        tau_cmd = p_smc.lambda_att * e_att_dot + u_sw_att
        I = np.array([self.dp.Ixx, self.dp.Iyy, self.dp.Izz])
        tau = I * tau_cmd

        self._smc._prev_e_att = e_att
        return self.thrust_to_rotors(T, tau)

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        pos_err = state[0:3] - setpoint[0:3]
        e_att = state[6:9]
        sigma = -state[3:6] + self._smc.params.lambda_pos * pos_err
        return float(np.dot(sigma, sigma) + 0.1*np.dot(e_att, e_att))


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

CONTROLLER_REGISTRY = {
    "PID":          PIDController,
    "SMC":          SMController,
    "PID+SMC":      HybridPIDSMC,
}

def create_controller(
    name: str, drone_params: DroneParams,
    pid_params: PIDParams = None, smc_params: SMCParams = None
) -> BaseController:
    """Factory function to instantiate controllers by name."""
    if name == "PID":
        return PIDController(drone_params, pid_params)
    elif name == "SMC":
        return SMController(drone_params, smc_params)
    elif name == "PID+SMC":
        return HybridPIDSMC(drone_params, pid_params, smc_params)
    else:
        raise ValueError(f"Unknown controller: {name}")
