"""
drone_sim/core/dynamics.py
==========================
6-DOF Quadrotor Dynamics Model

References:
    [1] Mahony, R., Kumar, V., & Corke, P. (2012). Multirotor Aerial Vehicles:
        Modeling, Estimation, and Control of Quadrotor. IEEE Robotics &
        Automation Magazine.
    [2] Bouabdallah, S. (2007). Design and Control of Quadrotors with
        Application to Autonomous Flying. EPFL PhD Thesis.
    [3] Mueller, M. W., & D'Andrea, R. (2016). Stability and control of
        a quadrocopter despite the complete loss of one, two, or three
        propellers. ICRA 2014.
    [4] Mistler, V., Benallegue, A., & M'Sirdi, N. K. (2001). Exact
        linearization and noninteracting control of a 4 rotors helicopter
        via dynamic feedback. IEEE RO-MAN.

State vector x ∈ ℝ¹²:
    [x, y, z,          -- position in inertial frame (m)
     vx, vy, vz,       -- velocity in inertial frame (m/s)
     φ, θ, ψ,          -- Euler angles: roll, pitch, yaw (rad)
     p, q, r]          -- angular velocity in body frame (rad/s)

Input vector u ∈ ℝ⁴:
    [Ω1², Ω2², Ω3², Ω4²]  -- squared rotor speeds (rad²/s²)
    or equivalently:
    [T, τ_φ, τ_θ, τ_ψ]    -- total thrust + torques
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DroneParams:
    """Physical parameters of a quadrotor UAV.

    Default values correspond to a typical 250mm racing/research quadrotor.
    All values in SI units.
    """
    # --- Mass & Inertia ---
    mass: float = 0.5          # kg
    Ixx: float = 4.9e-3        # kg·m² (roll inertia)
    Iyy: float = 4.9e-3        # kg·m² (pitch inertia)
    Izz: float = 8.8e-3        # kg·m² (yaw inertia)
    Ir: float = 3.36e-5        # kg·m² (rotor inertia)

    # --- Geometry ---
    arm_length: float = 0.23   # m (motor-to-center distance)

    # --- Aerodynamic coefficients ---
    # kT: thrust coefficient (N·s²/rad²)
    # Calibrated so hover occurs at ~500 rad/s for 0.5 kg drone
    # T_hover = m*g = 4.905 N = 4 * kT * omega_hover²
    # => kT = 4.905 / (4 * 500²) ≈ 4.905e-6
    kT: float = 4.905e-6       # N·s²/rad² (thrust coefficient)
    kD: float = 9.81e-8        # N·m·s²/rad² (drag torque, ~2% of kT)

    # --- Drag ---
    kDrag_lin: float = 0.01    # linear drag coefficient (N·s/m)
    kDrag_ang: float = 0.005   # angular drag coefficient (N·m·s/rad)

    # --- Gravity ---
    g: float = 9.81            # m/s²

    # --- Rotor limits ---
    omega_max: float = 1000.0  # rad/s max rotor speed (≈9549 RPM)
    omega_min: float = 0.0     # rad/s min rotor speed

    # --- Mixer matrix (X-frame configuration) ---
    # Rotor layout (top view):
    #    1(CW)  2(CCW)
    #    3(CCW) 4(CW)
    @property
    def mixer(self) -> np.ndarray:
        """Returns 4×4 mixer matrix mapping [T, τ_φ, τ_θ, τ_ψ] ← [kT·Ωi²]."""
        l = self.arm_length
        kT = self.kT
        kD = self.kD
        # [F1, F2, F3, F4] -> [T, τ_φ, τ_θ, τ_ψ]
        # Rotor spin directions: 1=CW, 2=CCW, 3=CCW, 4=CW
        return np.array([
            [ 1,    1,    1,    1  ],   # Thrust
            [ 0,   -l,    0,    l  ],   # Roll  (τ_φ)
            [ l,    0,   -l,    0  ],   # Pitch (τ_θ)
            [-kD/kT, kD/kT, -kD/kT, kD/kT],  # Yaw (τ_ψ)
        ])


class DroneState:
    """Container for the 12-DOF drone state with named accessors."""

    DIM = 12

    def __init__(self, x: np.ndarray = None):
        if x is None:
            self._x = np.zeros(12)
        else:
            self._x = np.array(x, dtype=float)

    # -- Position (inertial frame) --
    @property
    def pos(self) -> np.ndarray:       return self._x[0:3]
    @property
    def x(self) -> float:              return self._x[0]
    @property
    def y(self) -> float:              return self._x[1]
    @property
    def z(self) -> float:              return self._x[2]

    # -- Velocity (inertial frame) --
    @property
    def vel(self) -> np.ndarray:       return self._x[3:6]
    @property
    def vx(self) -> float:             return self._x[3]
    @property
    def vy(self) -> float:             return self._x[4]
    @property
    def vz(self) -> float:             return self._x[5]

    # -- Euler angles (ZYX convention) --
    @property
    def euler(self) -> np.ndarray:    return self._x[6:9]
    @property
    def phi(self) -> float:            return self._x[6]   # roll
    @property
    def theta(self) -> float:          return self._x[7]   # pitch
    @property
    def psi(self) -> float:            return self._x[8]   # yaw

    # -- Angular velocity (body frame) --
    @property
    def omega_body(self) -> np.ndarray: return self._x[9:12]
    @property
    def p(self) -> float:              return self._x[9]
    @property
    def q(self) -> float:              return self._x[10]
    @property
    def r(self) -> float:              return self._x[11]

    @property
    def vec(self) -> np.ndarray:       return self._x.copy()

    def __repr__(self):
        return (f"DroneState(pos={self.pos}, vel={self.vel}, "
                f"euler={np.degrees(self.euler):.1f}°, ω={self.omega_body})")


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler → rotation matrix R (body-to-inertial).

    R = Rz(ψ) @ Ry(θ) @ Rx(φ)
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [-sth,      cth*sphi,                   cth*cphi                 ],
    ])


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """Maps body angular velocity ω_body → Euler angle rates [φ̇, θ̇, ψ̇].

    [φ̇]   [1  sin(φ)tan(θ)   cos(φ)tan(θ)] [p]
    [θ̇] = [0  cos(φ)         -sin(φ)      ] [q]
    [ψ̇]   [0  sin(φ)/cos(θ)  cos(φ)/cos(θ)] [r]

    Note: Singularity at θ = ±90°. For full attitude, quaternions preferred.
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    # Guard against gimbal lock
    if abs(cth) < 1e-6:
        cth = np.sign(cth) * 1e-6

    tth = sth / cth
    return np.array([
        [1,  sphi*tth,  cphi*tth],
        [0,  cphi,     -sphi    ],
        [0,  sphi/cth,  cphi/cth],
    ])


class QuadrotorDynamics:
    """
    6-DOF quadrotor rigid-body dynamics.

    The equations of motion follow Newton-Euler formulation:
        ṗ = v
        v̇ = (1/m)[R·F_thrust + F_drag] - g·ẑ
        Φ̇ = W(Φ)·ω_b
        ω̇_b = I⁻¹[τ - ω_b × (I·ω_b) - Γ_gyro]

    where Γ_gyro accounts for gyroscopic effects of spinning rotors.
    """

    def __init__(self, params: DroneParams = None):
        self.p = params or DroneParams()
        self._I = np.diag([self.p.Ixx, self.p.Iyy, self.p.Izz])
        self._I_inv = np.linalg.inv(self._I)

    def compute_forces_torques(
        self, rotor_speeds: np.ndarray, state: DroneState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute thrust force and body torques from rotor speeds.

        Args:
            rotor_speeds: Ω = [Ω1, Ω2, Ω3, Ω4] in rad/s
            state: current drone state

        Returns:
            F_thrust: 3-vector (body frame), N
            tau: 3-vector [τ_φ, τ_θ, τ_ψ] (body frame), N·m
        """
        Omega2 = rotor_speeds ** 2
        kT, kD, l = self.p.kT, self.p.kD, self.p.arm_length

        # Individual rotor thrusts
        T_i = kT * Omega2  # [T1, T2, T3, T4]

        # Total thrust (body z-axis)
        T_total = np.sum(T_i)
        F_thrust = np.array([0.0, 0.0, T_total])

        # Torques (X-frame: rotors at 45° arms)
        # Rotor spin: 1=CW(-), 2=CCW(+), 3=CW(-), 4=CCW(+)
        tau_phi   = l * (-T_i[1] + T_i[3])                     # roll
        tau_theta = l * ( T_i[0] - T_i[2])                     # pitch
        tau_psi   = kD * (-Omega2[0] + Omega2[1] - Omega2[2] + Omega2[3])  # yaw

        tau = np.array([tau_phi, tau_theta, tau_psi])
        return F_thrust, tau

    def gyroscopic_torque(
        self, rotor_speeds: np.ndarray, omega_body: np.ndarray
    ) -> np.ndarray:
        """Gyroscopic torque from spinning rotors.

        Γ_gyro = Ir · (ω_b × ẑ) · Ω_r
        where Ω_r = Ω1 - Ω2 + Ω3 - Ω4 (net rotor angular momentum)
        """
        Omega_r = (rotor_speeds[0] - rotor_speeds[1]
                   + rotor_speeds[2] - rotor_speeds[3])
        e3 = np.array([0.0, 0.0, 1.0])
        return self.p.Ir * Omega_r * np.cross(omega_body, e3)

    def derivatives(
        self, t: float, x: np.ndarray, rotor_speeds: np.ndarray
    ) -> np.ndarray:
        """Compute ẋ = f(x, u) for the 12-DOF quadrotor.

        Args:
            t: time (unused, for ODE solver compatibility)
            x: state vector [pos(3), vel(3), euler(3), omega_body(3)]
            rotor_speeds: [Ω1, Ω2, Ω3, Ω4] rad/s

        Returns:
            dx/dt: 12-vector
        """
        state = DroneState(x)
        phi, theta, psi = state.phi, state.theta, state.psi
        vel = state.vel
        omega_b = state.omega_body

        # Clip rotor speeds to physical limits
        rotor_speeds = np.clip(rotor_speeds, self.p.omega_min, self.p.omega_max)

        # -- Forces --
        R = rotation_matrix(phi, theta, psi)
        F_thrust, tau = self.compute_forces_torques(rotor_speeds, state)

        # Linear drag (inertial frame)
        F_drag = -self.p.kDrag_lin * vel

        # Gravity
        F_gravity = np.array([0.0, 0.0, -self.p.mass * self.p.g])

        # Position derivative
        pos_dot = vel

        # Velocity derivative (Newton's second law)
        vel_dot = (R @ F_thrust + F_drag) / self.p.mass + F_gravity / self.p.mass

        # -- Attitude --
        W = euler_rate_matrix(phi, theta)
        euler_dot = W @ omega_b

        # Angular velocity derivative (Euler's rotation equation)
        Gamma_gyro = self.gyroscopic_torque(rotor_speeds, omega_b)
        angular_drag = -self.p.kDrag_ang * omega_b
        omega_dot = self._I_inv @ (
            tau - np.cross(omega_b, self._I @ omega_b)
            - Gamma_gyro + angular_drag
        )

        return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])

    def step(
        self, state: np.ndarray, rotor_speeds: np.ndarray,
        dt: float, method: str = 'rk4'
    ) -> np.ndarray:
        """Integrate dynamics one step forward.

        Args:
            state: current 12-vector
            rotor_speeds: [Ω1..4] rad/s
            dt: time step (s)
            method: 'rk4' or 'euler'

        Returns:
            next state 12-vector
        """
        if method == 'rk4':
            return self._rk4_step(state, rotor_speeds, dt)
        else:
            return state + dt * self.derivatives(0, state, rotor_speeds)

    def _rk4_step(
        self, x: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """Classic 4th-order Runge-Kutta integration."""
        k1 = self.derivatives(0, x,            u)
        k2 = self.derivatives(0, x + dt/2*k1,  u)
        k3 = self.derivatives(0, x + dt/2*k2,  u)
        k4 = self.derivatives(0, x + dt*k3,    u)
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
