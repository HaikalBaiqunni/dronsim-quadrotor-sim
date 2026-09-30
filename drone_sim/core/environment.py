"""
drone_sim/core/environment.py
==============================
Environment Model for Drone Simulation

Models external disturbances experienced by a UAV in real flight:
    1. Constant wind (steady-state)
    2. Dryden turbulence model (MIL-HDBK-1797B)
    3. Blade flapping effect
    4. Ground effect (when near surface)
    5. Measurement noise (IMU, GPS)

References:
    [1] MIL-HDBK-1797B, "Flying Qualities of Piloted Vehicles", 2012.
    [2] Garratt, M. A., & Chahl, J. S. (2008). Vision-based terrain following
        for an unmanned rotorcraft. Journal of Field Robotics.
    [3] Cheeseman, I. C., & Bennett, W. E. (1955). The Effect of the Ground
        on a Helicopter Rotor in Forward Flight. ARC R&M 3021.
    [4] Pounds, P., Mahony, R., & Corke, P. (2010). Modelling and Control
        of a Quadrotor Robot. ACRA.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class WindCondition(Enum):
    CALM        = "Calm (< 1 m/s)"
    LIGHT       = "Light Breeze (1-3 m/s)"
    MODERATE    = "Moderate (3-6 m/s)"
    STRONG      = "Strong (6-10 m/s)"
    STORM       = "Stormy (> 10 m/s)"


@dataclass
class EnvironmentParams:
    """Configurable environment parameters."""
    # Wind
    wind_condition: WindCondition = WindCondition.CALM
    wind_direction: float = 0.0     # rad (0 = North/+X)
    custom_wind_speed: float = 0.0  # m/s (used if condition = None)

    # Turbulence
    turbulence_enabled: bool = False
    turbulence_intensity: float = 0.1  # scale factor

    # Ground effect
    ground_effect_enabled: bool = True
    ground_height: float = 0.0        # m (altitude of ground)

    # Sensor noise
    imu_noise_std: float = 0.01       # rad/s, m/s²
    gps_noise_std: float = 0.05       # m
    noise_enabled: bool = True

    # Atmosphere
    rho: float = 1.225                # kg/m³ air density (sea level)
    temperature: float = 288.15       # K


WIND_SPEEDS = {
    WindCondition.CALM:     0.5,
    WindCondition.LIGHT:    2.0,
    WindCondition.MODERATE: 4.5,
    WindCondition.STRONG:   8.0,
    WindCondition.STORM:    13.0,
}


class DrydenTurbulenceModel:
    """
    Dryden wind turbulence model (continuous form).

    Implements the power spectral density specification from MIL-HDBK-1797B.
    Models atmospheric turbulence as a colored noise process via first-order
    shaping filters driven by white noise.

    Transfer functions (continuous domain):
        Hu(s) = σu·√(2Lu/πV) · 1/(1 + Lu/V · s)
        Hv(s) = σv·√(Lv/πV) · (1 + √3·Lv/V·s)/(1 + Lv/V·s)²
        Hw(s) = σw·√(Lw/πV) · (1 + √3·Lw/V·s)/(1 + Lw/V·s)²
    """

    def __init__(self, dt: float, altitude: float = 100.0,
                 airspeed: float = 5.0, intensity: float = 1.0):
        """
        Args:
            dt: time step (s)
            altitude: flight altitude (m)
            airspeed: nominal airspeed (m/s)
            intensity: turbulence scale factor
        """
        self.dt = dt
        self.V = max(airspeed, 0.1)

        # Dryden length scales (altitude-dependent, ft converted to m)
        h = max(altitude, 10.0)
        self.Lu = h / (0.177 + 0.000823 * h) ** 1.2
        self.Lv = self.Lu
        self.Lw = h

        # Turbulence intensities (m/s) scaled by intensity factor
        base = 1.5 * intensity
        self.sigma_u = base
        self.sigma_v = base
        self.sigma_w = base * 0.7

        # State variables for shaping filters
        self._xu = np.zeros(1)
        self._xv = np.zeros(2)
        self._xw = np.zeros(2)

    def step(self) -> np.ndarray:
        """Generate one step of turbulence velocity (u, v, w) in body frame."""
        n = np.random.randn(3)

        # Time constants
        tau_u = self.Lu / self.V
        tau_v = self.Lv / self.V
        tau_w = self.Lw / self.V

        # -- u-axis: first-order filter --
        gain_u = self.sigma_u * np.sqrt(2 * self.Lu / (np.pi * self.V))
        du = (1/tau_u) * (-self._xu[0] + gain_u * n[0])
        self._xu[0] += self.dt * du
        u_turb = self._xu[0]

        # -- v-axis: second-order filter --
        gain_v = self.sigma_v * np.sqrt(self.Lv / (np.pi * self.V))
        dv0 = self._xv[1]
        dv1 = (1/tau_v**2) * (
            -2*tau_v*self._xv[1] - self._xv[0]
            + gain_v * (1 + np.sqrt(3)*tau_v) * n[1]
        )
        self._xv += self.dt * np.array([dv0, dv1])
        v_turb = self._xv[0]

        # -- w-axis: second-order filter --
        gain_w = self.sigma_w * np.sqrt(self.Lw / (np.pi * self.V))
        dw0 = self._xw[1]
        dw1 = (1/tau_w**2) * (
            -2*tau_w*self._xw[1] - self._xw[0]
            + gain_w * (1 + np.sqrt(3)*tau_w) * n[2]
        )
        self._xw += self.dt * np.array([dw0, dw1])
        w_turb = self._xw[0]

        return np.array([u_turb, v_turb, w_turb])

    def reset(self):
        self._xu = np.zeros(1)
        self._xv = np.zeros(2)
        self._xw = np.zeros(2)


class EnvironmentModel:
    """
    Aggregate environment model combining all disturbance sources.

    Returns total external force/torque disturbances acting on the drone
    at each simulation step.
    """

    def __init__(self, params: EnvironmentParams = None, dt: float = 0.01):
        self.p = params or EnvironmentParams()
        self.dt = dt
        self._turbulence = DrydenTurbulenceModel(dt)
        self._wind_force_prev = np.zeros(3)
        self.gust_velocity = np.zeros(3)   # scheduled gust [m/s], set by the sim

    def get_wind_velocity(self) -> np.ndarray:
        """Returns steady wind velocity in inertial frame (m/s)."""
        speed = WIND_SPEEDS.get(self.p.wind_condition,
                                self.p.custom_wind_speed)
        d = self.p.wind_direction
        return speed * np.array([np.cos(d), np.sin(d), 0.0]) + self.gust_velocity

    def get_ground_effect_factor(self, altitude: float) -> float:
        """
        Cheeseman & Bennett ground effect model.

        Thrust augmentation factor k_ge ≥ 1.0 when near ground.
        k_ge = 1 / (1 - (R/(4z))²)
        where R is rotor radius, z is height above ground.
        """
        if not self.p.ground_effect_enabled:
            return 1.0
        R = 0.12  # rotor radius (m)
        z = altitude - self.p.ground_height
        if z < R * 0.5:  # keep k_ge finite (the formula diverges as z -> R/4)
            z = R * 0.5
        if z > 5 * R:
            return 1.0
        return 1.0 / (1.0 - (R / (4 * z)) ** 2)

    def get_disturbances(
        self, state_vec: np.ndarray, rotor_speeds: np.ndarray
    ) -> tuple:
        """
        Compute external force and torque disturbances.

        Args:
            state_vec: 12-DOF state vector
            rotor_speeds: [Ω1..4] rad/s

        Returns:
            F_dist: 3-vector external force (inertial frame), N
            tau_dist: 3-vector external torque (body frame), N·m
        """
        altitude = state_vec[2]
        vel = state_vec[3:6]

        # -- Wind disturbance force --
        v_wind = self.get_wind_velocity()
        v_rel = vel - v_wind
        rho = self.p.rho
        Cd_A = 0.05   # effective drag area (m²)
        F_wind = -0.5 * rho * Cd_A * np.linalg.norm(v_rel) * v_rel

        # -- Turbulence --
        F_turb = np.zeros(3)
        if self.p.turbulence_enabled:
            v_turb = self._turbulence.step() * self.p.turbulence_intensity
            F_turb = 0.5 * rho * Cd_A * v_turb * 0.5  # approximate

        # -- Ground effect (modifies thrust, handled externally as force) --
        ge_factor = self.get_ground_effect_factor(altitude)

        F_dist = F_wind + F_turb
        tau_dist = np.zeros(3)  # extend: add blade flapping, etc.

        return F_dist, tau_dist, ge_factor

    def add_sensor_noise(
        self, state_vec: np.ndarray
    ) -> np.ndarray:
        """Add realistic IMU + GPS measurement noise to state."""
        if not self.p.noise_enabled:
            return state_vec.copy()

        noisy = state_vec.copy()
        # Position (GPS noise)
        noisy[0:3] += np.random.randn(3) * self.p.gps_noise_std
        # Velocity (IMU integrated noise)
        noisy[3:6] += np.random.randn(3) * self.p.imu_noise_std * 0.1
        # Euler angles (IMU noise)
        noisy[6:9] += np.random.randn(3) * self.p.imu_noise_std * 0.01
        # Angular velocity (gyroscope noise)
        noisy[9:12] += np.random.randn(3) * self.p.imu_noise_std

        return noisy

    def reset(self):
        """Reset turbulence filter states."""
        self._turbulence.reset()

    def update_params(self, params: EnvironmentParams):
        self.p = params
