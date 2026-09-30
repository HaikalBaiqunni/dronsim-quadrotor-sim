"""
drone_sim/core/swarm.py
========================
Swarm Simulation Engine

Manages N independent drone simulations running concurrently,
each following its own waypoint trajectory.

Waypoint following logic:
  - Sequential waypoints with distance threshold switching
  - Optional hover time at each waypoint
  - Smooth interpolation between waypoints

Author : Haikal Hakim Baiqunni
License: MIT
"""

import numpy as np
import threading
import time
from typing import List, Optional, Callable
from dataclasses import dataclass
from collections import deque

from .dynamics import DroneParams, QuadrotorDynamics
from .environment import EnvironmentModel, EnvironmentParams
from .controllers import create_controller, PIDParams, SMCParams
from .simulation import SimConfig, SimulationLogger, SimMetrics
from .estimation import StateEstimator


@dataclass
class WaypointSpec:
    """Minimal waypoint data passed from GUI to engine."""
    x: float = 0.0
    y: float = 0.0
    z: float = 2.0
    yaw: float = 0.0       # rad
    hover_time: float = 0.0


class WaypointFollower:
    """
    State machine that advances through a list of waypoints.

    Switching logic:
        If |pos - wp| < REACH_RADIUS  → start hover timer
        If hover_timer >= wp.hover_time → advance to next waypoint
        After last waypoint: hold position (or loop)
    """
    REACH_RADIUS = 0.25   # m

    def __init__(self, waypoints: List[WaypointSpec], loop: bool = False):
        self._wps = waypoints
        self._loop = loop
        self._idx = 0
        self._hover_accum = 0.0
        self._done = False

    @property
    def current_setpoint(self) -> np.ndarray:
        if not self._wps:
            return np.array([0., 0., 2., 0.])
        wp = self._wps[min(self._idx, len(self._wps)-1)]
        return np.array([wp.x, wp.y, wp.z, wp.yaw])

    def update(self, pos: np.ndarray, dt: float) -> np.ndarray:
        """Update state machine and return current setpoint."""
        if not self._wps or self._done:
            return self.current_setpoint

        wp = self._wps[self._idx]
        target = np.array([wp.x, wp.y, wp.z])
        dist = np.linalg.norm(pos - target)

        if dist < self.REACH_RADIUS:
            self._hover_accum += dt
            if self._hover_accum >= wp.hover_time:
                self._hover_accum = 0.0
                self._idx += 1
                if self._idx >= len(self._wps):
                    if self._loop:
                        self._idx = 0
                    else:
                        self._idx = len(self._wps) - 1
                        self._done = True
        else:
            self._hover_accum = 0.0

        return self.current_setpoint

    def reset(self):
        self._idx = 0
        self._hover_accum = 0.0
        self._done = False

    @property
    def progress(self) -> float:
        """0.0 → 1.0 progress through waypoints."""
        if not self._wps:
            return 1.0
        return self._idx / len(self._wps)

    @property
    def current_wp_index(self) -> int:
        return self._idx


class SingleDroneSim:
    """
    Lightweight single-drone simulation (no threading).
    Stepped externally by SwarmSim.
    """

    def __init__(self, drone_id: int, waypoints: List[WaypointSpec],
                 drone_params: DroneParams, env_params: EnvironmentParams,
                 ctrl_name: str = "PID", dt: float = 0.01,
                 pid_params: PIDParams = None, smc_params: SMCParams = None,
                 controller=None, use_estimator: bool = False):
        self.id = drone_id
        self.dt = dt

        self._dynamics = QuadrotorDynamics(drone_params)
        self._env = EnvironmentModel(env_params, dt)
        self._controller = controller if controller is not None else             create_controller(ctrl_name, drone_params, pid_params, smc_params)
        self._estimator = StateEstimator(drone_params, dt=dt) if use_estimator else None
        self._last_accel = np.zeros(3)
        self.setpoint_filter = None          # optional CBF safety filter
        self.separation_filter = None        # CBFSafetyFilter used for drone-drone spacing
        self.separation_dist = 0.8
        self.peers = []                      # other SingleDroneSim objects
        self.obstacles = []
        self._follower = WaypointFollower(waypoints, loop=False)
        self.logger = SimulationLogger(max_samples=5000)
        self.metrics = SimMetrics()

        # State
        self._state = np.zeros(12)
        self._state[2] = 0.05
        self._rotor_speeds = np.zeros(4)
        self._t = 0.0
        self._pos_errors = deque(maxlen=300)

    def step(self):
        """Advance simulation by one dt."""
        pos = self._state[0:3]
        sp = self._follower.update(pos, self.dt)

        # Disturbances
        F_dist, _, ge = self._env.get_disturbances(self._state, self._rotor_speeds)
        if self._estimator is not None:
            state_meas = self._estimator.step(self._state, self._last_accel, self.dt)
        else:
            state_meas = self._env.add_sensor_noise(self._state)

        # Control
        sp_ctrl = sp
        if self.separation_filter is not None:
            agents = [(q._state[0:3].copy(), q._state[3:6].copy()) for q in self.peers]
            try:
                sp_ctrl = self.separation_filter.filter(state_meas[0:3], state_meas[3:6], sp,
                                                        agents, self.separation_dist + 0.3)   # +0.3 m buffer for tracking lag
            except Exception:
                sp_ctrl = sp
        elif self.setpoint_filter is not None:
            try:
                sp_ctrl = self.setpoint_filter(self._t, state_meas, sp)
            except Exception:
                sp_ctrl = sp
        try:
            rotors = self._controller.compute(state_meas, sp_ctrl, self.dt)
        except Exception:
            rotors = np.ones(4) * 500.0

        ge_safe = max(0.0, min(ge, 4.0))   # clamp ground effect factor
        rotors_eff = rotors * np.sqrt(ge_safe)

        # Integrate
        next_state = self._dynamics.step(self._state, rotors_eff, self.dt)
        next_state[3:6] += F_dist / self._dynamics.p.mass * self.dt
        next_state[2] = max(next_state[2], 0.0)
        self._last_accel = np.clip((next_state[3:6] - self._state[3:6]) / self.dt, -40.0, 40.0)

        self._state = next_state
        self._rotor_speeds = rotors
        self._t += self.dt

        # Metrics
        err = float(np.linalg.norm(self._state[0:3] - sp[0:3]))
        self._pos_errors.append(err)
        cost = self._controller.get_cost(self._state, sp) if hasattr(
            self._controller, 'get_cost') else 0.0
        energy = float(np.sum(rotors**2) * self.dt)

        w = self._env.get_wind_velocity()
        self.logger.log(
            time=self._t,
            x=self._state[0], y=self._state[1], z=self._state[2],
            vx=self._state[3], vy=self._state[4], vz=self._state[5],
            phi=np.degrees(self._state[6]),
            theta=np.degrees(self._state[7]),
            psi=np.degrees(self._state[8]),
            p=self._state[9], q=self._state[10], r=self._state[11],
            x_d=sp[0], y_d=sp[1], z_d=sp[2], psi_d=sp[3],
            ex=sp[0]-self._state[0], ey=sp[1]-self._state[1],
            ez=sp[2]-self._state[2],
            e_phi=0, e_theta=0, e_psi=0,
            w1=rotors[0], w2=rotors[1], w3=rotors[2], w4=rotors[3],
            thrust=self._dynamics.p.kT * np.sum(rotors**2),
            tau_phi=0, tau_theta=0, tau_psi=0,
            control_cost=cost, energy=energy,
            wind_x=w[0], wind_y=w[1], wind_z=w[2],
        )

        errs = list(self._pos_errors)
        self.metrics.rmse_pos = float(np.sqrt(np.mean(np.array(errs)**2))) if errs else 0.0
        self.metrics.control_cost = cost
        self.metrics.total_energy += energy
        self.metrics.sim_time = self._t
        self.metrics.pos_error = err
        self.metrics.max_pos_error = max(self.metrics.max_pos_error, err)
        self.metrics.max_tilt_deg = float(max(abs(np.degrees(self._state[6])), abs(np.degrees(self._state[7]))))
        self.metrics.is_stable = self.metrics.max_tilt_deg < 45.0
        self.metrics.is_crashed = self.metrics.max_tilt_deg > 80.0
        if self.obstacles:
            self.metrics.clearance = float(min(o.sdf(self._state[0:3]) for o in self.obstacles))

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def setpoint(self) -> np.ndarray:
        return self._follower.current_setpoint.copy()

    @property
    def rotor_speeds(self) -> np.ndarray:
        return self._rotor_speeds.copy()

    @property
    def wp_progress(self) -> float:
        return self._follower.progress

    @property
    def wp_index(self) -> int:
        return self._follower.current_wp_index

    def reset(self):
        self._state = np.zeros(12)
        self._state[2] = 0.05
        self._rotor_speeds = np.zeros(4)
        self._t = 0.0
        self._pos_errors.clear()
        self.metrics = SimMetrics()
        self.logger.clear()
        self._follower.reset()
        self._controller.reset()
        self._env.reset()
        if self._estimator is not None:
            self._estimator.reset(self._state)


class SwarmSimulation:
    """
    Orchestrates N drone simulations in a single background thread.
    All drones share the same time step and environment model.

    Thread safety: state access is lock-protected.
    """

    def __init__(self, drone_sims: List[SingleDroneSim],
                 real_time: bool = True, real_time_factor: float = 1.0):
        self._drones = drone_sims
        self._real_time = real_time
        self._rtf = real_time_factor
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._t = 0.0
        self.on_step: Optional[Callable] = None

    @property
    def n_drones(self) -> int:
        return len(self._drones)

    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def reset(self):
        with self._lock:
            for d in self._drones:
                d.reset()
            self._t = 0.0

    def get_states(self) -> List[np.ndarray]:
        with self._lock:
            return [d.state for d in self._drones]

    def get_setpoints(self) -> List[np.ndarray]:
        with self._lock:
            return [d.setpoint for d in self._drones]

    def get_rotor_speeds(self) -> List[np.ndarray]:
        with self._lock:
            return [d.rotor_speeds for d in self._drones]

    def get_metrics(self) -> List[SimMetrics]:
        return [d.metrics for d in self._drones]

    def get_loggers(self) -> List[SimulationLogger]:
        return [d.logger for d in self._drones]

    @property
    def time(self) -> float:
        return self._t

    def _loop(self):
        dt = self._drones[0].dt if self._drones else 0.01
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()

            with self._lock:
                for drone in self._drones:
                    drone.step()
                self._t += dt

            if self.on_step:
                try:
                    self.on_step(self._t)
                except Exception:
                    pass

            if self._real_time:
                elapsed = time.perf_counter() - t0
                sleep = (dt / self._rtf) - elapsed
                if sleep > 0:
                    time.sleep(sleep)
