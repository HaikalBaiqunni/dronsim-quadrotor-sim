"""
drone_sim/core/simulation.py
============================
Simulation Engine

Orchestrates the interaction between:
    - Drone rigid-body dynamics (6-DOF)
    - Environment model (wind, turbulence, noise)
    - Flight controller (PID / SMC / Hybrid / MPC / ADRC / Geometric / ...)
    - Optional sensor fusion, safety filter and scheduled events (faults, gusts)

Provides real-time and batch simulation capabilities with
comprehensive data logging for analysis and visualization.
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
from collections import deque
from itertools import islice

from .dynamics import QuadrotorDynamics, DroneParams, DroneState
from .environment import EnvironmentModel, EnvironmentParams
from .controllers import BaseController, PIDController, PIDParams
from .estimation import StateEstimator, EstimatorParams


@dataclass
class SimConfig:
    """Simulation run configuration."""
    dt: float = 0.01           # Integration time step (s)
    duration: float = 60.0     # Max simulation time (s)
    real_time: bool = True     # Enforce real-time pacing
    real_time_factor: float = 1.0  # Speed multiplier

    # Trajectory
    trajectory_type: str = "hover"   # hover, circle, figure8, step, custom
    waypoints: List[np.ndarray] = field(default_factory=list)

    # Sensing
    use_estimator: bool = False      # sensor fusion instead of raw noisy state

    # Safety limits
    max_altitude: float = 50.0   # m
    max_velocity: float = 20.0   # m/s
    crash_on_ground: bool = True


@dataclass
class ScheduledEvent:
    """Something that happens at a given simulation time."""
    t: float
    kind: str                       # 'rotor_fault' | 'gust'
    rotor: int = 0                  # 0-based rotor index (rotor_fault)
    effectiveness: float = 0.0      # 0 = total failure, 1 = healthy
    vec: tuple = (0.0, 0.0, 0.0)    # gust wind velocity [m/s]
    duration: float = 2.0           # gust length [s]
    label: str = ""

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.kind == "rotor_fault":
            pct = int(round((1.0 - self.effectiveness) * 100))
            return f"Rotor {self.rotor + 1} {'failed' if pct >= 100 else f'-{pct}% thrust'}"
        return f"Gust {np.linalg.norm(self.vec):.1f} m/s"


@dataclass
class SimMetrics:
    """Real-time performance metrics."""
    # Tracking
    rmse_pos: float = 0.0
    rmse_att: float = 0.0
    max_pos_error: float = 0.0
    pos_error: float = 0.0

    # Control effort
    control_cost: float = 0.0
    total_energy: float = 0.0   # sum(omega_i^2 * dt), proxy for power
    max_tilt_deg: float = 0.0

    # Disturbance
    wind_force_magnitude: float = 0.0
    disturbance_est: float = 0.0    # |lumped disturbance| from ADRC observer [m/s^2]
    has_disturbance_est: bool = False

    # Sensing / safety
    est_error: float = 0.0          # |estimated - true position| [m]
    clearance: float = float("inf") # distance to nearest obstacle surface [m]
    cbf_active: bool = False

    # System
    sim_time: float = 0.0
    step_count: int = 0
    is_crashed: bool = False
    is_stable: bool = True
    finished: bool = False
    error_count: int = 0            # exceptions swallowed in controller / filter
    last_error: str = ""


class SimulationLogger:
    """Circular buffer logger for simulation data."""

    FIELDS = [
        'time',
        # State
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        'phi', 'theta', 'psi', 'p', 'q', 'r',
        # Setpoint
        'x_d', 'y_d', 'z_d', 'psi_d',
        # Errors
        'ex', 'ey', 'ez', 'e_phi', 'e_theta', 'e_psi',
        # Rotor speeds
        'w1', 'w2', 'w3', 'w4',
        # Forces
        'thrust', 'tau_phi', 'tau_theta', 'tau_psi',
        # Cost
        'control_cost', 'energy',
        # Environment
        'wind_x', 'wind_y', 'wind_z',
        # Estimation / observers / safety
        'est_ex', 'est_ey', 'est_ez',
        'dist_x', 'dist_y', 'dist_z',
        'clearance',
    ]

    def __init__(self, max_samples: int = 20000):
        self.max_samples = max_samples
        self._data: Dict[str, deque] = {
            f: deque(maxlen=max_samples) for f in self.FIELDS
        }

    def log(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._data:
                self._data[key].append(float(val))

    def get(self, field: str) -> np.ndarray:
        return np.array(self._data[field])

    def get_all(self) -> Dict[str, np.ndarray]:
        return {f: self.get(f) for f in self.FIELDS if len(self._data[f]) > 0}

    def get_tail(self, n: int) -> Dict[str, np.ndarray]:
        """Last n samples of every field (cheap: does not copy the history)."""
        out = {}
        for f in self.FIELDS:
            d = self._data[f]
            if len(d):
                k = min(n, len(d))
                out[f] = np.fromiter(reversed(list(islice(reversed(d), k))),
                                     dtype=float, count=k)
        return out

    def clear(self):
        for q in self._data.values():
            q.clear()

    def export_csv(self, path: str):
        import csv
        cols = [f for f in self.FIELDS if len(self._data[f]) > 0]
        n = len(self._data['time'])
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            arrays = [list(self._data[c]) for c in cols]
            for i in range(n):
                w.writerow([f"{a[i]:.6g}" if i < len(a) else "" for a in arrays])

    def export_mat(self, path: str):
        from scipy.io import savemat
        savemat(path, {f: self.get(f) for f in self.FIELDS
                       if len(self._data[f]) > 0})

    @property
    def n_samples(self) -> int:
        return len(self._data['time'])


class TrajectoryGenerator:
    """Generates reference trajectories for the drone to follow."""

    @staticmethod
    def hover(t: float, setpoint: np.ndarray) -> np.ndarray:
        return setpoint.copy()

    @staticmethod
    def step(t: float, config: dict) -> np.ndarray:
        """Step change in position at t = step_time."""
        step_time = config.get('step_time', 5.0)
        p0 = np.array(config.get('p0', [0, 0, 2, 0]))
        p1 = np.array(config.get('p1', [2, 0, 2, 0]))
        return p1 if t >= step_time else p0

    @staticmethod
    def circle(t: float, config: dict) -> np.ndarray:
        """Horizontal circle trajectory."""
        r    = config.get('radius', 2.0)
        freq = config.get('freq', 0.1)    # Hz
        alt  = config.get('altitude', 2.0)
        omega = 2 * np.pi * freq
        return np.array([
            r * np.cos(omega * t),
            r * np.sin(omega * t),
            alt,
            omega * t  # track yaw to face direction
        ])

    @staticmethod
    def figure8(t: float, config: dict) -> np.ndarray:
        """Lemniscate of Bernoulli (figure-8) trajectory."""
        a    = config.get('scale', 2.0)
        freq = config.get('freq', 0.05)
        alt  = config.get('altitude', 2.0)
        omega = 2 * np.pi * freq
        denom = 1 + np.sin(omega * t)**2
        x = a * np.cos(omega * t) / denom
        y = a * np.sin(omega * t) * np.cos(omega * t) / denom
        return np.array([x, y, alt, 0.0])

    @staticmethod
    def spiral(t: float, config: dict) -> np.ndarray:
        """Ascending spiral."""
        r    = config.get('radius', 2.0)
        freq = config.get('freq', 0.1)
        climb = config.get('climb_rate', 0.3)
        alt0  = config.get('alt0', 1.0)
        omega = 2 * np.pi * freq
        return np.array([
            r * np.cos(omega * t),
            r * np.sin(omega * t),
            alt0 + climb * t,
            omega * t
        ])




class DroneSimulation:
    """
    Main simulation engine.

    Thread-safe real-time simulation loop that can be run in a
    background thread while the GUI renders results.

    Optional hooks (all default to None):
        setpoint_provider(t, true_state) -> [x,y,z,psi]   mission reference
        setpoint_filter(t, state_meas, sp) -> sp'          safety filter (CBF)
        on_step(t, next_state, sp, cost)                   after each step

    Usage:
        sim = DroneSimulation(controller, drone_params, env_params)
        sim.start()
        ...
        sim.stop()
    """

    def __init__(
        self,
        controller: BaseController,
        drone_params: DroneParams = None,
        env_params: EnvironmentParams = None,
        sim_config: SimConfig = None,
        estimator_params: EstimatorParams = None,
    ):
        self.controller = controller   # expose for GUI introspection
        self.config = sim_config or SimConfig()
        self.dp = drone_params or DroneParams()

        self._dynamics = QuadrotorDynamics(self.dp)
        self._env = EnvironmentModel(env_params or EnvironmentParams(), self.config.dt)
        self.estimator: Optional[StateEstimator] = None
        if self.config.use_estimator:
            self.estimator = StateEstimator(self.dp, estimator_params, self.config.dt)

        # State
        self._state = np.zeros(12)
        self._state[2] = 0.1          # start slightly above ground
        self._hover_sp = np.array([0.0, 0.0, 2.0, 0.0])   # default: hover at 2m
        self._sp_now = self._hover_sp.copy()
        self._rotor_speeds = np.zeros(4)
        self._effectiveness = np.ones(4)
        self._last_accel = np.zeros(3)
        self._est_state = self._state.copy()

        # Metrics & logging
        self.metrics = SimMetrics()
        self.logger = SimulationLogger()

        # Trajectory
        self._traj_config: Dict = {}
        self._traj_type: str = "hover"

        # Scheduled events (faults, gusts)
        self.events: List[ScheduledEvent] = []
        self.event_log: List[tuple] = []          # (time, text) as they fire
        self._fired: set = set()

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._t = 0.0
        self._step_count = 0

        # Hooks
        self.setpoint_provider: Optional[Callable] = None
        self.setpoint_filter: Optional[Callable] = None
        self.on_step: Optional[Callable] = None     # called each step
        self.on_crash: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None

        # Position error history for RMSE
        self._pos_errors: deque = deque(maxlen=500)
        self._max_err = 0.0
        self.obstacles: List = []                  # for the clearance metric only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0):
        with self._lock:
            self._hover_sp = np.array([x, y, z, yaw])
            self._sp_now = self._hover_sp.copy()

    def set_trajectory(self, traj_type: str, config: dict = None):
        with self._lock:
            self._traj_type = traj_type
            self._traj_config = config or {}

    def set_initial_state(self, pos=(0,0,0.1), vel=(0,0,0),
                           euler=(0,0,0), omega=(0,0,0)):
        with self._lock:
            self._state = np.array(list(pos)+list(vel)+list(euler)+list(omega), dtype=float)
            self._est_state = self._state.copy()
            if self.estimator:
                self.estimator.reset(self._state)

    def add_event(self, ev: ScheduledEvent):
        self.events.append(ev)

    def reset(self):
        with self._lock:
            self._state = np.zeros(12)
            self._state[2] = 0.1
            self._t = 0.0
            self._step_count = 0
            self.metrics = SimMetrics()
            self.logger.clear()
            self._pos_errors.clear()
            self._max_err = 0.0
            self._effectiveness = np.ones(4)
            self._last_accel = np.zeros(3)
            self._fired.clear()
            self.event_log.clear()
            self.controller.reset()
            self._env.reset()
            self._env.gust_velocity = np.zeros(3)
            if self.estimator:
                self.estimator.reset(self._state)

    def start(self):
        """Start simulation in background thread."""
        if self._running:
            return
        self._running = True
        self._paused = False
        if self.estimator:
            self.estimator.reset(self._state)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()

    @property
    def estimated_state(self) -> np.ndarray:
        with self._lock:
            return self._est_state.copy()

    @property
    def setpoint(self) -> np.ndarray:
        with self._lock:
            return self._sp_now.copy()

    @property
    def time(self) -> float:
        return self._t

    @property
    def rotor_speeds(self) -> np.ndarray:
        with self._lock:
            return self._rotor_speeds.copy()

    @property
    def rotor_effectiveness(self) -> np.ndarray:
        with self._lock:
            return self._effectiveness.copy()

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _get_setpoint(self, t: float) -> np.ndarray:
        """Get current reference trajectory point."""
        ttype = self._traj_type
        cfg = self._traj_config

        if ttype == "circle":
            return TrajectoryGenerator.circle(t, cfg)
        elif ttype == "figure8":
            return TrajectoryGenerator.figure8(t, cfg)
        elif ttype == "spiral":
            return TrajectoryGenerator.spiral(t, cfg)
        elif ttype == "step":
            return TrajectoryGenerator.step(t, cfg)
        else:
            return self._hover_sp.copy()

    def _note_error(self, where: str, exc: Exception):
        """Remember an exception that the loop had to swallow so the GUI can show it."""
        self.metrics.error_count += 1
        self.metrics.last_error = f"{where}: {type(exc).__name__}: {exc}"

    def _apply_events(self, t: float):
        gust = np.zeros(3)
        for i, ev in enumerate(self.events):
            if ev.kind == "rotor_fault":
                if t >= ev.t and i not in self._fired:
                    self._fired.add(i)
                    self._effectiveness[ev.rotor] = ev.effectiveness
                    self.event_log.append((t, ev.describe()))
            elif ev.kind == "gust":
                if ev.t <= t < ev.t + ev.duration:
                    gust = gust + np.asarray(ev.vec, dtype=float)
                    if i not in self._fired:
                        self._fired.add(i)
                        self.event_log.append((t, ev.describe()))
        self._env.gust_velocity = gust

    def _run_loop(self):
        """Main real-time simulation loop."""
        dt = self.config.dt
        rtf = self.config.real_time_factor

        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue

            loop_start = time.perf_counter()

            with self._lock:
                state_now = self._state.copy()
                t = self._t

            # Scheduled events (rotor faults, gusts)
            self._apply_events(t)

            # Mission reference
            if self.setpoint_provider is not None:
                sp = np.asarray(self.setpoint_provider(t, state_now), dtype=float)
            else:
                sp = self._get_setpoint(t)

            # Get environment disturbances
            F_dist, tau_dist, ge_factor = self._env.get_disturbances(state_now, self._rotor_speeds)

            # Sensing: fused estimate or raw noisy measurement
            if self.estimator is not None:
                state_measured = self.estimator.step(state_now, self._last_accel, dt)
            else:
                state_measured = self._env.add_sensor_noise(state_now)

            # Safety filter acts on the reference given to the controller
            sp_ctrl = sp
            if self.setpoint_filter is not None:
                try:
                    sp_ctrl = self.setpoint_filter(t, state_measured, sp)
                except Exception as e:
                    self._note_error("safety filter", e)
                    sp_ctrl = sp

            # Control computation
            try:
                rotor_speeds = self.controller.compute(state_measured, sp_ctrl, dt)
            except Exception as e:
                self._note_error(self.controller.name, e)
                rotor_speeds = np.ones(4) * 100.0   # safe fallback

            # Actuator health (fault injection) and ground effect
            rotor_speeds_eff = rotor_speeds * np.sqrt(ge_factor * self._effectiveness)

            # Integrate dynamics
            next_state = self._dynamics.step(state_now, rotor_speeds_eff, dt)

            # Apply wind force (as velocity perturbation)
            next_state[3:6] += F_dist / self.dp.mass * dt

            # Safety / ground collision
            if next_state[2] < 0.0 and self.config.crash_on_ground:
                next_state[2] = 0.0
                next_state[3:6] *= 0.0  # stop on ground

            # Acceleration seen by the (simulated) accelerometer next step
            self._last_accel = np.clip((next_state[3:6] - state_now[3:6]) / dt, -40.0, 40.0)

            # Compute metrics
            cost = self.controller.get_cost(state_now, sp) if hasattr(self.controller, 'get_cost') else 0.0
            energy = float(np.sum(rotor_speeds**2) * dt)
            pos_err = float(np.linalg.norm(state_now[0:3] - sp[0:3]))
            self._pos_errors.append(pos_err)
            self._max_err = max(self._max_err, pos_err)

            d_est = getattr(self.controller, 'disturbance_estimate', None)
            d_est = np.zeros(3) if d_est is None else np.asarray(d_est, dtype=float)
            est_err = (state_measured[0:3] - state_now[0:3]) if self.estimator is not None else np.zeros(3)
            clearance = min((o.sdf(state_now[0:3]) for o in self.obstacles), default=float("inf"))

            # Log
            phi, theta, psi = state_now[6], state_now[7], state_now[8]
            w = self._env.get_wind_velocity()
            thrust = self.dp.kT * np.sum(rotor_speeds**2)

            self.logger.log(
                time=t,
                x=state_now[0], y=state_now[1], z=state_now[2],
                vx=state_now[3], vy=state_now[4], vz=state_now[5],
                phi=np.degrees(phi), theta=np.degrees(theta), psi=np.degrees(psi),
                p=state_now[9], q=state_now[10], r=state_now[11],
                x_d=sp[0], y_d=sp[1], z_d=sp[2],
                psi_d=sp[3] if len(sp)>3 else 0,
                ex=sp[0]-state_now[0], ey=sp[1]-state_now[1], ez=sp[2]-state_now[2],
                e_phi=0, e_theta=0, e_psi=0,
                w1=rotor_speeds[0], w2=rotor_speeds[1],
                w3=rotor_speeds[2], w4=rotor_speeds[3],
                thrust=thrust,
                tau_phi=0, tau_theta=0, tau_psi=0,
                control_cost=cost,
                energy=energy,
                wind_x=w[0], wind_y=w[1], wind_z=w[2],
                est_ex=est_err[0], est_ey=est_err[1], est_ez=est_err[2],
                dist_x=d_est[0], dist_y=d_est[1], dist_z=d_est[2],
                clearance=min(clearance, 99.0),
            )

            # Update metrics
            m = self.metrics
            errs = list(self._pos_errors)
            m.rmse_pos = float(np.sqrt(np.mean(np.array(errs)**2))) if errs else 0.0
            m.pos_error = pos_err
            m.max_pos_error = self._max_err
            m.control_cost = cost
            m.total_energy += energy
            m.sim_time = t
            m.step_count = self._step_count
            m.wind_force_magnitude = float(np.linalg.norm(F_dist))
            m.max_tilt_deg = float(max(abs(np.degrees(phi)), abs(np.degrees(theta))))
            m.has_disturbance_est = hasattr(self.controller, 'disturbance_estimate')
            m.disturbance_est = float(np.linalg.norm(d_est))
            m.est_error = float(np.linalg.norm(est_err))
            m.clearance = float(clearance)
            owner = getattr(self.setpoint_filter, '__self__', None)
            m.cbf_active = bool(getattr(owner, 'active', False))
            if m.max_tilt_deg > 80.0 and not m.is_crashed:
                m.is_crashed = True
                if self.on_crash:
                    try:
                        self.on_crash()
                    except Exception:
                        pass
            m.is_stable = m.max_tilt_deg < 45.0 and state_now[2] > 0.05

            # Advance state
            with self._lock:
                self._state = next_state
                self._est_state = state_measured
                self._sp_now = sp
                self._rotor_speeds = rotor_speeds
                self._t += dt
                self._step_count += 1

            # Callback
            if self.on_step:
                try:
                    self.on_step(t, next_state, sp, cost)
                except Exception:
                    pass

            # Real-time pacing
            if self.config.real_time:
                elapsed = time.perf_counter() - loop_start
                sleep_time = (dt / rtf) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Check completion
            if t >= self.config.duration:
                self._running = False
                self.metrics.finished = True
                if self.on_complete:
                    self.on_complete()
