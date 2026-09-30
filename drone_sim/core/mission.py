"""
drone_sim/core/mission.py
==========================
Everything that describes *what the drone should do* and *how it stays safe*,
independent of the GUI:

  * Waypoint / trajectory / avoidance settings  (plain dataclasses)
  * Global planning: RRT* is run per blocked segment and the result is spliced
    into the waypoint list
  * Reactive layer: APF (force) or CBF (setpoint filter)
  * ``build_simulation`` glues a controller, environment, estimator, events and
    the mission into a ready-to-start ``DroneSimulation``

Used by the GUI and by the comparison tab, so both run identical scenarios.

Author : Haikal Hakim Baiqunni
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .dynamics import DroneParams
from .environment import EnvironmentParams
from .obstacles import APFPlanner, RRTStar
from .safety import CBFParams, CBFSafetyFilter
from .simulation import (DroneSimulation, SimConfig, ScheduledEvent,
                         TrajectoryGenerator)
from .swarm import WaypointFollower, WaypointSpec
from .trajectory import Trajectory3D, TrajWaypoint, TrajectoryConfig


# ── Data model ───────────────────────────────────────────────────────────────

class Waypoint:
    __slots__ = ["x", "y", "z", "yaw", "speed", "hover_time"]

    def __init__(self, x=0.0, y=0.0, z=2.0, yaw=0.0, speed=1.0, hover_time=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)              # degrees
        self.speed = float(speed)
        self.hover_time = float(hover_time)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, math.radians(self.yaw)])

    def copy(self) -> "Waypoint":
        return Waypoint(self.x, self.y, self.z, self.yaw, self.speed, self.hover_time)

    def __repr__(self):
        return f"WP({self.x:.1f},{self.y:.1f},z={self.z:.1f})"


ALGORITHMS = ["None", "APF", "RRT*", "APF+RRT", "CBF", "RRT*+CBF"]
ALGO_HELP = {
    "None": "No avoidance. The drone flies straight through everything.",
    "APF": "Reactive push away from obstacles. Can get stuck behind large ones.",
    "RRT*": "Plans a collision-free route around blocked segments before flight.",
    "APF+RRT": "RRT* route plus reactive APF push while flying.",
    "CBF": "Safety filter with a formal barrier condition. Minimal intervention.",
    "RRT*+CBF": "RRT* route plus the CBF safety filter. Recommended.",
}
POLY_ORDERS = {0: "Straight segments (waypoint follower)",
               3: "Cubic", 5: "Quintic", 7: "Min-jerk (order 7)",
               9: "Min-snap (order 9)", 11: "Order 11"}


@dataclass
class TrajSettings:
    poly_order: int = 0            # 0 = piecewise waypoint follower
    time_method: str = "trap"      # 'trap' | 'constant'
    v_max: float = 2.0
    a_max: float = 3.0
    avg_speed: float = 1.5
    loop: bool = False


@dataclass
class AvoidSettings:
    algorithm: str = "None"
    safety_margin: float = 0.4     # m, used by RRT* and CBF
    apf_eta: float = 8.0
    apf_zeta: float = 1.5
    apf_d0: float = 1.5
    apf_fmax: float = 6.0
    rrt_iter: int = 800
    rrt_step: float = 0.6
    rrt_bias: float = 0.15
    rrt_near: float = 1.2
    cbf_k0: float = 4.0
    cbf_k1: float = 4.0
    cbf_influence: float = 3.0

    @property
    def uses_rrt(self) -> bool:
        return self.algorithm in ("RRT*", "APF+RRT", "RRT*+CBF")

    @property
    def uses_apf(self) -> bool:
        return self.algorithm in ("APF", "APF+RRT")

    @property
    def uses_cbf(self) -> bool:
        return self.algorithm in ("CBF", "RRT*+CBF")


PRESETS = ["hover", "circle", "figure8", "spiral"]


@dataclass
class MissionSpec:
    kind: str = "preset"                       # 'preset' | 'waypoints'
    preset: str = "hover"
    hover_sp: Tuple[float, float, float, float] = (0.0, 0.0, 2.0, 0.0)  # x y z yaw°
    waypoints: List[Waypoint] = field(default_factory=list)
    traj: TrajSettings = field(default_factory=TrajSettings)
    avoid: AvoidSettings = field(default_factory=AvoidSettings)
    obstacles: List = field(default_factory=list)
    duration: float = 60.0

    def label(self) -> str:
        if self.kind == "waypoints" and len(self.waypoints) >= 2:
            return f"{len(self.waypoints)} waypoints"
        return self.preset


def default_waypoints() -> List[Waypoint]:
    return [Waypoint(0, 0, 0.1), Waypoint(0, 0, 2.0), Waypoint(3, 0, 2.0),
            Waypoint(3, 3, 2.0), Waypoint(0, 3, 2.0), Waypoint(0, 0, 2.0)]


# ── Global planning ──────────────────────────────────────────────────────────

def plan_waypoints(wps: List[Waypoint], obstacles: List, av: AvoidSettings
                   ) -> Tuple[List[Waypoint], List[np.ndarray]]:
    """Splice RRT* detours into every waypoint segment that hits an obstacle.

    Returns (augmented waypoints, list of planned detour paths).
    """
    if not av.uses_rrt or not obstacles or len(wps) < 2:
        return list(wps), []
    # sample only around the mission (the default +-8 m box misses far waypoints)
    pts = np.array([[w.x, w.y, w.z] for w in wps] +
                   [[o.cx, o.cy, o.center[2]] for o in obstacles])
    lo, hi = pts.min(axis=0) - 3.0, pts.max(axis=0) + 3.0
    bounds = [[lo[0], hi[0]], [lo[1], hi[1]], [max(0.1, lo[2]), max(hi[2], 4.0)]]
    rrt = RRTStar(bounds=bounds, max_iter=av.rrt_iter, step_size=av.rrt_step,
                  goal_bias=av.rrt_bias, near_radius=av.rrt_near,
                  drone_r=av.safety_margin)
    out: List[Waypoint] = [wps[0]]
    paths: List[np.ndarray] = []
    for a, b in zip(wps[:-1], wps[1:]):
        pa = np.array([a.x, a.y, a.z])
        pb = np.array([b.x, b.y, b.z])
        if not rrt._collision_free(pa, pb, obstacles, n_check=20):
            path = rrt.plan(pa, pb, obstacles)
            if path and len(path) > 2:
                paths.append(np.array(path))
                for q in path[1:-1]:
                    out.append(Waypoint(q[0], q[1], q[2], a.yaw, 1.0, 0.0))
        out.append(b)
    return out, paths


# ── Setpoint providers ───────────────────────────────────────────────────────

def build_setpoint_provider(spec: MissionSpec, wps: List[Waypoint],
                            dt: float = 0.01) -> Tuple[Callable, Optional[np.ndarray], np.ndarray]:
    """Returns (provider(t, state) -> sp, poly_path or None, start_position)."""
    if spec.kind == "waypoints" and len(wps) >= 2:
        start = np.array([wps[0].x, wps[0].y, 0.1])
        tr = spec.traj
        if tr.poly_order >= 3:
            cfg = TrajectoryConfig(poly_order=tr.poly_order, time_method=tr.time_method,
                                   v_max=tr.v_max, a_max=tr.a_max,
                                   avg_speed=tr.avg_speed, loop=tr.loop)
            traj = Trajectory3D([TrajWaypoint(w.x, w.y, w.z, math.radians(w.yaw)) for w in wps], cfg)
            return (lambda t, s, _tr=traj: _tr.setpoint(t)), traj.sample_path(300), start
        follower = WaypointFollower(
            [WaypointSpec(w.x, w.y, w.z, math.radians(w.yaw), w.hover_time) for w in wps],
            loop=tr.loop)
        return (lambda t, s, _f=follower: _f.update(s[0:3], dt)), None, start

    x, y, z, yaw = spec.hover_sp
    hover = np.array([x, y, z, math.radians(yaw)])
    cfg = {'radius': 2.0, 'altitude': float(z), 'freq': 0.1, 'scale': 2.5,
           'climb_rate': 0.2, 'alt0': float(z)}
    gen = {"circle": TrajectoryGenerator.circle, "figure8": TrajectoryGenerator.figure8,
           "spiral": TrajectoryGenerator.spiral}.get(spec.preset)
    start = np.array([0.0, 0.0, 0.1])
    if gen is None:
        return (lambda t, s, _h=hover: _h.copy()), None, start
    return (lambda t, s, _g=gen, _c=cfg: _g(t, _c)), None, start


# ── Simulation factory ───────────────────────────────────────────────────────

@dataclass
class BuiltSim:
    sim: DroneSimulation
    waypoints: List[Waypoint]            # after RRT* splicing
    poly_path: Optional[np.ndarray]
    rrt_paths: List[np.ndarray]
    start: np.ndarray


def build_simulation(spec: MissionSpec, controller, dp: DroneParams,
                     env: EnvironmentParams, use_estimator: bool = False,
                     events: Optional[List[ScheduledEvent]] = None,
                     real_time: bool = True, dt: float = 0.01,
                     planned: Optional[Tuple[List[Waypoint], List[np.ndarray]]] = None
                     ) -> BuiltSim:
    """Create a DroneSimulation for ``spec`` (call ``sim.start()`` yourself).

    ``planned`` lets several runs reuse the same RRT* result so controller
    comparisons fly the identical route.
    """
    av = spec.avoid
    wps, rrt_paths = planned if planned is not None else (
        plan_waypoints(spec.waypoints, spec.obstacles, av) if spec.kind == "waypoints"
        else ([], []))

    cfg = SimConfig(dt=dt, duration=spec.duration, real_time=real_time,
                    use_estimator=use_estimator)
    sim = DroneSimulation(controller, dp, env, cfg)
    provider, poly_path, start = build_setpoint_provider(spec, wps, dt)
    sim.set_initial_state(pos=tuple(start))
    sim.setpoint_provider = provider
    sim.obstacles = list(spec.obstacles)
    for ev in events or []:
        sim.add_event(ScheduledEvent(**ev.__dict__))

    obs = spec.obstacles
    if obs and av.algorithm != "None":
        if av.uses_cbf:
            kp, kd = controller.position_gains
            flt = CBFSafetyFilter(obs, CBFParams(
                r_safe=av.safety_margin, k0=av.cbf_k0, k1=av.cbf_k1,
                influence=av.cbf_influence, nominal_kp=kp, nominal_kd=kd))
            sim.setpoint_filter = lambda t, s, sp, _f=flt: _f.filter(s[0:3], s[3:6], sp)
        if av.uses_apf:
            apf = APFPlanner(eta=av.apf_eta, zeta=av.apf_zeta, d0=av.apf_d0,
                             max_force=av.apf_fmax)

            def apf_hook(t, state, sp, cost, _apf=apf, _obs=obs, _sim=sim):
                F = _apf.repulsive_force(state[0:3], _obs)
                with _sim._lock:
                    _sim._state[3:6] += F * _sim.config.dt
            sim.on_step = apf_hook
    return BuiltSim(sim, wps, poly_path, rrt_paths, start)
