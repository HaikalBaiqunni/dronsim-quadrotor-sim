"""
drone_sim/web/session.py
========================
UI-independent run session for the web front end. Owns one DroneSimulation
(built from the unchanged core), exposes a polling snapshot and a recorded
timeline. No NiceGUI imports, so it can be tested headless.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from drone_sim.core import (DroneParams, EnvironmentParams, WindCondition,
                            ScheduledEvent, BoxObstacle, CylinderObstacle)
from drone_sim.core.catalog import CATALOG, build_controller, default_values
from drone_sim.core.mission import (MissionSpec, Waypoint, build_simulation,
                                    default_waypoints)

WINDS = {w.name.title(): w for w in WindCondition}


def euler_to_R(phi: float, theta: float, psi: float) -> List[List[float]]:
    """ZYX body->world rotation matrix."""
    cf, sf, ct, st, cp, sp = (math.cos(phi), math.sin(phi), math.cos(theta),
                              math.sin(theta), math.cos(psi), math.sin(psi))
    return [[cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
            [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
            [-st, ct * sf, ct * cf]]


@dataclass
class Snapshot:
    t: float
    pos: np.ndarray
    euler: np.ndarray
    setpoint: np.ndarray
    rotors: np.ndarray            # rad/s
    effectiveness: np.ndarray
    error: float
    clearance: float
    running: bool
    events: List[tuple]
    trail: np.ndarray             # (N,3) recent positions


@dataclass
class StudioConfig:
    controller: str = "PID"
    wind: str = "Calm"
    turbulence: bool = False
    noise: bool = True
    estimator: bool = True
    preset: str = "circle"
    duration: float = 40.0
    use_waypoints: bool = False
    avoid: str = "None"
    obstacles: bool = False
    fault_t: float = 0.0           # 0 = none
    fault_rotor: int = 0
    fault_eff: float = 0.5
    gust_t: float = 0.0
    gust_speed: float = 4.0


class StudioSession:
    def __init__(self):
        self.cfg = StudioConfig()
        self.sim = None
        self.built = None
        self.spec: Optional[MissionSpec] = None
        self.dp = DroneParams()
        self.planned_events: List[ScheduledEvent] = []

    # -- setup ----------------------------------------------------------
    def make_events(self) -> List[ScheduledEvent]:
        c, ev = self.cfg, []
        if c.fault_t > 0:
            ev.append(ScheduledEvent(t=c.fault_t, kind="rotor_fault", rotor=c.fault_rotor,
                                     effectiveness=c.fault_eff))
        if c.gust_t > 0:
            ev.append(ScheduledEvent(t=c.gust_t, kind="gust", vec=(c.gust_speed, 0.0, 0.0),
                                     duration=2.0))
        return ev

    def make_spec(self) -> MissionSpec:
        c = self.cfg
        spec = MissionSpec(duration=c.duration)
        if c.use_waypoints:
            spec.kind = "waypoints"
            spec.waypoints = default_waypoints()
        else:
            spec.preset = c.preset
        spec.avoid.algorithm = c.avoid
        if c.obstacles:
            spec.obstacles = [BoxObstacle(cx=3.0, cy=1.5, cz=1.5, lx=1.0, ly=1.0, lz=3.0),
                              CylinderObstacle(cx=1.5, cy=3.0, z_bot=0.0, z_top=3.0, radius=0.5)]
        return spec

    def build(self, real_time: bool = True):
        self.stop()
        c = self.cfg
        self.spec = self.make_spec()
        env = EnvironmentParams(wind_condition=WINDS[c.wind], turbulence_enabled=c.turbulence,
                                noise_enabled=c.noise)
        ctrl = build_controller(c.controller, self.dp, default_values(c.controller))
        self.planned_events = self.make_events()
        self.built = build_simulation(self.spec, ctrl, self.dp, env, c.estimator,
                                      self.planned_events, real_time=real_time)
        self.sim = self.built.sim
        return self.built

    # -- run control ----------------------------------------------------
    def start(self):
        self.build()
        self.sim.start()

    def stop(self):
        if self.sim is not None:
            self.sim.stop()

    def toggle_pause(self) -> bool:
        if self.sim is None:
            return False
        if self.sim._paused:
            self.sim.resume()
        else:
            self.sim.pause()
        return self.sim._paused

    # -- polling --------------------------------------------------------
    def snapshot(self, trail_n: int = 400) -> Optional[Snapshot]:
        s = self.sim
        if s is None:
            return None
        st, sp = s.state, s.setpoint
        tail = s.logger.get_tail(trail_n)
        trail = (np.column_stack((tail["x"], tail["y"], tail["z"]))
                 if "x" in tail else np.zeros((0, 3)))
        clr = tail["clearance"][-1] if "clearance" in tail else float("inf")
        return Snapshot(t=s.time, pos=st[0:3], euler=st[6:9], setpoint=sp[0:3],
                        rotors=s.rotor_speeds, effectiveness=s.rotor_effectiveness,
                        error=float(np.linalg.norm(st[0:3] - sp[0:3])), clearance=float(clr),
                        running=s.is_running, events=list(s.event_log), trail=trail)
