"""
drone_sim/core/compare.py
==========================
Headless controller comparison: fly the *same* mission (same RRT* route, same
environment, same events) with several controllers and collect metrics.

Author : Haikal Hakim Baiqunni
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .catalog import CATALOG, build_controller
from .dynamics import DroneParams
from .environment import EnvironmentParams
from .mission import MissionSpec, build_simulation, plan_waypoints
from .simulation import ScheduledEvent


@dataclass
class CompareResult:
    key: str
    label: str
    time: np.ndarray
    error: np.ndarray            # |position error| [m]
    pos: np.ndarray              # (N, 3)
    rmse: float
    max_error: float
    energy: float
    max_tilt: float
    min_clearance: float
    crashed: bool
    wall_time: float
    crash_time: Optional[float] = None   # first time tilt exceeded 80 deg (the core's crash criterion)


def _crash_time(d, crashed: bool) -> Optional[float]:
    if not crashed:
        return None
    tilt = np.maximum(np.abs(d["phi"]), np.abs(d["theta"]))
    idx = np.where(tilt > 80.0)[0]
    return float(d["time"][idx[0]]) if len(idx) else None


def run_comparison(spec: MissionSpec, keys: List[str],
                   values_by_key: Dict[str, dict], dp: DroneParams,
                   env: EnvironmentParams, use_estimator: bool,
                   events: Optional[List[ScheduledEvent]] = None,
                   duration: float = 20.0,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   cancel: Optional[Callable[[], bool]] = None
                   ) -> List[CompareResult]:
    """Fly every controller in ``keys`` for ``duration`` seconds (non real-time)."""
    saved = spec.duration
    spec.duration = duration
    planned = (plan_waypoints(spec.waypoints, spec.obstacles, spec.avoid)
               if spec.kind == "waypoints" else ([], []))
    results: List[CompareResult] = []
    try:
        for i, key in enumerate(keys):
            if cancel and cancel():
                break
            label = CATALOG[key].label
            if progress:
                progress(i, len(keys), label)
            ctrl = build_controller(key, dp, values_by_key.get(key))
            np.random.seed(1234)            # identical noise/turbulence draws
            built = build_simulation(spec, ctrl, dp, env, use_estimator,
                                     events, real_time=False, planned=planned)
            sim = built.sim
            t0 = time.time()
            sim.start()
            while sim.is_running and not (cancel and cancel()):
                time.sleep(0.02)
            sim.stop()
            d = sim.logger.get_all()
            if "time" not in d:
                continue
            err = np.sqrt(d["ex"] ** 2 + d["ey"] ** 2 + d["ez"] ** 2)
            m = sim.metrics
            results.append(CompareResult(
                key=key, label=label, time=d["time"], error=err,
                pos=np.column_stack((d["x"], d["y"], d["z"])),
                rmse=float(np.sqrt(np.mean(err ** 2))), max_error=float(err.max()),
                energy=float(m.total_energy), max_tilt=float(
                    max(np.abs(d["phi"]).max(), np.abs(d["theta"]).max())),
                min_clearance=float(d["clearance"].min()) if "clearance" in d else float("inf"),
                crashed=bool(m.is_crashed), wall_time=time.time() - t0,
                crash_time=_crash_time(d, bool(m.is_crashed))))
        if progress:
            progress(len(keys), len(keys), "")
    finally:
        spec.duration = saved
    return results
