"""
drone_sim/web/ratings.py
========================
Controller ratings MEASURED from simulation runs (core.compare.run_comparison),
never hand-written. Two scenarios are flown for every rated controller:

  precision   circle trajectory, calm air + sensor noise, estimator on.
              Metric: position RMSE after a 5 s settle (the climb from the
              ground is excluded). Stars are relative to the best controller.
  robustness  hover in moderate wind + turbulence, rotor 1 loses 50 % thrust at
              t = 8 s. Metric: crash yes/no, RMSE in the last 8 s.

CPU cost is wall time of the precision run relative to the fastest controller.
Results are cached on disk, keyed by a hash of the core sources.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from drone_sim.core import (DroneParams, EnvironmentParams, ScheduledEvent,
                            WindCondition)
from drone_sim.core.catalog import CATALOG
from drone_sim.core.compare import run_comparison
from drone_sim.core.mission import MissionSpec

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings_cache.json")
DURATION = 20.0
SETTLE = 5.0
TAIL = 8.0
# star thresholds: value <= limit -> that many stars (first match wins)
# precision is relative: RMSE / best RMSE among rated controllers (the circle task is lag-dominated,
# so absolute thresholds would give everyone one star)
PRECISION_STARS = ((1.25, 5), (2.0, 4), (3.0, 3), (5.0, 2))
ROBUST_STARS = ((0.10, 5), (0.25, 4), (0.50, 3), (1.00, 2))
CPU_STARS = ((1.5, 5), (2.5, 4), (5.0, 3), (10.0, 2))


def stars(value: float, table) -> int:
    for limit, n in table:
        if value <= limit:
            return n
    return 1


@dataclass
class Rating:
    key: str
    precision_rmse: float
    precision_stars: int
    crashed: bool
    robust_rmse: float                 # nan if crashed
    robust_stars: int                  # 0 when crashed
    cpu_rel: float                     # wall time relative to the fastest
    cpu_stars: int
    curve_t: List[float] = field(default_factory=list)       # precision run, for the preview
    curve_err: List[float] = field(default_factory=list)


def source_hash() -> str:
    core = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core")
    h = hashlib.sha1()
    for n in sorted(os.listdir(core)):
        if n.endswith(".py"):
            with open(os.path.join(core, n), "rb") as f:
                h.update(n.encode() + f.read())
    h.update(repr((DURATION, SETTLE, TAIL)).encode())
    return h.hexdigest()[:12]


def rated_keys() -> List[str]:
    return [k for k, v in CATALOG.items() if not v.experimental]


def _tail_rmse(r, t0: float) -> float:
    m = r.time >= t0
    return float(np.sqrt(np.mean(r.error[m] ** 2))) if m.any() else float("nan")


def measure(progress: Optional[Callable[[str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Rating]:
    keys, dp = rated_keys(), DroneParams()
    say = progress or (lambda s: None)

    say("Precision runs (circle, calm air)")
    calm = EnvironmentParams(noise_enabled=True)
    prec = {r.key: r for r in run_comparison(MissionSpec(preset="circle"), keys, {}, dp, calm,
                                             True, [], duration=DURATION, cancel=cancel)}
    say("Robustness runs (wind + 50% rotor loss)")
    windy = EnvironmentParams(wind_condition=WindCondition.MODERATE, turbulence_enabled=True,
                              noise_enabled=True)
    fault = [ScheduledEvent(t=8.0, kind="rotor_fault", rotor=0, effectiveness=0.5)]
    hov = MissionSpec(preset="hover")
    hov.hover_sp = (0.0, 0.0, 2.0, 0.0)
    rob = {r.key: r for r in run_comparison(hov, keys, {}, dp, windy, True, fault,
                                            duration=DURATION, cancel=cancel)}
    fastest = min(r.wall_time for r in prec.values())
    best_rmse = min(_tail_rmse(r, SETTLE) for r in prec.values())
    out: Dict[str, Rating] = {}
    for k in keys:
        if k not in prec or k not in rob:
            continue
        p, r = prec[k], rob[k]
        p_rmse = _tail_rmse(p, SETTLE)
        crashed = bool(r.crashed)
        r_rmse = float("nan") if crashed else _tail_rmse(r, DURATION - TAIL)
        rel = p.wall_time / fastest
        step = max(1, len(p.time) // 100)
        out[k] = Rating(k, p_rmse, stars(p_rmse / best_rmse, PRECISION_STARS), crashed, r_rmse,
                        0 if crashed else stars(r_rmse, ROBUST_STARS), rel,
                        stars(rel, CPU_STARS), p.time[::step].tolist(), p.error[::step].tolist())
    return out


def load_cached() -> Optional[Dict[str, Rating]]:
    """Ratings from disk if they match the current core sources, else None."""
    if not os.path.exists(CACHE):
        return None
    try:
        with open(CACHE) as f:
            blob = json.load(f)
        if blob.get("hash") == source_hash():
            return {k: Rating(**v) for k, v in blob["ratings"].items()}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def measure_and_save(progress=None, cancel=None) -> Dict[str, Rating]:
    res = measure(progress, cancel)
    if cancel and cancel():
        return res
    try:
        with open(CACHE, "w") as f:
            json.dump({"hash": source_hash(), "ratings": {k: asdict(v) for k, v in res.items()}}, f)
    except OSError:
        pass
    return res
