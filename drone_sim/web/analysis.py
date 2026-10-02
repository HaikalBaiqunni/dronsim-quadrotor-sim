"""
drone_sim/web/analysis.py
=========================
UI-independent analysis of comparison runs (layout F): flying a set of
controllers, the leaderboard, and insight cards. Every insight is a rule over the
recorded data; nothing is written by hand per controller.

Insight rules (thresholds are named constants below):
  crash      the run ended in a crash
  disturb    error after a fault or gust event vs just before it, and recovery time
  offset     the last seconds sit at a constant non-zero error
  winner     lowest RMSE, and by how much it beats the runner-up
  energy     lowest energy use, when it is not the winner
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from drone_sim.core import (DroneParams, EnvironmentParams, ScheduledEvent,
                            WindCondition)
from drone_sim.core.compare import CompareResult, run_comparison
from drone_sim.core.mission import MissionSpec

# ---- rule thresholds -------------------------------------------------------
PRE_WINDOW = 2.0          # s before an event used as the baseline
POST_WINDOW = 5.0         # s after an event searched for the peak
JUMP_RATIO = 2.0          # peak / baseline above this counts as a jump ...
JUMP_MIN = 0.2            # ... if it is also at least this many metres higher
RECOVER_RATIO = 1.5       # recovered when error <= baseline * this (and <= RECOVER_FLOOR)
RECOVER_FLOOR = 0.3       # m
OFFSET_TAIL = 5.0         # s
OFFSET_MIN = 0.3          # m, mean error in the tail
OFFSET_FLAT = 0.25        # std / mean below this means "constant"
WIN_MARGIN = 0.05         # winner must beat the runner-up by at least 5 % to be called out


@dataclass
class Run:
    key: str
    label: str
    time: np.ndarray
    error: np.ndarray
    pos: np.ndarray
    rmse: float
    max_error: float
    energy: float
    max_tilt: float
    min_clearance: float
    crashed: bool
    crash_time: Optional[float] = None
    events: List[ScheduledEvent] = field(default_factory=list)

    def at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.time, t), 0, len(self.time) - 1))


@dataclass
class Insight:
    kind: str           # 'good' | 'warn' | 'bad' | 'info'
    title: str
    detail: str
    run: Optional[str] = None


SCENARIOS = {
    "circle": "Circle",
    "figure8": "Figure-8",
    "hover": "Hover at 2 m",
}


def fly(keys: List[str], scenario: str, wind: WindCondition, turbulence: bool,
        fault_t: float, gust_t: float, duration: float,
        progress: Optional[Callable[[int, int, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> List[Run]:
    spec = MissionSpec(preset=scenario)
    if scenario == "hover":
        spec.hover_sp = (0.0, 0.0, 2.0, 0.0)
    env = EnvironmentParams(wind_condition=wind, turbulence_enabled=turbulence, noise_enabled=True)
    events = []
    if fault_t > 0:
        events.append(ScheduledEvent(t=fault_t, kind="rotor_fault", rotor=0, effectiveness=0.5))
    if gust_t > 0:
        events.append(ScheduledEvent(t=gust_t, kind="gust", vec=(4.0, 0.0, 0.0), duration=2.0))
    res = run_comparison(spec, keys, {}, DroneParams(), env, True, events,
                         duration=duration, progress=progress, cancel=cancel)
    return [Run(r.key, r.label, r.time, r.error, r.pos, r.rmse, r.max_error, r.energy,
                r.max_tilt, r.min_clearance, r.crashed, r.crash_time, list(events)) for r in res]


# ---- leaderboard -------------------------------------------------------------

def leaderboard(runs: List[Run]) -> List[Run]:
    """Crashed runs last; otherwise by RMSE."""
    return sorted(runs, key=lambda r: (r.crashed, r.rmse))


# ---- insights ----------------------------------------------------------------

def _disturbance(run: Run, ev: ScheduledEvent) -> Optional[Insight]:
    t = run.time
    pre = run.error[(t >= ev.t - PRE_WINDOW) & (t < ev.t)]
    post_mask = (t >= ev.t) & (t <= ev.t + POST_WINDOW)
    if len(pre) == 0 or not post_mask.any():
        return None
    base, peak = float(np.median(pre)), float(run.error[post_mask].max())
    if not (peak > base * JUMP_RATIO and peak - base >= JUMP_MIN):
        return None
    # recovered = first time after the peak that the error is back within limits
    i_peak = int(np.argmax(np.where(post_mask, run.error, -1)))
    back = np.where((np.arange(len(t)) > i_peak) & (run.error <= max(base * RECOVER_RATIO, RECOVER_FLOOR)))[0]
    what = ev.describe()
    if len(back):
        rec = f"back within {max(base * RECOVER_RATIO, RECOVER_FLOOR):.2f} m after {t[back[0]] - ev.t:.1f} s"
        kind = "warn"
    else:
        rec = "does not recover before the run ends"
        kind = "bad"
    return Insight(kind, f"{run.label}: error jumps {peak / max(base, 1e-3):.0f}× at {ev.t:.0f} s",
                   f"{what}: error goes from {base:.2f} m to {peak:.2f} m, {rec}.", run.key)


def insights(runs: List[Run]) -> List[Insight]:
    out: List[Insight] = []
    ok = [r for r in runs if not r.crashed]
    for r in runs:
        if r.crashed:
            tc = r.crash_time if r.crash_time is not None else r.time[-1]
            prior = [e for e in r.events if e.t <= tc]
            out.append(Insight("bad", f"{r.label} crashed at {tc:.1f} s",
                               "Tilt passed 80°" + (f", {tc - prior[-1].t:.1f} s after: {prior[-1].describe()}."
                                                   if prior else " with no scheduled event before it."), r.key))
            continue
        for ev in r.events:
            ins = _disturbance(r, ev)
            if ins:
                out.append(ins)
        tail = r.error[r.time >= r.time[-1] - OFFSET_TAIL]
        if len(tail) > 10 and tail.mean() >= OFFSET_MIN and tail.std() / tail.mean() <= OFFSET_FLAT:
            out.append(Insight("warn", f"{r.label} settles with a steady offset",
                               f"Over the last {OFFSET_TAIL:.0f} s the error stays near {tail.mean():.2f} m "
                               "with little variation: a constant bias or lag, not noise.", r.key))
    if len(ok) >= 2:
        b = leaderboard(ok)
        first, second = b[0], b[1]
        gain = 1 - first.rmse / second.rmse
        if gain >= WIN_MARGIN:
            out.append(Insight("good", f"{first.label} tracks tightest",
                               f"RMSE {first.rmse:.2f} m, {100 * gain:.0f}% lower than {second.label} "
                               f"({second.rmse:.2f} m).", first.key))
        cheapest = min(ok, key=lambda r: r.energy)
        if cheapest is not first:
            out.append(Insight("info", f"{cheapest.label} uses the least control effort",
                               f"Control effort {cheapest.energy / 1e6:.1f}e6 vs {first.energy / 1e6:.1f}e6 for {first.label} "
                               f"(sum of rotor speed squared over time, a power proxy), "
                               f"at {cheapest.rmse:.2f} m RMSE.", cheapest.key))
    elif len(ok) == 1 and len(runs) > 1:
        out.append(Insight("info", f"Only {ok[0].label} finished the run",
                           "Every other controller crashed, so there is nothing to rank it against.", ok[0].key))
    order = {"bad": 0, "warn": 1, "good": 2, "info": 3}
    out.sort(key=lambda i: order[i.kind])
    return out
