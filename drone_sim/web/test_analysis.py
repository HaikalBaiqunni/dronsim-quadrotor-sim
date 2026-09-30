"""Synthetic checks of the insight rules (no NiceGUI, no simulation). Run: python drone_sim/web/test_analysis.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from drone_sim.core import ScheduledEvent  # noqa: E402
from drone_sim.web.analysis import Run, insights, leaderboard  # noqa: E402

t = np.arange(0, 20, 0.01)
FAULT = [ScheduledEvent(t=8, kind="rotor_fault", rotor=0, effectiveness=0.5)]


def mk(key, err, crashed=False, crash_time=None, energy=100.0):
    return Run(key, key, t, err, np.zeros((len(t), 3)), float(np.sqrt((err ** 2).mean())),
               float(err.max()), energy, 10.0, 9.0, crashed, crash_time, list(FAULT))


def titles(runs):
    return [i.title for i in insights(runs)]


recovers = np.full_like(t, 0.05)
recovers[t >= 8] = 0.6 * np.exp(-(t[t >= 8] - 8)) + 0.05
stuck = np.full_like(t, 0.05)
stuck[t >= 8] = 0.05 + 0.05 * (t[t >= 8] - 8)
offset = np.full_like(t, 0.5)
dead = np.full_like(t, 0.2)

runs = [mk("A", recovers), mk("B", stuck), mk("C", offset), mk("D", dead, True, 8.3)]
got = insights(runs)
names = [i.title for i in got]
assert any("D crashed at 8.3 s" in n for n in names), names
assert any("A: error jumps" in n for n in names), names
b = next(i for i in got if "B: error jumps" in i.title)
assert b.kind == "bad" and "does not recover" in b.detail, b
assert next(i for i in got if "A: error jumps" in i.title).kind == "warn"
assert any("C settles with a steady offset" in n for n in names), names
assert any("A tracks tightest" in n for n in names), names
assert [r.key for r in leaderboard(runs)][-1] == "D"          # crashed run ranks last
assert not any("A settles" in n for n in names), names         # a recovered run has no offset
assert not insights([mk("E", np.full_like(t, 0.05)), mk("F", np.full_like(t, 0.05))]), "quiet runs give no insights"
print("analysis insight rules: ok")
