"""
drone_sim/core/safety.py
=========================
Control Barrier Function (CBF) safety filter for obstacle avoidance

For every obstacle i the barrier is  h_i(p) = sdf_i(p) - r_safe  (h >= 0 is safe).
With double-integrator translational dynamics the condition

        n_i · a  >=  -k1 (n_i · v) - k0 h_i         (n_i = ∇h_i)

makes the safe set forward-invariant. The filter finds the acceleration
closest to the nominal one that satisfies every active constraint, and turns
the correction into a setpoint offset so the real controller (PID, ADRC,
MPC, ...) tracks it. Guarantees hold for the double-integrator model, not the
full closed-loop quadrotor; the safety margin absorbs the mismatch.

A pure CBF has local minima (obstacle directly between drone and goal).
Combine it with RRT* (`RRT*+CBF` in the GUI) to plan around them.

References:
  [1] Ames, A. D. et al. (2019). Control barrier functions: theory and
      applications. ECC 2019.
  [2] Wu, G. & Sreenath, K. (2016). Safety-critical control of a planar
      quadrotor. ACC 2016.

Author : Haikal Hakim Baiqunni
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CBFParams:
    r_safe: float = 0.45       # m, drone radius + margin
    k0: float = 4.0            # barrier gain (class-K, position)
    k1: float = 4.0            # barrier gain (velocity)
    influence: float = 3.0     # m, ignore obstacles farther than this
    a_max: float = 6.0         # m/s², limit on the filtered acceleration
    nominal_kp: float = 1.5    # nominal PD used to define the "desired" accel
    nominal_kd: float = 2.2
    max_shift: float = 2.5     # m, cap on how far the setpoint may be moved
    smooth: float = 0.6        # 0..1 low-pass on the setpoint shift (higher = smoother)
    swirl: float = 2.5         # m/s², sideways bias between drones to break head-on deadlock


class CBFSafetyFilter:
    """Minimum-intervention safety filter operating on the setpoint."""

    def __init__(self, obstacles: List, params: CBFParams = None):
        self.obstacles = obstacles
        self.p = params or CBFParams()
        self.last_h_min = float("inf")
        self.active = False
        self._shift = np.zeros(3)

    @staticmethod
    def _grad(obs, pos: np.ndarray) -> np.ndarray:
        eps = 0.02
        g = np.zeros(3)
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            g[k] = (obs.sdf(pos + d) - obs.sdf(pos - d)) / (2 * eps)
        n = np.linalg.norm(g)
        return g / n if n > 1e-6 else g

    def filter(self, pos: np.ndarray, vel: np.ndarray,
               setpoint: np.ndarray, agents=None, r_sep: float = 0.8) -> np.ndarray:
        """Return a (possibly shifted) setpoint that keeps h_i >= 0.

        ``agents``: optional list of (position, velocity) of other drones. Each one
        adds the barrier h = |p - p_j| - r_sep on the relative state (the other
        drone is assumed not to accelerate, which is conservative).
        """
        p = self.p
        a_nom = p.nominal_kp * (setpoint[0:3] - pos) - p.nominal_kd * vel

        rows, rhs = [], []
        swirl = np.zeros(3)
        h_min = float("inf")
        for obs in self.obstacles:
            d = obs.sdf(pos)
            h = d - p.r_safe
            h_min = min(h_min, h)
            if d > p.influence:
                continue
            n = self._grad(obs, pos)
            rows.append(n)
            rhs.append(-p.k1 * float(n @ vel) - p.k0 * h)
        for pj, vj in agents or []:
            diff = pos - pj
            d = float(np.linalg.norm(diff))
            if d > p.influence + r_sep or d < 1e-6:
                continue
            n = diff / d
            h = d - r_sep
            h_min = min(h_min, h)
            rows.append(n)
            rhs.append(-p.k1 * float(n @ (vel - vj)) - p.k0 * h)
            w = max(0.0, 1.0 - d / (p.influence + r_sep))
            swirl += p.swirl * w * np.array([-n[1], n[0], 0.0])   # pass on the right
        self.last_h_min = h_min

        if not rows:
            self.active = False
            self._shift *= p.smooth                  # release smoothly
            if np.linalg.norm(self._shift) < 1e-3:
                self._shift[:] = 0.0
                return setpoint
            sp = setpoint.copy()
            sp[0:3] = setpoint[0:3] + self._shift
            return sp

        a_nom = a_nom + swirl
        a = a_nom.copy()
        for _ in range(25):                      # sequential projection
            for n, b in zip(rows, rhs):
                viol = b - float(n @ a)
                if viol > 0.0:
                    a = a + n * viol
            mag = np.linalg.norm(a)
            if mag > p.a_max:
                a *= p.a_max / mag
        self.active = bool(np.linalg.norm(a - a_nom) > 1e-3)

        sp = setpoint.copy()
        shift = (a - a_nom) / p.nominal_kp + swirl / p.nominal_kp
        n_shift = float(np.linalg.norm(shift))
        if n_shift > p.max_shift:
            shift *= p.max_shift / n_shift
        self._shift = p.smooth * self._shift + (1.0 - p.smooth) * shift
        sp[0:3] = setpoint[0:3] + self._shift
        return sp
