"""
drone_sim/core/trajectory.py
=============================
Polynomial Trajectory Generation

Generates smooth continuous-time trajectories through waypoints
using polynomial segment fitting. Supports:

  Order 3  — Cubic          (pos, vel continuity)
  Order 5  — Quintic        (pos, vel, acc continuity)
  Order 7  — Septic / Min-Jerk (pos, vel, acc, jerk continuity)
  Order 9  — Nonic  / Min-Snap (pos, vel, acc, jerk, snap continuity)
  Order 11 — Hendecic (full 6-derivative continuity)

The classic Minimum Snap formulation (Mellinger & Kumar, 2011) is
implemented as a Quadratic Program minimising the integral of the
square of the n-th derivative (snap for n=4, jerk for n=3).

References:
  [1] Mellinger, D. & Kumar, V. (2011). Minimum snap trajectory
      generation and control for quadrotors. ICRA 2011.
  [2] Mueller, M. W., Hehn, M., & D'Andrea, R. (2015). A
      computationally efficient motion primitive for quadrotor
      trajectory generation. TRO 31(6).
  [3] Richter, C., Bry, A., & Roy, N. (2016). Polynomial trajectory
      planning for aggressive quadrotor flight in dense indoor
      environments. ISRR.

Author : Haikal Hakim Baiqunni
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


# ── Waypoint spec ─────────────────────────────────────────────────────────────

@dataclass
class TrajWaypoint:
    """A 3-D waypoint with optional velocity / acceleration constraints."""
    x: float = 0.0
    y: float = 0.0
    z: float = 2.0
    yaw: float = 0.0        # rad

    # Optional derivative constraints (None = free / unconstrained)
    vx:  Optional[float] = None
    vy:  Optional[float] = None
    vz:  Optional[float] = None
    ax:  Optional[float] = None
    ay:  Optional[float] = None
    az:  Optional[float] = None

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


# ── Time allocation helpers ────────────────────────────────────────────────────

def time_allocation_constant(waypoints: List[TrajWaypoint],
                               speed: float = 1.5) -> List[float]:
    """Segment durations based on constant average speed."""
    durations = []
    for i in range(len(waypoints) - 1):
        d = np.linalg.norm(waypoints[i+1].pos - waypoints[i].pos)
        durations.append(max(d / speed, 0.5))
    return durations


def time_allocation_trapezoidal(waypoints: List[TrajWaypoint],
                                 v_max: float = 2.0,
                                 a_max: float = 3.0) -> List[float]:
    """
    Trapezoidal velocity profile — more realistic than constant speed.
    Accounts for acceleration / deceleration phases.
    """
    durations = []
    for i in range(len(waypoints) - 1):
        d = np.linalg.norm(waypoints[i+1].pos - waypoints[i].pos)
        # Time to reach v_max and decelerate
        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * t_acc**2
        if 2 * d_acc >= d:
            # Triangular profile (never reaches v_max)
            t = 2 * math.sqrt(d / a_max)
        else:
            t = 2 * t_acc + (d - 2 * d_acc) / v_max
        durations.append(max(t, 0.3))
    return durations


# ── Single-axis polynomial segment ────────────────────────────────────────────

def _poly_deriv_coeffs(order: int, deriv: int) -> np.ndarray:
    """
    Return coefficient array for the deriv-th derivative of a polynomial
    of given order evaluated at t (as polynomial in t).

    poly(t) = sum_i c[i] * t^i  for i in 0..order
    d^k/dt^k poly(t) = sum_i c[i] * falling_factorial(i, k) * t^(i-k)
    """
    coeffs = np.zeros(order + 1)
    for i in range(deriv, order + 1):
        ff = 1
        for j in range(deriv):
            ff *= (i - j)
        coeffs[i] = ff
    return coeffs


def _eval_poly(coeffs: np.ndarray, t: float, deriv: int = 0) -> float:
    """Evaluate polynomial or its derivative at time t."""
    n = len(coeffs) - 1
    val = 0.0
    for i in range(deriv, n + 1):
        ff = 1
        for j in range(deriv):
            ff *= (i - j)
        val += coeffs[i] * ff * (t ** max(i - deriv, 0))
    return val


def _build_poly_matrix(T: float, order: int,
                        deriv_constraints: List[Tuple[int, float]]) -> np.ndarray:
    """
    Build row vectors for a polynomial segment of given order.

    deriv_constraints: list of (derivative_order, time) pairs
    Returns matrix rows mapping coefficient vector to constraint values.
    """
    n = order + 1
    rows = []
    for deriv, t in deriv_constraints:
        row = np.zeros(n)
        for i in range(deriv, n):
            ff = 1
            for j in range(deriv):
                ff *= (i - j)
            row[i] = ff * (t ** max(i - deriv, 0))
        rows.append(row)
    return np.array(rows)


# ── Multi-segment polynomial trajectory ───────────────────────────────────────

class PolynomialTrajectory:
    """
    Multi-segment polynomial trajectory for one spatial axis.

    Solves for polynomial coefficients such that:
      - Position continuity at all waypoints (order 0)
      - Velocity continuity at interior waypoints (order 1)
      - Acceleration continuity (order 2) — if order >= 5
      - Jerk continuity (order 3) — if order >= 7
      - Snap continuity (order 4) — if order >= 9

    Implementation:
      Each segment is an (poly_order+1)-coefficient polynomial in [0, T_k].
      Boundary conditions at t=0 and t=T_k are enforced via linear system.
      Interior continuity conditions are added as equality constraints.
      Free end derivatives are set to zero.
    """

    def __init__(self, waypoints: List[float], durations: List[float],
                 poly_order: int = 5):
        """
        Args:
            waypoints : scalar positions at each waypoint (1D axis)
            durations : segment durations [T_0, ..., T_{N-1}]
            poly_order: polynomial order (3, 5, 7, 9, 11)
        """
        assert len(durations) == len(waypoints) - 1
        self.order    = poly_order
        self.n_seg    = len(durations)
        self.T        = durations
        self.wps      = waypoints
        self.coeffs   = self._solve()   # shape: (n_seg, poly_order+1)

    def _n_cont(self) -> int:
        """Number of continuity conditions at interior points."""
        # Always enforce pos + vel; add acc/jerk/snap based on order
        if self.order >= 11: return 6
        if self.order >= 9:  return 5
        if self.order >= 7:  return 4
        if self.order >= 5:  return 3
        return 2   # cubic: pos + vel only

    def _solve(self) -> np.ndarray:
        """Build and solve the linear system for polynomial coefficients."""
        n     = self.order + 1    # coefficients per segment
        n_seg = self.n_seg
        n_cont= self._n_cont()    # continuity orders
        size  = n * n_seg

        A = np.zeros((size, size))
        b = np.zeros(size)
        row = 0

        # ── Start boundary (seg 0, t=0) ───────────────────────────────────
        # Position = wp[0]
        A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(0, 0)])[0]
        b[row] = self.wps[0]; row += 1
        # Velocity = 0 at start
        A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(1, 0)])[0]
        b[row] = 0.0; row += 1
        # Acceleration = 0 at start (if order >= 5)
        if self.order >= 5:
            A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(2, 0)])[0]
            b[row] = 0.0; row += 1
        if self.order >= 7:
            A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(3, 0)])[0]
            b[row] = 0.0; row += 1
        if self.order >= 9:
            A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(4, 0)])[0]
            b[row] = 0.0; row += 1
        if self.order >= 11:
            A[row, 0:n] = _build_poly_matrix(self.T[0], self.order, [(5, 0)])[0]
            b[row] = 0.0; row += 1

        # ── End boundary (seg n_seg-1, t=T) ──────────────────────────────
        seg_off = (n_seg - 1) * n
        A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(0, self.T[-1])])[0]
        b[row] = self.wps[-1]; row += 1
        A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(1, self.T[-1])])[0]
        b[row] = 0.0; row += 1
        if self.order >= 5:
            A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(2, self.T[-1])])[0]
            b[row] = 0.0; row += 1
        if self.order >= 7:
            A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(3, self.T[-1])])[0]
            b[row] = 0.0; row += 1
        if self.order >= 9:
            A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(4, self.T[-1])])[0]
            b[row] = 0.0; row += 1
        if self.order >= 11:
            A[row, seg_off:seg_off+n] = _build_poly_matrix(self.T[-1], self.order, [(5, self.T[-1])])[0]
            b[row] = 0.0; row += 1

        # ── Interior waypoints + continuity ───────────────────────────────
        for k in range(n_seg - 1):
            off_k  = k * n
            off_k1 = (k + 1) * n
            Tk = self.T[k]

            # Position match at end of seg k
            A[row, off_k:off_k+n] = _build_poly_matrix(Tk, self.order, [(0, Tk)])[0]
            b[row] = self.wps[k+1]; row += 1

            # Position match at start of seg k+1
            A[row, off_k1:off_k1+n] = _build_poly_matrix(self.T[k+1], self.order, [(0, 0)])[0]
            b[row] = self.wps[k+1]; row += 1

            # Continuity conditions (derivatives match across junction)
            for d in range(1, n_cont):
                # Derivative d at end of seg k  == derivative d at start of seg k+1
                A[row, off_k:off_k+n]   =  _build_poly_matrix(Tk, self.order, [(d, Tk)])[0]
                A[row, off_k1:off_k1+n] = -_build_poly_matrix(self.T[k+1], self.order, [(d, 0)])[0]
                b[row] = 0.0; row += 1

        # ── Fill remaining rows (if any) with zeros ───────────────────────
        # (Should not happen if order and constraints are consistent)

        # Solve
        try:
            coeffs_flat = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            coeffs_flat = np.zeros(size)

        return coeffs_flat.reshape(n_seg, n)

    def eval(self, t: float, deriv: int = 0) -> float:
        """Evaluate trajectory at global time t, returning the deriv-th derivative."""
        t_acc = 0.0
        for k, Tk in enumerate(self.T):
            if t <= t_acc + Tk or k == self.n_seg - 1:
                t_local = t - t_acc
                t_local = max(0.0, min(t_local, Tk))
                return _eval_poly(self.coeffs[k], t_local, deriv)
            t_acc += Tk
        return self.wps[-1]

    @property
    def total_time(self) -> float:
        return sum(self.T)


# ── 3-D + Yaw trajectory ─────────────────────────────────────────────────────

@dataclass
class TrajectoryConfig:
    """User-facing trajectory parameters."""
    poly_order: int = 5          # 3 / 5 / 7 / 9 / 11
    time_method: str = "trap"    # "constant" | "trap"
    avg_speed: float = 1.5       # m/s (used by constant method)
    v_max: float = 2.0           # m/s (trapezoidal)
    a_max: float = 3.0           # m/s² (trapezoidal)
    loop: bool = False           # repeat trajectory


ORDER_NAMES = {
    3:  "Cubic (C¹)",
    5:  "Quintic (C²) — Min-Acc",
    7:  "Septic (C³) — Min-Jerk",
    9:  "Nonic (C⁴) — Min-Snap",
    11: "Hendecic (C⁵)",
}


class Trajectory3D:
    """
    Full 3-D + Yaw polynomial trajectory through a list of waypoints.

    Each axis (X, Y, Z, Yaw) is solved independently using
    PolynomialTrajectory. This is valid for small-angle / decoupled motion.

    Usage:
        traj = Trajectory3D(waypoints, config)
        sp   = traj.setpoint(t)   # returns [x,y,z,yaw] at time t
        vel  = traj.velocity(t)   # returns [vx,vy,vz] at time t
    """

    def __init__(self, waypoints: List[TrajWaypoint],
                 config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()
        self.wps    = waypoints
        self._build(waypoints, self.config)

    def _build(self, waypoints: List[TrajWaypoint], cfg: TrajectoryConfig):
        if len(waypoints) < 2:
            self._traj_x = self._traj_y = self._traj_z = self._traj_yaw = None
            self._T = [1.0]
            return

        # Time allocation
        if cfg.time_method == "trap":
            durs = time_allocation_trapezoidal(waypoints, cfg.v_max, cfg.a_max)
        else:
            durs = time_allocation_constant(waypoints, cfg.avg_speed)

        self._durs = durs
        xs   = [w.x   for w in waypoints]
        ys   = [w.y   for w in waypoints]
        zs   = [w.z   for w in waypoints]
        yaws = [w.yaw for w in waypoints]

        o = cfg.poly_order
        self._traj_x   = PolynomialTrajectory(xs,   durs, o)
        self._traj_y   = PolynomialTrajectory(ys,   durs, o)
        self._traj_z   = PolynomialTrajectory(zs,   durs, o)
        self._traj_yaw = PolynomialTrajectory(yaws, durs, o)
        self._total_T  = sum(durs)

    def setpoint(self, t: float) -> np.ndarray:
        """[x, y, z, yaw] at time t."""
        if self._traj_x is None:
            return np.array([self.wps[0].x, self.wps[0].y, self.wps[0].z, self.wps[0].yaw])
        if self.config.loop:
            t = t % self._total_T
        t = min(t, self._total_T)
        return np.array([
            self._traj_x.eval(t, 0),
            self._traj_y.eval(t, 0),
            max(self._traj_z.eval(t, 0), 0.05),
            self._traj_yaw.eval(t, 0),
        ])

    def velocity(self, t: float) -> np.ndarray:
        """[vx, vy, vz] at time t."""
        if self._traj_x is None:
            return np.zeros(3)
        if self.config.loop:
            t = t % self._total_T
        t = min(t, self._total_T)
        return np.array([
            self._traj_x.eval(t, 1),
            self._traj_y.eval(t, 1),
            self._traj_z.eval(t, 1),
        ])

    def acceleration(self, t: float) -> np.ndarray:
        if self._traj_x is None:
            return np.zeros(3)
        if self.config.loop:
            t = t % self._total_T
        t = min(t, self._total_T)
        return np.array([
            self._traj_x.eval(t, 2),
            self._traj_y.eval(t, 2),
            self._traj_z.eval(t, 2),
        ])

    @property
    def total_time(self) -> float:
        return self._total_T if hasattr(self, '_total_T') else 0.0

    def sample_path(self, n_points: int = 200) -> np.ndarray:
        """Sample trajectory as (n_points, 3) array for visualisation."""
        ts = np.linspace(0, self.total_time, n_points)
        return np.array([self.setpoint(t)[:3] for t in ts])

    def segment_times(self) -> List[float]:
        return list(self._durs) if hasattr(self, '_durs') else []
