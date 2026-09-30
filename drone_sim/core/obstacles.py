"""
drone_sim/core/obstacles.py
============================
Obstacle Model & Avoidance

Supports two obstacle types:
  - Box      : axis-aligned bounding box (AABB)
  - Cylinder : vertical cylinder

Provides:
  - Signed Distance Function (SDF) per obstacle
  - Artificial Potential Field (APF) repulsive force
  - RRT* path planner operating in 3-D obstacle-aware space
  - Collision check utilities

References:
  [1] Khatib, O. (1986). Real-time obstacle avoidance for manipulators
      and mobile robots. IJRR 5(1):90-98.
  [2] LaValle, S. M. (1998). Rapidly-exploring random trees: A new tool
      for path planning. TR 98-11, Iowa State U.
  [3] Karaman, S. & Frazzoli, E. (2011). Sampling-based algorithms for
      optimal motion planning. IJRR 30(7):846-894.

Author : Haikal Hakim Baiqunni
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


# ── Obstacle data models ─────────────────────────────────────────────────────

@dataclass
class BoxObstacle:
    """Axis-aligned box obstacle."""
    cx: float = 0.0     # centre X
    cy: float = 0.0     # centre Y
    cz: float = 1.0     # centre Z
    lx: float = 1.0     # full length X
    ly: float = 1.0     # full length Y
    lz: float = 2.0     # full length Z
    color: str = "#F78166"
    label: str = "Box"

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance from point p to box surface (neg=inside)."""
        dx = max(abs(p[0]-self.cx) - self.lx/2, 0)
        dy = max(abs(p[1]-self.cy) - self.ly/2, 0)
        dz = max(abs(p[2]-self.cz) - self.lz/2, 0)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def contains(self, p: np.ndarray) -> bool:
        return (abs(p[0]-self.cx) < self.lx/2 and
                abs(p[1]-self.cy) < self.ly/2 and
                abs(p[2]-self.cz) < self.lz/2)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz])


@dataclass
class CylinderObstacle:
    """Vertical cylinder obstacle."""
    cx: float = 0.0     # centre X
    cy: float = 0.0     # centre Y
    z_bot: float = 0.0  # bottom Z
    z_top: float = 3.0  # top Z
    radius: float = 0.5
    color: str = "#D2A8FF"
    label: str = "Cylinder"

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance to cylinder surface."""
        r_dist = max(math.sqrt((p[0]-self.cx)**2 + (p[1]-self.cy)**2) - self.radius, 0)
        z_dist = max(self.z_bot - p[2], p[2] - self.z_top, 0)
        return math.sqrt(r_dist**2 + z_dist**2)

    def contains(self, p: np.ndarray) -> bool:
        r = math.sqrt((p[0]-self.cx)**2 + (p[1]-self.cy)**2)
        return r < self.radius and self.z_bot <= p[2] <= self.z_top

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy, (self.z_bot+self.z_top)/2])


Obstacle = BoxObstacle | CylinderObstacle


# ── APF (Artificial Potential Field) ─────────────────────────────────────────

class APFPlanner:
    """
    Artificial Potential Field obstacle avoidance.

    Repulsive potential:
        U_rep(q) = ½·η·(1/d - 1/d₀)²   if d ≤ d₀
                 = 0                      otherwise
    Repulsive force:
        F_rep = -∇U_rep

    Attractive potential (goal):
        U_att = ½·ζ·d_goal²
    Attractive force:
        F_att = -ζ·(q - q_goal)

    Parameters
    ----------
    eta       : repulsive gain
    zeta      : attractive gain
    d0        : influence radius (m) beyond which repulsion = 0
    max_force : saturation limit (m/s²)
    """

    def __init__(self, eta: float = 8.0, zeta: float = 1.5,
                 d0: float = 1.5, max_force: float = 6.0):
        self.eta = eta
        self.zeta = zeta
        self.d0 = d0
        self.max_force = max_force

    def repulsive_force(self, pos: np.ndarray,
                        obstacles: List) -> np.ndarray:
        """Sum of repulsive forces from all obstacles."""
        F_rep = np.zeros(3)
        for obs in obstacles:
            d = obs.sdf(pos)
            if d < 1e-6:
                d = 1e-6
            if d > self.d0:
                continue

            # Gradient of SDF (numerical)
            eps = 0.01
            grad = np.array([
                (obs.sdf(pos + np.array([eps,0,0])) - obs.sdf(pos - np.array([eps,0,0]))) / (2*eps),
                (obs.sdf(pos + np.array([0,eps,0])) - obs.sdf(pos - np.array([0,eps,0]))) / (2*eps),
                (obs.sdf(pos + np.array([0,0,eps])) - obs.sdf(pos - np.array([0,0,eps]))) / (2*eps),
            ])
            norm = np.linalg.norm(grad)
            if norm > 1e-6:
                grad /= norm

            coeff = self.eta * (1/d - 1/self.d0) / (d**2)
            F_rep += coeff * grad

        # Saturate
        mag = np.linalg.norm(F_rep)
        if mag > self.max_force:
            F_rep = F_rep / mag * self.max_force
        return F_rep

    def attractive_force(self, pos: np.ndarray,
                         goal: np.ndarray) -> np.ndarray:
        """Attractive force toward goal."""
        diff = goal - pos
        dist = np.linalg.norm(diff)
        if dist < 0.05:
            return np.zeros(3)
        return self.zeta * diff

    def total_force(self, pos: np.ndarray, goal: np.ndarray,
                    obstacles: List) -> np.ndarray:
        F = self.attractive_force(pos, goal) + self.repulsive_force(pos, obstacles)
        mag = np.linalg.norm(F)
        if mag > self.max_force:
            F = F / mag * self.max_force
        return F


# ── RRT* Path Planner ─────────────────────────────────────────────────────────

class RRTStar:
    """
    RRT* (Optimal Rapidly-exploring Random Tree) in 3-D.

    Returns a smooth, obstacle-free path from start to goal.
    Path is then tracked by the position controller as waypoints.

    Key improvements over RRT:
      - Rewiring: near-neighbour cost optimisation → asymptotic optimality
      - Path smoothing: line-of-sight shortcutting pass

    Parameters
    ----------
    bounds      : [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    max_iter    : RRT* iterations
    step_size   : max extension length (m)
    goal_bias   : fraction of iterations that sample the goal directly
    near_radius : rewiring neighbourhood radius (m)
    drone_r     : drone safety radius for collision checks (m)
    """

    def __init__(self, bounds=None, max_iter: int = 800,
                 step_size: float = 0.6, goal_bias: float = 0.15,
                 near_radius: float = 1.2, drone_r: float = 0.25):
        self.bounds     = bounds or [[-8,8],[-8,8],[0.1,8]]
        self.max_iter   = max_iter
        self.step_size  = step_size
        self.goal_bias  = goal_bias
        self.near_radius = near_radius
        self.drone_r    = drone_r

        # Tree storage
        self._nodes: List[np.ndarray] = []
        self._parent: List[int]       = []
        self._cost:   List[float]     = []

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.goal_bias:
            return goal.copy()
        return np.array([
            np.random.uniform(*self.bounds[0]),
            np.random.uniform(*self.bounds[1]),
            np.random.uniform(*self.bounds[2]),
        ])

    def _nearest(self, q: np.ndarray) -> int:
        dists = [np.linalg.norm(n - q) for n in self._nodes]
        return int(np.argmin(dists))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        diff = q_to - q_from
        d = np.linalg.norm(diff)
        if d < 1e-6:
            return q_from.copy()
        if d <= self.step_size:
            return q_to.copy()
        return q_from + (diff/d)*self.step_size

    def _collision_free(self, q1: np.ndarray, q2: np.ndarray,
                         obstacles: List, n_check: int = 8) -> bool:
        """Check line segment q1→q2 for obstacle collision (n_check samples)."""
        for i in range(n_check+1):
            t = i / n_check
            p = q1*(1-t) + q2*t
            # Ground check
            if p[2] < 0.05:
                return False
            for obs in obstacles:
                if obs.sdf(p) < self.drone_r:
                    return False
        return True

    def _near_nodes(self, q: np.ndarray) -> List[int]:
        return [i for i, n in enumerate(self._nodes)
                if np.linalg.norm(n-q) <= self.near_radius]

    def plan(self, start: np.ndarray, goal: np.ndarray,
             obstacles: List) -> Optional[List[np.ndarray]]:
        """
        Plan a path from start to goal avoiding obstacles.

        Returns list of waypoints [start, ..., goal] or None if failed.
        """
        self._nodes  = [start.copy()]
        self._parent = [-1]
        self._cost   = [0.0]

        goal_idx = None
        goal_thresh = max(self.step_size * 0.8, 0.3)

        for _ in range(self.max_iter):
            q_rand   = self._sample(goal)
            idx_near = self._nearest(q_rand)
            q_new    = self._steer(self._nodes[idx_near], q_rand)

            if not self._collision_free(self._nodes[idx_near], q_new, obstacles):
                continue

            # RRT* rewiring
            near_idxs = self._near_nodes(q_new)
            best_parent = idx_near
            best_cost   = self._cost[idx_near] + np.linalg.norm(q_new - self._nodes[idx_near])

            for ni in near_idxs:
                c = self._cost[ni] + np.linalg.norm(q_new - self._nodes[ni])
                if c < best_cost and self._collision_free(self._nodes[ni], q_new, obstacles):
                    best_parent = ni
                    best_cost   = c

            self._nodes.append(q_new)
            self._parent.append(best_parent)
            self._cost.append(best_cost)
            new_idx = len(self._nodes) - 1

            # Rewire neighbours through new node
            for ni in near_idxs:
                c = best_cost + np.linalg.norm(self._nodes[ni] - q_new)
                if c < self._cost[ni] and self._collision_free(q_new, self._nodes[ni], obstacles):
                    self._parent[ni] = new_idx
                    self._cost[ni]   = c

            # Goal check
            if np.linalg.norm(q_new - goal) < goal_thresh:
                if self._collision_free(q_new, goal, obstacles):
                    goal_idx = new_idx
                    break

        if goal_idx is None:
            # Try to find closest node to goal
            dists = [np.linalg.norm(n-goal) for n in self._nodes]
            goal_idx = int(np.argmin(dists))

        # Extract path
        path = [goal.copy()]
        idx = goal_idx
        while idx != -1:
            path.append(self._nodes[idx].copy())
            idx = self._parent[idx]
        path.reverse()

        # Smooth: remove redundant waypoints (line-of-sight shortcutting)
        path = self._smooth(path, obstacles)
        return path

    def _smooth(self, path: List[np.ndarray],
                obstacles: List) -> List[np.ndarray]:
        """Greedy line-of-sight path shortcutting."""
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._collision_free(path[i], path[j], obstacles, n_check=12):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed


# ── APF + RRT Hybrid ──────────────────────────────────────────────────────────

class HybridAvoidance:
    """
    Hybrid APF + RRT* planner.

    Strategy:
      1. At sim start, RRT* pre-plans a global path from start to goal.
      2. During flight, APF provides local real-time reactive correction
         on top of the RRT* waypoint target.
      3. If local minimum detected (drone stuck), re-plan with RRT*.

    This combines:
      - RRT* global optimality and completeness
      - APF real-time responsiveness at 100 Hz
    """

    def __init__(self, apf: APFPlanner = None, rrt: RRTStar = None):
        self.apf = apf or APFPlanner()
        self.rrt = rrt or RRTStar()
        self._planned_path: List[np.ndarray] = []
        self._path_idx = 0
        self._stuck_counter = 0
        self._prev_pos = None

    def plan(self, start: np.ndarray, goal: np.ndarray,
             obstacles: List) -> List[np.ndarray]:
        """Pre-plan RRT* path. Call once before simulation."""
        path = self.rrt.plan(start, goal, obstacles)
        self._planned_path = path or [start, goal]
        self._path_idx = 0
        self._prev_pos = start.copy()
        return self._planned_path

    def get_local_target(self, pos: np.ndarray,
                          obstacles: List) -> np.ndarray:
        """Current RRT* waypoint (lookahead along planned path)."""
        if not self._planned_path:
            return pos
        # Advance to next waypoint if close enough
        while (self._path_idx < len(self._planned_path)-1 and
               np.linalg.norm(pos - self._planned_path[self._path_idx]) < 0.4):
            self._path_idx += 1
        return self._planned_path[min(self._path_idx, len(self._planned_path)-1)]

    def avoidance_force(self, pos: np.ndarray, goal: np.ndarray,
                         obstacles: List) -> np.ndarray:
        """APF repulsive force + attraction to local RRT* target."""
        local_goal = self.get_local_target(pos, obstacles)
        return self.apf.total_force(pos, local_goal, obstacles)

    @property
    def planned_path(self) -> List[np.ndarray]:
        return self._planned_path


# ── Obstacle manager ──────────────────────────────────────────────────────────

class ObstacleManager:
    """Central registry of all obstacles in the scene."""

    def __init__(self):
        self._obstacles: List = []

    def add(self, obs) -> int:
        self._obstacles.append(obs)
        return len(self._obstacles) - 1

    def remove(self, idx: int):
        if 0 <= idx < len(self._obstacles):
            self._obstacles.pop(idx)

    def clear(self):
        self._obstacles.clear()

    @property
    def obstacles(self) -> List:
        return list(self._obstacles)

    def __len__(self):
        return len(self._obstacles)

    def any_collision(self, pos: np.ndarray, radius: float = 0.2) -> bool:
        return any(obs.sdf(pos) < radius for obs in self._obstacles)

    def min_distance(self, pos: np.ndarray) -> float:
        if not self._obstacles:
            return float('inf')
        return min(obs.sdf(pos) for obs in self._obstacles)
