"""
drone_sim/gui/drone_3d.py
==========================
3D Drone Visualizer

Renders a realistic quadrotor 3D model in a matplotlib Axes3D:
  - Central body (box)
  - 4 arms (X-frame configuration)
  - Rotor disks (spinning animation, color-coded by speed)
  - Orientation axes (body frame xyz)
  - Trajectory trail
  - Setpoint marker

Author : Haikal Hakim Baiqunni
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
from typing import List, Optional
from collections import deque


# ── Colour palette (matches app THEME) ────────────────────────────────────
C_BODY    = "#30363D"
C_ARM     = "#8B949E"
C_ROTOR1  = "#58A6FF"   # front-right  (CW)
C_ROTOR2  = "#3FB950"   # front-left   (CCW)
C_ROTOR3  = "#F78166"   # rear-left    (CW)
C_ROTOR4  = "#D2A8FF"   # rear-right   (CCW)
C_TRAIL   = "#58A6FF"
C_SP      = "#3FB950"
C_XAXIS   = "#F85149"
C_YAXIS   = "#3FB950"
C_ZAXIS   = "#58A6FF"

ROTOR_COLORS = [C_ROTOR1, C_ROTOR2, C_ROTOR3, C_ROTOR4]
TRAIL_LEN    = 300        # samples kept in trail


def _rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler → rotation matrix (body → inertial)."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,              ct*cp            ],
    ])


def _make_box_faces(cx, cy, cz, lx, ly, lz):
    """Return list of (4,3) vertex arrays for a box, centred at (cx,cy,cz)."""
    dx, dy, dz = lx/2, ly/2, lz/2
    v = np.array([
        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
        [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz],
    ])
    faces = [
        [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
        [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]],
    ]
    return faces


def _rotor_disk_points(center_body: np.ndarray, radius: float,
                        n: int = 24) -> np.ndarray:
    """Circle of points in XY-body plane at given body-frame center."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.zeros((n, 3))
    pts[:, 0] = center_body[0] + radius * np.cos(angles)
    pts[:, 1] = center_body[1] + radius * np.sin(angles)
    pts[:, 2] = center_body[2]
    return pts


class DroneVisualizer3D:
    """
    Manages a 3D matplotlib subplot showing an animated quadrotor.

    Usage:
        vis = DroneVisualizer3D(ax)
        vis.update(state_12dof, rotor_speeds, setpoint_xyz)
    """

    # Geometry (metres, body frame)
    BODY_LX   = 0.12
    BODY_LY   = 0.12
    BODY_LZ   = 0.04
    ARM_LEN   = 0.23          # motor-to-centre distance
    ROTOR_R   = 0.09          # rotor disk radius
    AXIS_LEN  = 0.25          # body-frame axis arrow length

    # X-frame arm directions (body frame, unit vectors to each motor)
    #   1=FR(+x,+y)  2=FL(-x,+y)  3=RL(-x,-y)  4=RR(+x,-y)
    _ARM_DIRS = np.array([
        [ 1,  1, 0],
        [-1,  1, 0],
        [-1, -1, 0],
        [ 1, -1, 0],
    ], dtype=float) / np.sqrt(2)

    def __init__(self, ax: Axes3D):
        self.ax = ax
        self._trail_x: deque = deque(maxlen=TRAIL_LEN)
        self._trail_y: deque = deque(maxlen=TRAIL_LEN)
        self._trail_z: deque = deque(maxlen=TRAIL_LEN)
        self._angle_offsets = np.zeros(4)   # rotor spin phase
        self._artists = []
        self._initialized = False
        self.scale = 3.0                 # visual size multiplier (real quad is tiny on a 12 m axis)

        # Pre-build persistent line artists for trail
        self._trail_line,   = ax.plot([], [], [], '-',
                                       color=C_TRAIL, lw=1.2, alpha=0.5,
                                       zorder=1)
        self._sp_marker,    = ax.plot([], [], [], '*',
                                       color=C_SP, ms=12, zorder=5)
        self._sp_vert,      = ax.plot([], [], [], '--',
                                       color=C_SP, lw=0.8, alpha=0.4)

    # ── Public API ─────────────────────────────────────────────────────────

    def reset_trail(self):
        self._trail_x.clear()
        self._trail_y.clear()
        self._trail_z.clear()
        # Also clear the persistent line artist data so the green line disappears immediately
        self._trail_line.set_data_3d([], [], [])
        self._sp_marker.set_data_3d([], [], [])
        self._sp_vert.set_data_3d([], [], [])

    def hide(self):
        """Remove the drone model, trail and setpoint marker from the axes."""
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()
        self.reset_trail()

    def update(self, state: np.ndarray, rotor_speeds: np.ndarray,
               setpoint: np.ndarray, dt: float = 0.08):
        """Redraw drone at current state."""
        x, y, z       = state[0], state[1], state[2]
        phi, theta, psi = state[6], state[7], state[8]
        R = _rotation_matrix(phi, theta, psi)

        # Advance rotor spin angles (visual only)
        spin_dirs = np.array([-1, 1, -1, 1])   # CW=-1, CCW=+1
        self._angle_offsets += spin_dirs * rotor_speeds * dt * 0.3
        self._angle_offsets %= (2 * np.pi)

        # Update trail
        self._trail_x.append(x)
        self._trail_y.append(y)
        self._trail_z.append(max(z, 0.0))

        # Remove old transient artists
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()

        pos = np.array([x, y, z])

        # ── Body box ─────────────────────────────────────────────────────
        self._draw_body(pos, R)

        # ── Arms + rotors ─────────────────────────────────────────────────
        motor_positions = []
        for i, d in enumerate(self._ARM_DIRS):
            motor_body = d * (self.ARM_LEN*self.scale)
            motor_world = pos + R @ motor_body
            motor_positions.append(motor_world)
            self._draw_arm(pos, motor_world)
            self._draw_rotor(motor_world, R, i, rotor_speeds[i])

        # ── Body-frame axes ───────────────────────────────────────────────
        self._draw_body_axes(pos, R)

        # ── Trail ─────────────────────────────────────────────────────────
        tx = list(self._trail_x)
        ty = list(self._trail_y)
        tz = list(self._trail_z)
        self._trail_line.set_data_3d(tx, ty, tz)

        # ── Setpoint marker ───────────────────────────────────────────────
        sx, sy, sz = setpoint[0], setpoint[1], max(setpoint[2], 0.0)
        self._sp_marker.set_data_3d([sx], [sy], [sz])
        self._sp_vert.set_data_3d([sx, sx], [sy, sy], [0.0, sz])

    # ── Private draw helpers ───────────────────────────────────────────────

    def _draw_body(self, pos: np.ndarray, R: np.ndarray):
        """Draw rotated body box."""
        dx, dy, dz = (self.BODY_LX*self.scale)/2, (self.BODY_LY*self.scale)/2, (self.BODY_LZ*self.scale)/2
        # 8 corners in body frame
        corners_b = np.array([
            [-dx,-dy,-dz], [+dx,-dy,-dz], [+dx,+dy,-dz], [-dx,+dy,-dz],
            [-dx,-dy,+dz], [+dx,-dy,+dz], [+dx,+dy,+dz], [-dx,+dy,+dz],
        ])
        corners_w = (R @ corners_b.T).T + pos

        face_idx = [
            [0,1,2,3], [4,5,6,7],
            [0,1,5,4], [2,3,7,6],
            [0,3,7,4], [1,2,6,5],
        ]
        verts = [[corners_w[i] for i in fi] for fi in face_idx]
        poly = Poly3DCollection(verts, alpha=0.75,
                                 facecolor=C_BODY, edgecolor=C_ARM,
                                 linewidth=0.5, zorder=3)
        self.ax.add_collection3d(poly)
        self._artists.append(poly)

    def _draw_arm(self, p_center: np.ndarray, p_motor: np.ndarray):
        """Draw a single arm as a thick 3D line segment."""
        xs = [p_center[0], p_motor[0]]
        ys = [p_center[1], p_motor[1]]
        zs = [p_center[2], p_motor[2]]
        ln, = self.ax.plot(xs, ys, zs, '-', color=C_ARM,
                            lw=3.0, solid_capstyle='round', zorder=2)
        self._artists.append(ln)

    def _draw_rotor(self, motor_pos: np.ndarray, R: np.ndarray,
                     rotor_idx: int, omega: float):
        """Draw rotor disk (filled polygon) and hub dot."""
        color = ROTOR_COLORS[rotor_idx]
        n_pts = 20
        angle0 = self._angle_offsets[rotor_idx]
        angles = np.linspace(angle0, angle0 + 2*np.pi, n_pts, endpoint=False)

        # Disk points in body plane, rotated with drone
        pts_b = np.zeros((n_pts, 3))
        pts_b[:, 0] = (self.ROTOR_R*self.scale) * np.cos(angles)
        pts_b[:, 1] = (self.ROTOR_R*self.scale) * np.sin(angles)
        pts_b[:, 2] = 0.0
        pts_w = (R @ pts_b.T).T + motor_pos

        # Opacity scales with rotor speed
        alpha = np.clip(omega / 600.0, 0.15, 0.55)

        verts = [list(pts_w)]
        poly = Poly3DCollection(verts, alpha=alpha,
                                 facecolor=color, edgecolor=color,
                                 linewidth=0.8, zorder=4)
        self.ax.add_collection3d(poly)
        self._artists.append(poly)

        # Blade lines (2 blades per rotor for visual detail)
        for k in range(2):
            a = angle0 + k * np.pi
            p1_b = np.array([(self.ROTOR_R*self.scale)*np.cos(a),
                              (self.ROTOR_R*self.scale)*np.sin(a), 0])
            p2_b = -p1_b
            p1_w = R @ p1_b + motor_pos
            p2_w = R @ p2_b + motor_pos
            ln, = self.ax.plot([p1_w[0], p2_w[0]],
                                [p1_w[1], p2_w[1]],
                                [p1_w[2], p2_w[2]],
                                '-', color=color, lw=2.0,
                                alpha=0.85, zorder=5)
            self._artists.append(ln)

        # Motor hub dot
        hub, = self.ax.plot([motor_pos[0]], [motor_pos[1]], [motor_pos[2]],
                             'o', color=color, ms=5, zorder=6)
        self._artists.append(hub)

    def _draw_body_axes(self, pos: np.ndarray, R: np.ndarray):
        """Draw body-frame X/Y/Z axes as coloured arrows."""
        axes_def = [
            (np.array([1,0,0]), C_XAXIS, "x"),
            (np.array([0,1,0]), C_YAXIS, "y"),
            (np.array([0,0,1]), C_ZAXIS, "z"),
        ]
        for unit_b, color, label in axes_def:
            tip = pos + R @ (unit_b * (self.AXIS_LEN*self.scale))
            ln, = self.ax.plot([pos[0], tip[0]],
                                [pos[1], tip[1]],
                                [pos[2], tip[2]],
                                '-', color=color, lw=2.0, alpha=0.9, zorder=7)
            self._artists.append(ln)


# ── Multi-drone renderer ───────────────────────────────────────────────────

class SwarmVisualizer3D:
    """
    Manages multiple DroneVisualizer3D instances for swarm display.
    Each drone gets a unique colour tint.
    """

    SWARM_COLORS = [
        "#58A6FF", "#3FB950", "#F78166", "#D2A8FF",
        "#F0883E", "#79C0FF", "#56D364", "#FFA657",
    ]

    def __init__(self, ax: Axes3D, n_drones: int):
        self.ax = ax
        self.n = n_drones
        self._visualizers: List[DroneVisualizer3D] = []
        for i in range(n_drones):
            vis = DroneVisualizer3D(ax)
            # Recolour trail per drone
            vis._trail_line.set_color(self.SWARM_COLORS[i % len(self.SWARM_COLORS)])
            self._visualizers.append(vis)

    def update(self, states: List[np.ndarray], rotor_speeds_list: List[np.ndarray],
               setpoints: List[np.ndarray], dt: float = 0.08):
        for i in range(self.n):
            self._visualizers[i].update(
                states[i], rotor_speeds_list[i], setpoints[i], dt
            )

    def hide(self):
        for v in self._visualizers:
            v.hide()

    def reset_trails(self):
        for v in self._visualizers:
            v.reset_trail()
