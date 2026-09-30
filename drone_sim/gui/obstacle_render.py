"""
drone_sim/gui/obstacle_render.py
=================================
Draws obstacles (boxes / cylinders) on a matplotlib Axes3D.

Artists are tagged so only obstacle artists are removed when redrawing.
"""

import numpy as np
from typing import List
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..core.obstacles import BoxObstacle, CylinderObstacle


class Obstacle3DRenderer:
    """
    Renders obstacle geometry on a matplotlib Axes3D.

    Strategy: rebuild artists from scratch every time using ax.cla()-safe
    approach — we keep a reference to the axes and the current obstacle list,
    and draw fresh artists each call without trying to .remove() old ones
    (Poly3DCollection.remove() is unreliable in matplotlib 3D).

    Instead we track artists via ax.collections and ax.lines with a custom
    tag stored in each artist's label, and remove only the tagged ones.
    """

    _TAG = "__dronsim_obs__"

    def __init__(self):
        self._ax = None

    def clear(self, ax):
        """Remove all previously drawn obstacle artists (tagged ones only)."""
        # Remove tagged collections (Poly3DCollection for boxes + cylinder caps)
        to_remove = [c for c in list(ax.collections)
                     if getattr(c, '_label', '') == self._TAG]
        for c in to_remove:
            try:
                c.remove()
            except Exception:
                pass

        # Remove tagged lines (cylinder edges, box edges)
        to_remove_lines = [l for l in list(ax.lines)
                           if getattr(l, '_label', '') == self._TAG]
        for l in to_remove_lines:
            try:
                l.remove()
            except Exception:
                pass

    def draw(self, ax, obstacles: List):
        """Clear old obstacles and draw fresh ones."""
        self.clear(ax)
        self._ax = ax
        for obs in obstacles:
            if isinstance(obs, BoxObstacle):
                self._draw_box(ax, obs)
            elif isinstance(obs, CylinderObstacle):
                self._draw_cylinder(ax, obs)

    def _tag(self, artist):
        """Mark an artist as belonging to this renderer.
        MUST be called AFTER ax.add_collection3d() or ax.plot(),
        because matplotlib overwrites _label during add_collection3d.
        """
        artist._label = self._TAG
        return artist

    def _draw_box(self, ax, obs: BoxObstacle):
        cx, cy, cz = obs.cx, obs.cy, obs.cz
        lx, ly, lz = obs.lx/2, obs.ly/2, obs.lz/2
        v = np.array([
            [cx-lx, cy-ly, cz-lz], [cx+lx, cy-ly, cz-lz],
            [cx+lx, cy+ly, cz-lz], [cx-lx, cy+ly, cz-lz],
            [cx-lx, cy-ly, cz+lz], [cx+lx, cy-ly, cz+lz],
            [cx+lx, cy+ly, cz+lz], [cx-lx, cy+ly, cz+lz],
        ])
        face_idx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        verts = [[v[i] for i in f] for f in face_idx]
        poly = Poly3DCollection(verts, alpha=0.30,
                                 facecolor=obs.color,
                                 edgecolor=obs.color,
                                 linewidth=0.8)
        ax.add_collection3d(poly)
        self._tag(poly)   # tag AFTER add

    def _draw_cylinder(self, ax, obs: CylinderObstacle):
        theta = np.linspace(0, 2*np.pi, 24)

        # Top and bottom cap rings
        for z in [obs.z_bot, obs.z_top]:
            xs = obs.cx + obs.radius * np.cos(theta)
            ys = obs.cy + obs.radius * np.sin(theta)
            zs = np.full_like(xs, z)
            ln, = ax.plot(xs, ys, zs, color=obs.color, lw=2.0, alpha=0.9)
            self._tag(ln)   # tag AFTER plot

        # Vertical edges
        for t in theta[::6]:
            xs = [obs.cx + obs.radius*np.cos(t)] * 2
            ys = [obs.cy + obs.radius*np.sin(t)] * 2
            zs = [obs.z_bot, obs.z_top]
            ln, = ax.plot(xs, ys, zs, color=obs.color, lw=1.2, alpha=0.7)
            self._tag(ln)

        # Filled caps
        for z in [obs.z_bot, obs.z_top]:
            verts = [[(obs.cx + obs.radius*np.cos(t),
                       obs.cy + obs.radius*np.sin(t), z)
                      for t in theta]]
            poly = Poly3DCollection(verts, alpha=0.20,
                                     facecolor=obs.color, edgecolor='none')
            ax.add_collection3d(poly)
            self._tag(poly)   # tag AFTER add

        # Filled side surface
        n = len(theta) - 1
        for i in range(n):
            t0, t1 = theta[i], theta[i+1]
            x0 = obs.cx + obs.radius*np.cos(t0); y0 = obs.cy + obs.radius*np.sin(t0)
            x1 = obs.cx + obs.radius*np.cos(t1); y1 = obs.cy + obs.radius*np.sin(t1)
            quad = [[(x0,y0,obs.z_bot),(x1,y1,obs.z_bot),
                     (x1,y1,obs.z_top),(x0,y0,obs.z_top)]]
            poly = Poly3DCollection(quad, alpha=0.15,
                                     facecolor=obs.color, edgecolor='none')
            ax.add_collection3d(poly)
            self._tag(poly)   # tag AFTER add
