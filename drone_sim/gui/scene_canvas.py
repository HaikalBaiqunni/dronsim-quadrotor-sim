"""
drone_sim/gui/scene_canvas.py
==============================
Top-down 2D scene: place waypoints and obstacles, see the planned route and
(while flying) the live drone position and trail.

Tools live in a vertical strip on the left so they never overflow the canvas.
"""

import math
import tkinter as tk
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..core.mission import Waypoint
from ..core.obstacles import BoxObstacle, CylinderObstacle, ObstacleManager
from .theme import C, F, SERIES
from .widgets import Btn, Chip, WrapLabel, caption, combo, divider

BOX_COLORS = ["#FF9E7A", "#F5B942", "#B392F0", "#4C9AFF"]
CYL_COLORS = ["#B392F0", "#5CD6D6", "#3DDC97", "#FF7AB6"]
CANVAS_BG = "#0A0D12"


class SceneCanvas(tk.Frame):
    TOOLS = {
        "wp_add":   ("Add", "Click to add a waypoint. Drag an existing one to move it.", "wp"),
        "wp_move":  ("Move", "Drag a waypoint to reposition it.", "wp"),
        "wp_del":   ("Delete", "Click a waypoint to delete it.", "wp"),
        "obs_box":  ("Box", "Click to place a box obstacle. Drag placed obstacles to move.", "obs"),
        "obs_cyl":  ("Cylinder", "Click to place a cylinder. Drag placed obstacles to move.", "obs"),
        "obs_move": ("Move", "Drag an obstacle to reposition it.", "obs"),
        "obs_del":  ("Delete", "Click an obstacle to delete it. Right-click also deletes.", "obs"),
    }

    def __init__(self, parent, obs_manager: ObstacleManager,
                 on_waypoints_changed: Optional[Callable] = None,
                 on_obstacles_changed: Optional[Callable] = None):
        super().__init__(parent, bg=C["surface"])
        self._obs = obs_manager
        self._on_wp = on_waypoints_changed
        self._on_obs = on_obstacles_changed
        self._waypoints: List[Waypoint] = []
        self._tool = tk.StringVar(value="wp_add")
        self._drag: Optional[Tuple[str, int]] = None
        self._rrt_paths: List[np.ndarray] = []
        self._poly_path: Optional[np.ndarray] = None
        self._extra_paths: List[Tuple[str, List[Tuple[float, float]]]] = []
        self._live: List[Tuple[float, float, float]] = []       # x, y, yaw per drone
        self._trails: List[np.ndarray] = []
        self._read_only = False

        self.z_var = tk.StringVar(value="2.0")
        self.yaw_var = tk.StringVar(value="0")
        self.h_var = tk.StringVar(value="3.0")
        self.size_var = tk.StringVar(value="1.0")
        self.range_var = tk.StringVar(value="8")
        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        strip = tk.Frame(self, bg=C["surface"], width=118)
        strip.pack(side="left", fill="y", padx=(10, 6), pady=10)
        strip.pack_propagate(False)

        caption(strip, "Waypoints", pady=(0, 4))
        self._btns = {}
        for t in ("wp_add", "wp_move", "wp_del"):
            self._tool_btn(strip, t)
        caption(strip, "Obstacles", pady=(14, 4))
        for t in ("obs_box", "obs_cyl", "obs_move", "obs_del"):
            self._tool_btn(strip, t)

        right = tk.Frame(self, bg=C["surface"])
        right.pack(side="left", fill="both", expand=True, pady=10, padx=(0, 10))
        self._hint = WrapLabel(right, "", fg=C["text2"], bg=C["surface"], font=F["small"])
        self._hint.pack(fill="x", pady=(0, 4))
        self._param_frame = tk.Frame(right, bg=C["surface"])
        self._param_frame.pack(fill="x", pady=(0, 6))
        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=1,
                                highlightbackground=C["border"], cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self._coord = tk.Label(right, text="x 0.00 m    y 0.00 m", bg=C["surface"], fg=C["text3"],
                               font=F["mono_small"], anchor="w")
        self._coord.pack(fill="x", pady=(6, 0))
        foot = tk.Frame(right, bg=C["surface"])
        foot.pack(fill="x", pady=(4, 0))
        tk.Label(foot, text="Range (m)", bg=C["surface"], fg=C["text2"], font=F["small"]).pack(side="left")
        combo(foot, ["4", "6", "8", "10", "15", "20"], "8",
              lambda v: (self.range_var.set(v), self._redraw()), width=3).pack(side="left", padx=(6, 0))
        Btn(foot, "Clear obstacles", self._clear_obs, kind="ghost", padx=8, pady=4,
            font=F["small_bold"]).pack(side="right", padx=(4, 0))
        Btn(foot, "Clear waypoints", self._clear_wps, kind="ghost", padx=8, pady=4,
            font=F["small_bold"]).pack(side="right", padx=4)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self._set_tool("wp_add")

    def _tool_btn(self, parent, tool):
        label = self.TOOLS[tool][0]
        b = tk.Button(parent, text=label, relief="flat", bd=0, font=F["bold"], anchor="w",
                      padx=12, pady=6, cursor="hand2", highlightthickness=0,
                      command=lambda t=tool: self._set_tool(t))
        b.pack(fill="x", pady=1)
        self._btns[tool] = b

    def _set_tool(self, tool: str):
        self._tool.set(tool)
        for t, b in self._btns.items():
            on = t == tool
            col = C["accent"] if self.TOOLS[t][2] == "wp" else C["coral"]
            b.configure(bg=col if on else C["surface3"], fg=C["accent_fg"] if on else C["text2"],
                        activebackground=col if on else "#313B49",
                        activeforeground=C["accent_fg"] if on else C["text"])
        self._hint.configure(text=self.TOOLS[tool][1])
        for w in self._param_frame.winfo_children():
            w.destroy()
        if tool in ("wp_add", "wp_move"):
            self._param_row("New waypoint height (m)", self.z_var)
            self._param_row("Yaw (deg)", self.yaw_var)
        elif tool in ("obs_box", "obs_cyl"):
            self._param_row("Obstacle height (m)", self.h_var)
            self._param_row("Width (m)" if tool == "obs_box" else "Diameter (m)", self.size_var)
        self._redraw()

    def _param_row(self, label, var):
        f = tk.Frame(self._param_frame, bg=C["surface"])
        f.pack(side="left", padx=(0, 14))
        tk.Label(f, text=label, bg=C["surface"], fg=C["text2"], font=F["small"]).pack(side="left", padx=(0, 6))
        tk.Entry(f, textvariable=var, width=5, justify="right", bg=C["surface3"], fg=C["accent"],
                 insertbackground=C["text"], relief="flat", font=F["mono"],
                 highlightthickness=1, highlightbackground=C["border"],
                 highlightcolor=C["accent"]).pack(side="left", ipady=3)

    # ── Coordinates ──────────────────────────────────────────────────────────

    def _range(self) -> float:
        try:
            return max(1.0, float(self.range_var.get()))
        except ValueError:
            return 8.0

    def _size(self):
        return max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)

    def _scale(self) -> float:
        w, h = self._size()
        return min(w, h) / (2 * self._range())      # px per metre (square cells)

    def _w2c(self, wx, wy):
        w, h = self._size()
        s = self._scale()
        return w / 2 + wx * s, h / 2 - wy * s

    def _c2w(self, cx, cy):
        w, h = self._size()
        s = self._scale()
        return round((cx - w / 2) / s, 2), round((h / 2 - cy) / s, 2)

    def _m2px(self, m):
        return m * self._scale()

    def _float(self, var, default):
        try:
            return float(var.get())
        except ValueError:
            return default

    def _hit_wp(self, cx, cy, r=14):
        for i, wp in enumerate(self._waypoints):
            px, py = self._w2c(wp.x, wp.y)
            if abs(cx - px) <= r and abs(cy - py) <= r:
                return i
        return None

    def _hit_obs(self, cx, cy):
        for i, o in enumerate(self._obs.obstacles):
            px, py = self._w2c(o.cx, o.cy)
            r = max(self._m2px(o.lx / 2 if isinstance(o, BoxObstacle) else o.radius), 12)
            if abs(cx - px) <= r and abs(cy - py) <= r:
                return i
        return None

    # ── Events ───────────────────────────────────────────────────────────────

    def set_read_only(self, flag: bool):
        self._read_only = flag
        self.canvas.configure(cursor="arrow" if flag else "crosshair")

    def _on_click(self, ev):
        if self._read_only:
            return
        tool = self._tool.get()
        wx, wy = self._c2w(ev.x, ev.y)
        if tool == "wp_add":
            idx = self._hit_wp(ev.x, ev.y)
            if idx is not None:
                self._drag = ("wp", idx)
            else:
                self._waypoints.append(Waypoint(wx, wy, self._float(self.z_var, 2.0),
                                                self._float(self.yaw_var, 0.0)))
                self._drag = ("wp", len(self._waypoints) - 1)
                self._notify_wp()
        elif tool == "wp_move":
            idx = self._hit_wp(ev.x, ev.y)
            if idx is not None:
                self._drag = ("wp", idx)
        elif tool == "wp_del":
            idx = self._hit_wp(ev.x, ev.y)
            if idx is not None:
                self._waypoints.pop(idx)
                self._notify_wp()
        elif tool in ("obs_box", "obs_cyl"):
            idx = self._hit_obs(ev.x, ev.y)
            if idx is not None:
                self._drag = ("obs", idx)
            else:
                n = len(self._obs.obstacles)
                h = max(0.2, self._float(self.h_var, 3.0))
                sz = max(0.2, self._float(self.size_var, 1.0))
                if tool == "obs_box":
                    self._obs.add(BoxObstacle(cx=wx, cy=wy, cz=h / 2, lx=sz, ly=sz, lz=h,
                                              color=BOX_COLORS[n % 4], label=f"Box {n + 1}"))
                else:
                    self._obs.add(CylinderObstacle(cx=wx, cy=wy, z_bot=0, z_top=h, radius=sz / 2,
                                                   color=CYL_COLORS[n % 4], label=f"Cylinder {n + 1}"))
                self._drag = ("obs", len(self._obs.obstacles) - 1)
                self._notify_obs()
        elif tool == "obs_move":
            idx = self._hit_obs(ev.x, ev.y)
            if idx is not None:
                self._drag = ("obs", idx)
        elif tool == "obs_del":
            idx = self._hit_obs(ev.x, ev.y)
            if idx is not None:
                self._obs.remove(idx)
                self._notify_obs()
        self._redraw()

    def _on_drag(self, ev):
        if not self._drag or self._read_only:
            return
        kind, idx = self._drag
        wx, wy = self._c2w(ev.x, ev.y)
        if kind == "wp" and idx < len(self._waypoints):
            self._waypoints[idx].x, self._waypoints[idx].y = wx, wy
        elif kind == "obs" and idx < len(self._obs.obstacles):
            o = self._obs.obstacles[idx]
            o.cx, o.cy = wx, wy
        self._redraw()

    def _on_release(self, ev):
        if self._drag:
            (self._notify_wp if self._drag[0] == "wp" else self._notify_obs)()
        self._drag = None

    def _on_right(self, ev):
        if self._read_only:
            return
        i = self._hit_wp(ev.x, ev.y)
        j = self._hit_obs(ev.x, ev.y)
        if i is not None:
            self._waypoints.pop(i)
            self._notify_wp()
        elif j is not None:
            self._obs.remove(j)
            self._notify_obs()
        self._redraw()

    def _on_hover(self, ev):
        wx, wy = self._c2w(ev.x, ev.y)
        i = self._hit_wp(ev.x, ev.y)
        if i is not None:
            w = self._waypoints[i]
            self._coord.configure(text=f"Waypoint {i + 1}:  x {w.x:+.2f}  y {w.y:+.2f}  z {w.z:.2f}  yaw {w.yaw:.0f}°")
        else:
            self._coord.configure(text=f"x {wx:+.2f} m    y {wy:+.2f} m")

    def _notify_wp(self):
        if self._on_wp:
            self._on_wp(self._waypoints)

    def _notify_obs(self):
        if self._on_obs:
            self._on_obs()

    def _clear_wps(self):
        self._waypoints.clear()
        self._notify_wp()
        self._redraw()

    def _clear_obs(self):
        self._obs.clear()
        self._notify_obs()
        self._redraw()

    # ── Public API ───────────────────────────────────────────────────────────

    def set_waypoints(self, wps: List[Waypoint]):
        self._waypoints = list(wps)
        self._redraw()

    def get_waypoints(self) -> List[Waypoint]:
        return list(self._waypoints)

    def set_rrt_paths(self, paths: List[np.ndarray]):
        self._rrt_paths = paths or []
        self._redraw()

    def set_poly_path(self, path: Optional[np.ndarray]):
        self._poly_path = path
        self._redraw()

    def set_extra_paths(self, paths):
        self._extra_paths = paths or []
        self._redraw()

    def set_live(self, drones: List[Tuple[float, float, float]], trails: List[np.ndarray]):
        self._live, self._trails = drones, trails
        self._redraw()

    def clear_live(self):
        self._live, self._trails = [], []
        self._redraw()

    def refresh(self):
        self._redraw()

    def fit_range(self):
        pts = [(w.x, w.y) for w in self._waypoints] + [(o.cx, o.cy) for o in self._obs.obstacles]
        if pts:
            m = max(max(abs(x), abs(y)) for x, y in pts) + 2
            opts = [4, 6, 8, 10, 15, 20]
            self.range_var.set(str(next((o for o in opts if o >= m), 20)))

    # ── Rendering ────────────────────────────────────────────────────────────

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        r = self._range()
        W, H = self._size()

        # grid
        step = 1.0 if r <= 10 else 2.0
        g = -math.floor(r / step) * step
        while g <= r + 1e-6:
            major = abs(g) < 1e-6
            col = "#2A3442" if major else "#161C25"
            x1, y1 = self._w2c(g, -r); x2, y2 = self._w2c(g, r)
            c.create_line(x1, y1, x2, y2, fill=col)
            x1, y1 = self._w2c(-r, g); x2, y2 = self._w2c(r, g)
            c.create_line(x1, y1, x2, y2, fill=col)
            if abs(g) > 1e-6 and (int(round(g / step)) % 2 == 0 or r <= 6):
                px, py = self._w2c(g, 0)
                c.create_text(px, py + 12, text=f"{g:.0f}", fill=C["text3"], font=F["small"])
                px, py = self._w2c(0, g)
                c.create_text(px - 14, py, text=f"{g:.0f}", fill=C["text3"], font=F["small"])
            g += step
        ox, oy = self._w2c(0, 0)
        ex, _ = self._w2c(r * 0.93, 0)
        _, ey = self._w2c(0, r * 0.93)
        c.create_line(ox, oy, ex, oy, fill=C["red"], width=2, arrow="last")
        c.create_line(ox, oy, ox, ey, fill=C["green"], width=2, arrow="last")
        c.create_text(ex - 8, oy - 12, text="X", fill=C["red"], font=F["small_bold"])
        c.create_text(ox + 12, ey + 6, text="Y", fill=C["green"], font=F["small_bold"])

        # polynomial trajectory
        if self._poly_path is not None and len(self._poly_path) > 1:
            pts = [v for p in self._poly_path for v in self._w2c(p[0], p[1])]
            c.create_line(*pts, fill=C["green"], width=2, smooth=True)
        # RRT* detours
        for path in self._rrt_paths:
            pts = [v for p in path for v in self._w2c(p[0], p[1])]
            c.create_line(*pts, fill=C["purple"], width=2, dash=(8, 4))
            for p in path:
                px, py = self._w2c(p[0], p[1])
                c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=C["purple"], outline="")

        # obstacles
        tool = self._tool.get()
        for o in self._obs.obstacles:
            px, py = self._w2c(o.cx, o.cy)
            col = o.color
            if isinstance(o, BoxObstacle):
                hw, hh = self._m2px(o.lx / 2), self._m2px(o.ly / 2)
                c.create_rectangle(px - hw, py - hh, px + hw, py + hh, fill=col, outline=col,
                                   width=2, stipple="gray25")
                size_txt = f"{o.lx:.1f} x {o.ly:.1f} x {o.lz:.1f} m"
                top = py - hh
            else:
                rp = self._m2px(o.radius)
                c.create_oval(px - rp, py - rp, px + rp, py + rp, fill=col, outline=col,
                              width=2, stipple="gray25")
                size_txt = f"r {o.radius:.1f} m,  h {o.z_top - o.z_bot:.1f} m"
                top = py - rp
            c.create_text(px, top - 22, text=getattr(o, "label", ""), fill=col, font=F["small_bold"])
            c.create_text(px, top - 9, text=size_txt, fill=C["text3"], font=F["small"])

        # swarm / extra paths
        for col, pts in self._extra_paths:
            if len(pts) > 1:
                flat = [v for p in pts for v in self._w2c(p[0], p[1])]
                c.create_line(*flat, fill=col, width=2, dash=(6, 3))
                for p in pts:
                    px, py = self._w2c(*p)
                    c.create_oval(px - 4, py - 4, px + 4, py + 4, fill=col, outline="")

        # waypoint path + markers
        wps = self._waypoints
        if len(wps) >= 2:
            for a, b in zip(wps[:-1], wps[1:]):
                ax, ay = self._w2c(a.x, a.y)
                bx, by = self._w2c(b.x, b.y)
                c.create_line(ax, ay, bx, by, fill=C["accent"], width=2, dash=(7, 3))
                ang = math.atan2(by - ay, bx - ax)
                mx, my = (ax + bx) / 2, (ay + by) / 2
                for s in (-0.45, 0.45):
                    c.create_line(mx, my, mx - 9 * math.cos(ang + s), my - 9 * math.sin(ang + s),
                                  fill=C["accent"], width=2)
        for i, wp in enumerate(wps):
            px, py = self._w2c(wp.x, wp.y)
            col = SERIES[i % len(SERIES)]
            c.create_oval(px - 12, py - 12, px + 12, py + 12, fill=col, outline=C["text"], width=1)
            c.create_text(px, py, text=str(i + 1), fill=C["accent_fg"], font=F["small_bold"])
            yaw = math.radians(wp.yaw)
            c.create_line(px, py, px + 22 * math.cos(yaw), py - 22 * math.sin(yaw),
                          fill=C["amber"], width=2, arrow="last")

        # live drone(s)
        for tr, col in zip(self._trails, SERIES):
            if len(tr) > 1:
                flat = [v for p in tr for v in self._w2c(p[0], p[1])]
                c.create_line(*flat, fill=col, width=2)
        for i, (x, y, yaw) in enumerate(self._live):
            px, py = self._w2c(x, y)
            col = SERIES[i % len(SERIES)]
            c.create_oval(px - 8, py - 8, px + 8, py + 8, fill=col, outline="white", width=2)
            c.create_line(px, py, px + 20 * math.cos(yaw), py - 20 * math.sin(yaw),
                          fill="white", width=2, arrow="last")

        # legend
        items = [(C["accent"], "Waypoints"), (C["purple"], "RRT* detour"),
                 (C["green"], "Smooth trajectory")]
        lx, ly = 12, H - 12 - 20 * len(items)
        for col, txt in items:
            c.create_line(lx, ly + 8, lx + 20, ly + 8, fill=col, width=3)
            c.create_text(lx + 28, ly + 8, text=txt, fill=C["text2"], font=F["small"], anchor="w")
            ly += 20
