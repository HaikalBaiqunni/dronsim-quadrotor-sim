"""
drone_sim/gui/inspector.py
===========================
Right-hand panel of the Map tab. Four pages:

  Waypoints : editable table for the drone selected in the swarm list
  Obstacles : editable list of obstacles
  Route     : trajectory smoothing + obstacle-avoidance algorithm
  Swarm     : number of drones and formation helper

Every edit is pushed back to the shared model (waypoint lists, obstacle
manager, settings dataclasses) and the map canvas is refreshed.
"""

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

import numpy as np

from ..core.catalog import Param
from ..core.mission import (ALGO_HELP, ALGORITHMS, POLY_ORDERS, AvoidSettings,
                            TrajSettings, Waypoint, plan_waypoints)
from ..core.obstacles import BoxObstacle, CylinderObstacle, ObstacleManager
from ..core.trajectory import TrajWaypoint, Trajectory3D, TrajectoryConfig
from .scene_canvas import BOX_COLORS, CYL_COLORS, SceneCanvas
from .theme import C, F, SERIES
from .widgets import (Btn, Card, Chip, LabeledEntry, RadioList, ScrollFrame,
                      Section, Segmented, SliderField, WrapLabel, caption, check,
                      combo, divider)

PRESET_ROUTES = {
    "Square": [(0, 0, 0.1), (0, 0, 2), (3, 0, 2), (3, 3, 2), (0, 3, 2), (0, 0, 2)],
    "Circle": [(0, 0, 0.1), (0, 0, 2), (2, 0, 2), (0, 2, 2), (-2, 0, 2), (0, -2, 2), (2, 0, 2)],
    "Lawn mower": [(-3, -3, 0.1), (-3, -3, 2), (3, -3, 2), (3, -1, 2), (-3, -1, 2),
                   (-3, 1, 2), (3, 1, 2), (3, 3, 2), (-3, 3, 2)],
    "Climbing spiral": [(0, 0, 0.1), (2, 0, 1.5), (0, 2, 2.5), (-2, 0, 3.5), (0, -2, 4.5), (1, 0, 5.5)],
}


def _mono_entry(parent, var, width=4, fg=None):
    return tk.Entry(parent, textvariable=var, width=width, justify="right", bg=C["surface3"],
                    fg=fg or C["accent"], insertbackground=C["text"], relief="flat",
                    font=F["mono_small"], highlightthickness=1,
                    highlightbackground=C["border"], highlightcolor=C["accent"])


# ── Waypoint table ───────────────────────────────────────────────────────────

class WaypointTable(tk.Frame):
    COLS = ["X", "Y", "Z", "Yaw", "Hover"]
    HINTS = {"X": "m", "Y": "m", "Z": "m", "Yaw": "deg", "Hover": "s"}

    def __init__(self, parent, on_change: Callable[[List[Waypoint]], None], bg: str = None):
        super().__init__(parent, bg=bg or C["surface"])
        self.on_change = on_change
        self._wps: List[Waypoint] = []
        self._vars: List[List[tk.StringVar]] = []
        self.grid_columnconfigure(0, minsize=26)
        for i in range(1, 6):
            self.grid_columnconfigure(i, weight=1, uniform="wpcol")
        self.grid_columnconfigure(6, minsize=30)
        for i, name in enumerate(self.COLS):
            tk.Label(self, text=f"{name}", bg=self.cget("bg"), fg=C["text3"], font=F["small_bold"]
                     ).grid(row=0, column=i + 1, sticky="e", padx=2, pady=(0, 2))
        self._first_row = 1

    def set_waypoints(self, wps: List[Waypoint]):
        self._wps = wps
        for w in list(self.children.values()):
            info = w.grid_info()
            if info and int(info.get("row", 0)) >= self._first_row:
                w.destroy()
        self._vars = []
        if not wps:
            tk.Label(self, text="No waypoints yet. Click on the map to add some.",
                     bg=self.cget("bg"), fg=C["text3"], font=F["small"], wraplength=280,
                     justify="left").grid(row=1, column=0, columnspan=7, sticky="w", pady=8)
        for r, wp in enumerate(wps):
            row = r + self._first_row
            col = SERIES[r % len(SERIES)]
            tk.Label(self, text=str(r + 1), bg=col, fg=C["accent_fg"], font=F["small_bold"],
                     width=2).grid(row=row, column=0, padx=(0, 4), pady=2)
            vs = []
            for c, val in enumerate([wp.x, wp.y, wp.z, wp.yaw, wp.hover_time]):
                v = tk.StringVar(value=f"{val:.1f}" if abs(val - round(val)) > 1e-9 else f"{val:.0f}")
                e = _mono_entry(self, v, width=3)
                e.grid(row=row, column=c + 1, sticky="ew", padx=2, pady=2, ipady=3)
                e.bind("<Return>", lambda ev: self._commit())
                e.bind("<FocusOut>", lambda ev: self._commit())
                vs.append(v)
            self._vars.append(vs)
            tk.Button(self, text="✕", bg=C["surface"], fg=C["red"], relief="flat", bd=0,
                      font=F["small_bold"], cursor="hand2", width=2, highlightthickness=0,
                      command=lambda i=r: self._delete(i)).grid(row=row, column=6, padx=(2, 0))

    def _commit(self):
        changed = False
        for wp, vs in zip(self._wps, self._vars):
            try:
                new = [float(v.get()) for v in vs]
            except ValueError:
                continue
            old = [wp.x, wp.y, wp.z, wp.yaw, wp.hover_time]
            if any(abs(a - b) > 1e-9 for a, b in zip(new, old)):
                wp.x, wp.y, wp.z, wp.yaw, wp.hover_time = new
                changed = True
        if changed:
            self.on_change(self._wps)

    def _delete(self, idx: int):
        if 0 <= idx < len(self._wps):
            self._wps.pop(idx)
            self.on_change(self._wps)


# ── Obstacle list ────────────────────────────────────────────────────────────

class ObstacleList(tk.Frame):
    def __init__(self, parent, manager: ObstacleManager, on_change: Callable):
        super().__init__(parent, bg=C["surface"])
        self.mgr, self.on_change = manager, on_change

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        obs = self.mgr.obstacles
        if not obs:
            WrapLabel(self, "No obstacles. Use the Box or Cylinder tool on the map, or the buttons above.",
                      fg=C["text3"], bg=C["surface"]).pack(fill="x", pady=8)
            return
        for i, o in enumerate(obs):
            card = Card(self, pad=8)
            card.pack(fill="x", pady=3)
            top = tk.Frame(card.inner, bg=card.cget("bg"))
            top.pack(fill="x")
            tk.Label(top, text="●", fg=o.color, bg=card.cget("bg"), font=F["bold"]).pack(side="left")
            tk.Label(top, text=getattr(o, "label", f"Obstacle {i + 1}"), fg=C["text"],
                     bg=card.cget("bg"), font=F["bold"]).pack(side="left", padx=6)
            tk.Button(top, text="Remove", bg=card.cget("bg"), fg=C["red"], relief="flat", bd=0,
                      font=F["small_bold"], cursor="hand2", highlightthickness=0,
                      command=lambda k=i: self._remove(k)).pack(side="right")
            grid = tk.Frame(card.inner, bg=card.cget("bg"))
            grid.pack(fill="x", pady=(6, 0))
            if isinstance(o, BoxObstacle):
                fields = [("X", "cx"), ("Y", "cy"), ("Width", "lx"), ("Height", "lz")]
            else:
                fields = [("X", "cx"), ("Y", "cy"), ("Radius", "radius"), ("Height", "z_top")]
            for c, (lbl, attr) in enumerate(fields):
                grid.grid_columnconfigure(c, weight=1, uniform="ofield")
                tk.Label(grid, text=lbl, bg=card.cget("bg"), fg=C["text3"], font=F["small"],
                         anchor="w").grid(row=0, column=c, sticky="w", padx=2)
                v = tk.StringVar(value=f"{getattr(o, attr):.2f}".rstrip("0").rstrip("."))
                e = _mono_entry(grid, v, width=3)
                e.grid(row=1, column=c, sticky="ew", padx=2, ipady=3)
                cb = lambda ev, o=o, a=attr, v=v: self._apply(o, a, v)
                e.bind("<Return>", cb)
                e.bind("<FocusOut>", cb)

    def _apply(self, o, attr, var):
        try:
            val = float(var.get())
        except ValueError:
            return
        if abs(val - getattr(o, attr)) < 1e-9:
            return
        setattr(o, attr, val)
        if isinstance(o, BoxObstacle):
            if attr == "lx":
                o.ly = val
            elif attr == "lz":
                o.cz = val / 2
        self.on_change()

    def _remove(self, idx):
        self.mgr.remove(idx)
        self.on_change()
        self.rebuild()


# ── Inspector ────────────────────────────────────────────────────────────────

class Inspector(tk.Frame):
    """Owns waypoint lists (per drone), trajectory + avoidance settings."""

    PAGES = [("wps", "Waypoints"), ("obs", "Obstacles"), ("route", "Route"), ("swarm", "Swarm")]

    def __init__(self, parent, canvas: SceneCanvas, manager: ObstacleManager,
                 on_mission_changed: Optional[Callable] = None):
        super().__init__(parent, bg=C["surface"])
        self.canvas, self.mgr = canvas, manager
        self.on_mission_changed = on_mission_changed or (lambda: None)
        self.traj = TrajSettings()
        self.avoid = AvoidSettings()
        self.drone_wps: List[List[Waypoint]] = [[]]
        self.swarm_enabled = tk.IntVar(value=0)
        self.active = 0
        self._pages = {}
        self._build()
        self.set_drone_count(1)

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def waypoints(self) -> List[Waypoint]:
        return self.drone_wps[0]

    def set_waypoints(self, wps: List[Waypoint]):
        self.drone_wps[0] = list(wps)
        if self.active == 0:
            self.canvas.set_waypoints(self.drone_wps[0])
        self._sync_table()
        self._update_extra_paths()

    def swarm_separation(self):
        return bool(self.sep_on.get()), float(self._sep_field.get())

    def n_drones(self) -> int:
        return len(self.drone_wps) if self.swarm_enabled.get() else 1

    def swarm_paths(self) -> List[List[Waypoint]]:
        return [list(w) for w in self.drone_wps] if self.swarm_enabled.get() else []

    def show_page(self, key: str):
        self._page_bar.set(key, notify=True)

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self):
        self._page_bar = Segmented(self, self.PAGES, "wps", self._show_page,
                                   font=F["small_bold"], pady=6)
        self._page_bar.pack(fill="x", padx=10, pady=(10, 8))
        self._host = tk.Frame(self, bg=C["surface"])
        self._host.pack(fill="both", expand=True)
        for key, _ in self.PAGES:
            sf = ScrollFrame(self._host)
            self._pages[key] = sf
        self._build_wps(self._pages["wps"].inner)
        self._build_obs(self._pages["obs"].inner)
        self._build_route(self._pages["route"].inner)
        self._build_swarm(self._pages["swarm"].inner)
        self._show_page("wps")

    def _show_page(self, key):
        for k, sf in self._pages.items():
            sf.pack_forget()
        self._pages[key].pack(fill="both", expand=True, padx=(10, 4), pady=(0, 8))
        if key == "obs":
            self._obs_list.rebuild()

    # ── Waypoints page ───────────────────────────────────────────────────────

    def _build_wps(self, p):
        self._drone_label = WrapLabel(p, "", fg=C["text2"], bg=C["surface"], font=F["body"])
        row = tk.Frame(p, bg=C["surface"])
        row.pack(fill="x", pady=(0, 6))
        self._wps_first_row = row
        row.grid_columnconfigure(0, weight=1)
        self._preset = combo(row, list(PRESET_ROUTES), "Square", width=14)
        self._preset.grid(row=0, column=0, sticky="ew")
        Btn(row, "Load route", self._load_preset, kind="secondary", padx=10, pady=5,
            font=F["small_bold"]).grid(row=0, column=1, padx=(6, 0))
        WrapLabel(p, "Yaw is in degrees. Hover is how long the drone waits at that point. "
                     "Point 1 should sit on the ground (z 0.1) so the drone takes off there.",
                  fg=C["text3"], bg=C["surface"]).pack(fill="x", pady=(0, 6))
        self._table = WaypointTable(p, self._on_table_change)
        self._table.pack(fill="x")
        Btn(p, "Add waypoint", self._add_row, kind="ghost", padx=10, pady=5,
            font=F["small_bold"]).pack(fill="x", pady=(8, 0))

    def _load_preset(self):
        pts = PRESET_ROUTES[self._preset.get()]
        self._active_list()[:] = [Waypoint(*p) for p in pts]
        self._on_table_change(self._active_list())

    def _add_row(self):
        wps = self._active_list()
        last = wps[-1] if wps else Waypoint(0, 0, 2)
        wps.append(Waypoint(last.x + 1, last.y, last.z))
        self._on_table_change(wps)

    def _active_list(self) -> List[Waypoint]:
        return self.drone_wps[self.active]

    def _on_table_change(self, wps):
        self.canvas.set_waypoints(wps)
        self._sync_table()
        self._update_extra_paths()
        self.on_mission_changed()

    def _sync_table(self):
        wps = self._active_list()
        self._table.set_waypoints(wps)
        if len(self.drone_wps) > 1 and self.swarm_enabled.get():
            self._drone_label.configure(text=f"Editing drone {self.active + 1} of {len(self.drone_wps)}")
            self._drone_label.pack(fill="x", pady=(0, 6), before=self._wps_first_row)
        else:
            self._drone_label.pack_forget()

    def on_canvas_waypoints(self, wps):
        """Called by the map when the user adds, moves or deletes waypoints."""
        self.drone_wps[self.active] = wps
        self._sync_table()
        self._update_extra_paths()
        self.on_mission_changed()

    # ── Obstacles page ───────────────────────────────────────────────────────

    def _build_obs(self, p):
        row = tk.Frame(p, bg=C["surface"])
        row.pack(fill="x", pady=(0, 8))
        for i in range(3):
            row.grid_columnconfigure(i, weight=1, uniform="obsbtn")
        Btn(row, "Add box", lambda: self._quick_add("box"), padx=6, pady=5,
            font=F["small_bold"]).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        Btn(row, "Add cylinder", lambda: self._quick_add("cyl"), padx=6, pady=5,
            font=F["small_bold"]).grid(row=0, column=1, sticky="ew", padx=3)
        Btn(row, "Clear all", self._clear_obs, kind="danger", padx=6, pady=5,
            font=F["small_bold"]).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        self._obs_list = ObstacleList(p, self.mgr, self.on_obstacles_changed)
        self._obs_list.pack(fill="x")

    def _quick_add(self, kind):
        n = len(self.mgr.obstacles)
        if kind == "box":
            self.mgr.add(BoxObstacle(cx=1.5, cy=0, cz=1.5, lx=1, ly=1, lz=3,
                                     color=BOX_COLORS[n % 4], label=f"Box {n + 1}"))
        else:
            self.mgr.add(CylinderObstacle(cx=1.5, cy=0, z_bot=0, z_top=3, radius=0.5,
                                          color=CYL_COLORS[n % 4], label=f"Cylinder {n + 1}"))
        self.on_obstacles_changed()
        self._obs_list.rebuild()

    def _clear_obs(self):
        self.mgr.clear()
        self.on_obstacles_changed()
        self._obs_list.rebuild()

    def on_obstacles_changed(self):
        self.canvas.refresh()
        if self._pages["obs"].winfo_ismapped():
            pass
        self.on_mission_changed()

    # ── Route page ───────────────────────────────────────────────────────────

    def _build_route(self, p):
        tr = self.traj
        caption(p, "Trajectory", pady=(0, 4))
        self._order = tk.IntVar(value=tr.poly_order)
        RadioList(p, [(str(k), v) for k, v in POLY_ORDERS.items()], value=str(tr.poly_order),
                  command=lambda v: self._set_traj("poly_order", int(v)), bg=C["surface"]).pack(fill="x")
        tk.Label(p, text="Time allocation", bg=C["surface"], fg=C["text2"], font=F["body"],
                 anchor="w").pack(fill="x", pady=(8, 3))
        Segmented(p, [("trap", "Trapezoid"), ("constant", "Constant")], tr.time_method,
                  lambda v: self._set_traj("time_method", v), font=F["small_bold"], pady=5
                  ).pack(fill="x")
        self._vmax = LabeledEntry(p, "Max speed", tr.v_max, unit="m/s", on_change=self._read_traj)
        self._amax = LabeledEntry(p, "Max acceleration", tr.a_max, unit="m/s²", on_change=self._read_traj)
        self._avg = LabeledEntry(p, "Average speed", tr.avg_speed, unit="m/s", on_change=self._read_traj)
        for w in (self._vmax, self._amax, self._avg):
            w.pack(fill="x", pady=3)
        self._loop = tk.IntVar(value=0)
        check(p, "Loop the route", self._loop, self._read_traj).pack(fill="x", pady=(2, 6))
        Btn(p, "Preview smooth trajectory", self._preview_traj, kind="secondary", padx=10, pady=6,
            font=F["small_bold"]).pack(fill="x")
        self._traj_info = WrapLabel(p, "", fg=C["text2"], bg=C["surface"])
        self._traj_info.pack(fill="x", pady=(6, 0))

        divider(p, pady=12)
        caption(p, "Obstacle avoidance", pady=(0, 4))
        self._algo = combo(p, ALGORITHMS, self.avoid.algorithm, self._set_algo, width=14)
        self._algo.pack(fill="x")
        self._algo_help = WrapLabel(p, ALGO_HELP[self.avoid.algorithm], fg=C["text3"], bg=C["surface"])
        self._algo_help.pack(fill="x", pady=(4, 6))

        self._av_fields = {}
        self._av_groups = {}
        specs = {
            "common": [Param("safety_margin", "Safety margin", 0.4, 0.15, 1.0, 0.05, unit="m",
                             hint="Distance kept from obstacle surfaces (RRT* and CBF)")],
            "apf": [Param("apf_eta", "Repulsive gain", 8.0, 1, 30, 0.5),
                    Param("apf_d0", "Influence radius", 1.5, 0.5, 4, 0.1, unit="m"),
                    Param("apf_fmax", "Max push", 6.0, 1, 15, 0.5, unit="m/s²")],
            "rrt": [Param("rrt_iter", "Iterations per segment", 800, 200, 3000, 100, kind="int",
                          hint="More iterations give shorter routes but take longer"),
                    Param("rrt_step", "Step size", 0.6, 0.2, 1.5, 0.05, unit="m")],
            "cbf": [Param("cbf_k0", "Barrier gain k0", 4.0, 0.5, 12, 0.5),
                    Param("cbf_k1", "Barrier gain k1", 4.0, 0.5, 12, 0.5),
                    Param("cbf_influence", "Influence distance", 3.0, 1, 6, 0.5, unit="m")],
        }
        titles = {"common": None, "apf": "APF settings", "rrt": "RRT* settings", "cbf": "CBF settings"}
        for grp, plist in specs.items():
            f = tk.Frame(p, bg=C["surface"])
            if titles[grp]:
                caption(f, titles[grp], pady=(8, 2))
            for prm in plist:
                fld = SliderField(f, prm, getattr(self.avoid, prm.key), self._set_avoid)
                fld.pack(fill="x", pady=4)
                self._av_fields[prm.key] = fld
            self._av_groups[grp] = f
        self._plan_btn = Btn(p, "Plan RRT* route now", self._plan_route, kind="secondary",
                             padx=10, pady=6, font=F["small_bold"])
        self._plan_info = WrapLabel(p, "", fg=C["text2"], bg=C["surface"])
        self._refresh_av_groups()

    def _set_traj(self, key, val):
        setattr(self.traj, key, val)
        self.canvas.set_poly_path(None)
        self.on_mission_changed()

    def _read_traj(self):
        self.traj.v_max = self._vmax.get_float(self.traj.v_max)
        self.traj.a_max = self._amax.get_float(self.traj.a_max)
        self.traj.avg_speed = self._avg.get_float(self.traj.avg_speed)
        self.traj.loop = bool(self._loop.get())
        self.on_mission_changed()

    def _preview_traj(self):
        wps = self.waypoints
        if len(wps) < 2:
            self._traj_info.configure(text="Add at least two waypoints first.")
            return
        order = self.traj.poly_order
        if order < 3:
            self._traj_info.configure(text="Straight segments selected. Pick a polynomial order to preview a smooth path.")
            self.canvas.set_poly_path(None)
            return
        try:
            cfg = TrajectoryConfig(poly_order=order, time_method=self.traj.time_method,
                                   v_max=self.traj.v_max, a_max=self.traj.a_max,
                                   avg_speed=self.traj.avg_speed, loop=self.traj.loop)
            tj = Trajectory3D([TrajWaypoint(w.x, w.y, w.z, w.yaw) for w in wps], cfg)
            self.canvas.set_poly_path(tj.sample_path(200))
            self._traj_info.configure(
                text=f"{len(wps) - 1} segments, total time {tj.total_time:.1f} s.")
        except Exception as e:  # noqa: BLE001
            self._traj_info.configure(text=f"Could not build trajectory: {e}")

    def _set_algo(self, name):
        self.avoid.algorithm = name
        self._algo_help.configure(text=ALGO_HELP[name])
        self._refresh_av_groups()
        self.on_mission_changed()

    def _set_avoid(self, key, val):
        setattr(self.avoid, key, val)
        self.on_mission_changed()

    def _refresh_av_groups(self):
        a = self.avoid
        for f in list(self._av_groups.values()) + [self._plan_btn, self._plan_info]:
            f.pack_forget()
        if a.algorithm == "None":
            return
        self._av_groups["common"].pack(fill="x")
        if a.uses_apf:
            self._av_groups["apf"].pack(fill="x")
        if a.uses_rrt:
            self._av_groups["rrt"].pack(fill="x")
            self._plan_btn.pack(fill="x", pady=(10, 0))
            self._plan_info.pack(fill="x", pady=(6, 0))
        if a.uses_cbf:
            self._av_groups["cbf"].pack(fill="x")

    def _plan_route(self):
        wps = self.waypoints
        obs = self.mgr.obstacles
        if len(wps) < 2 or not obs:
            self._plan_info.configure(text="Needs at least two waypoints and one obstacle.")
            return
        self._plan_info.configure(text="Planning…")
        self.update_idletasks()
        aug, paths = plan_waypoints(wps, obs, self.avoid)
        self.canvas.set_rrt_paths(paths)
        if paths:
            self._plan_info.configure(
                text=f"{len(paths)} blocked segment(s) rerouted. {len(aug) - len(wps)} points added.")
        else:
            self._plan_info.configure(text="Every segment is already clear of obstacles.")

    # ── Swarm page ───────────────────────────────────────────────────────────

    def _build_swarm(self, p):
        WrapLabel(p, "Fly several drones at once. Each drone has its own waypoint list; pick which one "
                     "you are editing below. Swarm flights use straight waypoint following.",
                  fg=C["text2"], bg=C["surface"]).pack(fill="x", pady=(0, 8))
        check(p, "Fly as a swarm", self.swarm_enabled, self._swarm_toggled).pack(fill="x", pady=4)
        row = tk.Frame(p, bg=C["surface"])
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, weight=1)
        tk.Label(row, text="Number of drones", bg=C["surface"], fg=C["text2"], font=F["body"],
                 anchor="w").grid(row=0, column=0, sticky="w")
        self._n_combo = combo(row, [str(i) for i in range(2, 9)], "3", width=4)
        self._n_combo.grid(row=0, column=1, sticky="e")
        self._n_combo.bind("<<ComboboxSelected>>", lambda e: self._swarm_toggled())
        row2 = tk.Frame(p, bg=C["surface"])
        row2.pack(fill="x", pady=4)
        row2.grid_columnconfigure(0, weight=1)
        tk.Label(row2, text="Editing drone", bg=C["surface"], fg=C["text2"], font=F["body"],
                 anchor="w").grid(row=0, column=0, sticky="w")
        self._active_combo = combo(row2, ["1"], "1", width=4)
        self._active_combo.grid(row=0, column=1, sticky="e")
        self._active_combo.bind("<<ComboboxSelected>>", lambda e: self._set_active(int(self._active_combo.get()) - 1))
        Btn(p, "Auto formation (ring around drone 1)", self._auto_formation, kind="secondary",
            padx=10, pady=6, font=F["small_bold"]).pack(fill="x", pady=(8, 0))
        divider(p, pady=10)
        caption(p, "Drone-to-drone spacing", pady=(0, 2))
        self.sep_on = tk.IntVar(value=1)
        check(p, "Keep drones apart (CBF)", self.sep_on, self.on_mission_changed).pack(fill="x")
        self._sep_field = SliderField(p, Param("sep", "Minimum separation", 0.8, 0.4, 2.0, 0.1, unit="m",
                                               hint="Drones share position and velocity and push each other "
                                                    "away to stay at least this far apart."),
                                      0.8, lambda k, v: self.on_mission_changed())
        self._sep_field.pack(fill="x", pady=4)
        self._swarm_info = WrapLabel(p, "", fg=C["text3"], bg=C["surface"])
        self._swarm_info.pack(fill="x", pady=(8, 0))

    def _swarm_toggled(self):
        n = int(self._n_combo.get()) if self.swarm_enabled.get() else 1
        self.set_drone_count(n)
        self.on_mission_changed()

    def set_drone_count(self, n: int):
        while len(self.drone_wps) < n:
            base = self.drone_wps[0]
            i = len(self.drone_wps)
            ang = 2 * math.pi * i / max(n, 2)
            dx, dy = 1.5 * math.cos(ang), 1.5 * math.sin(ang)
            self.drone_wps.append([Waypoint(w.x + dx, w.y + dy, w.z, w.yaw, 1.0, w.hover_time) for w in base])
        del self.drone_wps[max(n, 1):]
        vals = [str(i + 1) for i in range(len(self.drone_wps))]
        self._active_combo.configure(values=vals)
        if self.active >= len(self.drone_wps):
            self.active = 0
        self._active_combo.set(str(self.active + 1))
        self._set_active(self.active)
        self._swarm_info.configure(
            text=f"{len(self.drone_wps)} drones." if self.swarm_enabled.get() else "Swarm is off. One drone will fly.")

    def _set_active(self, idx):
        self.active = max(0, min(idx, len(self.drone_wps) - 1))
        self.canvas.set_waypoints(self.drone_wps[self.active])
        self._sync_table()
        self._update_extra_paths()

    def _auto_formation(self):
        n = len(self.drone_wps)
        base = self.drone_wps[0]
        for i in range(1, n):
            ang = 2 * math.pi * i / n
            dx, dy = 1.5 * math.cos(ang), 1.5 * math.sin(ang)
            self.drone_wps[i] = [Waypoint(w.x + dx, w.y + dy, w.z, w.yaw, 1.0, w.hover_time) for w in base]
        self._set_active(self.active)
        self.on_mission_changed()

    def _update_extra_paths(self):
        if not self.swarm_enabled.get():
            self.canvas.set_extra_paths([])
            return
        extra = [(SERIES[i % len(SERIES)], [(w.x, w.y) for w in wps])
                 for i, wps in enumerate(self.drone_wps) if i != self.active]
        self.canvas.set_extra_paths(extra)
