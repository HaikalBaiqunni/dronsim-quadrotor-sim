"""
drone_sim/gui/app.py
=====================
DRONSIM v2.0 main window.

  ┌ top bar: brand · setup steps · Run / Pause / Stop / Reset ─────────────┐
  │ sidebar        │ centre tabs: 3D scene · Map · Plots · Compare │ telemetry │
  ├────────────────┴───────────────────────────────────────────────┴───────┤
  └ timeline: progress + event markers · speed · export ───────────────────┘

Author : Haikal Hakim Baiqunni
License: MIT
"""

import math
import tkinter as tk
from tkinter import filedialog

import numpy as np

from ..core.catalog import CATALOG, build_controller
from ..core.mission import (MissionSpec, Waypoint, build_simulation, default_waypoints,
                            plan_waypoints)
from ..core.obstacles import ObstacleManager
from ..core.safety import CBFParams, CBFSafetyFilter
from ..core.swarm import SingleDroneSim, SwarmSimulation, WaypointSpec
from .inspector import Inspector
from .plots import ComparePanel, ScenePlot, SignalPlots
from .scene_canvas import SceneCanvas
from .sidebar import Sidebar
from .telemetry import TelemetryPanel, Timeline
from .theme import C, F, apply_style
from .widgets import Btn, Chip, Segmented, WrapLabel

AUTHOR = "Haikal Hakim Baiqunni"
VERSION = "2.0"

STEPS = [("vehicle", "Vehicle"), ("mission", "Mission"), ("controller", "Controller"),
         ("world", "World"), ("events", "Events")]


class DroneSimApp:
    TABS = [("scene", "3D scene"), ("map", "Map"), ("plots", "Plots"), ("compare", "Compare")]
    TELEMETRY_TABS = ("scene", "plots")

    def __init__(self):
        self.root = tk.Tk()
        apply_style(self.root)
        self.root.title(f"DRONSIM v{VERSION}")
        self.root.configure(bg=C["bg"])
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1600, sw - 60), min(940, sh - 100)
        self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        self.root.minsize(1240, 700)

        self.obs = ObstacleManager()
        self._sim = None
        self._swarm = None
        self._running = False
        self._paused = False
        self._job = None
        self._speed = 1.0
        self._tab = "scene"
        self._static_dirty = True
        self._active_drone = 0
        self._run_info = {}

        self._build()
        self.inspector.set_waypoints(default_waypoints())
        self._on_mission_changed()
        self.sidebar.refresh_summaries()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<space>", self._on_space)
        self._show_tab("scene")

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        self._build_topbar()
        self.timeline = Timeline(self.root, self._on_speed, self._export_csv, self._export_mat)
        self.timeline.pack(side="bottom", fill="x")
        tk.Frame(self.root, bg=C["border"], height=1).pack(side="bottom", fill="x")

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        self.sidebar = Sidebar(body, on_open_map=lambda: self._show_tab("map"),
                               on_change=self._on_setup_changed, on_section=self._on_section)
        self.sidebar.pack(side="left", fill="y")
        tk.Frame(body, bg=C["border"], width=1).pack(side="left", fill="y")

        self.telemetry = TelemetryPanel(body, on_select_drone=self._select_drone)
        self.telemetry.pack(side="right", fill="y")
        self._tele_sep = tk.Frame(body, bg=C["border"], width=1)
        self._tele_sep.pack(side="right", fill="y")

        self.center = tk.Frame(body, bg=C["surface"])
        self.center.pack(side="left", fill="both", expand=True)
        self._body = body

        bar = tk.Frame(self.center, bg=C["surface"])
        bar.pack(fill="x", padx=12, pady=(10, 6))
        self.tabbar = Segmented(bar, self.TABS, "scene", self._show_tab, bg=C["surface"], pady=7)
        self.tabbar.pack(side="left")
        self.tab_hint = tk.Label(bar, text="", bg=C["surface"], fg=C["text3"], font=F["small"], anchor="e")
        self.tab_hint.pack(side="right", fill="x", expand=True)

        self.pages = tk.Frame(self.center, bg=C["surface"])
        self.pages.pack(fill="both", expand=True)

        self.scene = ScenePlot(self.pages)
        self.signals = SignalPlots(self.pages)
        self.map_page = tk.Frame(self.pages, bg=C["surface"])
        self.map = SceneCanvas(self.map_page, self.obs, self._on_map_waypoints, self._on_map_obstacles)
        self.inspector = Inspector(self.map_page, self.map, self.obs, self._on_mission_changed)
        self.inspector.configure(width=372)
        self.inspector.pack_propagate(False)
        self.inspector.pack(side="right", fill="y")
        tk.Frame(self.map_page, bg=C["border"], width=1).pack(side="right", fill="y")
        self.map.pack(side="left", fill="both", expand=True)
        self.compare = ComparePanel(self.pages, self._compare_scenario)
        self._pages = {"scene": self.scene, "map": self.map_page,
                       "plots": self.signals, "compare": self.compare}

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=C["surface2"], height=60)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="DRONSIM", bg=C["surface2"], fg=C["text"], font=F["h1"]).pack(side="left", padx=(16, 6))
        Chip(bar, f"v{VERSION}", "accent").pack(side="left", padx=(0, 18))

        self._step_btns = {}
        for i, (key, label) in enumerate(STEPS):
            b = tk.Button(bar, text=f"{i + 1}  {label}", relief="flat", bd=0, font=F["body"],
                          padx=12, pady=6, cursor="hand2", highlightthickness=0,
                          bg=C["surface2"], fg=C["text2"], activebackground=C["surface3"],
                          activeforeground=C["text"], command=lambda k=key: self._goto_step(k))
            b.pack(side="left", padx=2)
            self._step_btns[key] = b

        Btn(bar, "About", self._show_about, kind="ghost", padx=12, pady=6).pack(side="right", padx=(4, 14))
        self.btn_reset = Btn(bar, "Reset", self._reset, kind="secondary")
        self.btn_reset.pack(side="right", padx=4)
        self.btn_stop = Btn(bar, "Stop", self._stop, kind="secondary")
        self.btn_stop.pack(side="right", padx=4)
        self.btn_run = Btn(bar, "▶  Run", self._run_or_pause, kind="success", padx=18)
        self.btn_run.pack(side="right", padx=4)
        self.status_lbl = tk.Label(bar, text="", bg=C["surface2"], fg=C["text2"], font=F["small"], anchor="e")
        self.status_lbl.pack(side="right", padx=12, fill="x", expand=True)

    def _goto_step(self, key):
        self.sidebar.open(key)
        if key == "mission" and self._tab not in ("map",):
            pass

    def _on_section(self, key):
        for k, b in self._step_btns.items():
            on = k == key
            b.configure(bg=C["accent_bg"] if on else C["surface2"], fg=C["accent"] if on else C["text2"])

    def _msg(self, text: str):
        self.status_lbl.configure(text=text)

    # ── Tabs ─────────────────────────────────────────────────────────────────

    def _show_tab(self, key):
        self._tab = key
        for k, p in self._pages.items():
            p.pack_forget()
        self._pages[key].pack(fill="both", expand=True)
        self.tabbar.set(key)
        want_tele = key in self.TELEMETRY_TABS
        if want_tele and not self.telemetry.winfo_ismapped():
            self._tele_sep.pack(side="right", fill="y", before=self.center)
            self.telemetry.pack(side="right", fill="y", before=self._tele_sep)
        elif not want_tele and self.telemetry.winfo_ismapped():
            self.telemetry.pack_forget()
            self._tele_sep.pack_forget()
        hints = {"scene": "Live 3D view. Drag to rotate, scroll to zoom.",
                 "map": "Click the map to place waypoints and obstacles.",
                 "plots": "Live signals. Red dotted lines mark events.",
                 "compare": "Fly the same mission with several controllers."}
        self.tab_hint.configure(text=hints[key])
        if key == "scene" and self._static_dirty:
            self._refresh_static()
        if key == "map":
            self.map.refresh()

    # ── Setup change handlers ────────────────────────────────────────────────

    def _on_setup_changed(self):
        self.timeline.set_progress(0, self.sidebar.get_mission_basics()[3], self._scheduled_marks())

    def _scheduled_marks(self):
        return [(e.t, e.describe()) for e in self.sidebar.get_events()]

    def _on_map_waypoints(self, wps):
        self.inspector.on_canvas_waypoints(wps)
        if len(wps) >= 2 and self.sidebar.kind.get() == "preset":
            self.sidebar.set_mission_kind("waypoints")

    def _on_map_obstacles(self):
        self.inspector.on_obstacles_changed()
        self.inspector._obs_list.rebuild()

    def _on_mission_changed(self):
        n = self.inspector.n_drones()
        self.sidebar.set_waypoint_summary(len(self.inspector.waypoints), n)
        self.telemetry.set_drone_count(n)
        self._static_dirty = True
        if self._tab == "scene" and not self._running:
            self._refresh_static()

    def _refresh_static(self):
        self._static_dirty = False
        wps = self.inspector.waypoints if self.sidebar.kind.get() == "waypoints" else None
        start = (wps[0].x, wps[0].y, 0.1) if wps else (0.0, 0.0, 0.1)
        self.scene.set_static(self.obs.obstacles, wps, self.map._rrt_paths, self.map._poly_path, start)

    # ── Scenario ─────────────────────────────────────────────────────────────

    def _build_spec(self) -> MissionSpec:
        kind, preset, hover, duration = self.sidebar.get_mission_basics()
        return MissionSpec(kind=kind, preset=preset, hover_sp=hover,
                           waypoints=[w.copy() for w in self.inspector.waypoints],
                           traj=self.inspector.traj, avoid=self.inspector.avoid,
                           obstacles=self.obs.obstacles, duration=duration)

    def _compare_scenario(self) -> dict:
        return dict(spec=self._build_spec(), values_by_key=self.sidebar.all_values(),
                    dp=self.sidebar.get_drone_params(), env=self.sidebar.get_env_params(),
                    use_estimator=self.sidebar.use_estimator(), events=self.sidebar.get_events())

    # ── Run control ──────────────────────────────────────────────────────────

    def _on_space(self, ev):
        if ev.widget.winfo_class() in ("Entry", "TCombobox", "Text"):
            return
        self._run_or_pause()

    def _run_or_pause(self):
        if self._running and not self._paused:
            self._pause()
        elif self._running and self._paused:
            self._resume()
        else:
            self._launch()

    def _pause(self):
        (self._sim or self._swarm).pause()
        self._paused = True
        self.btn_run.configure(text="▶  Resume")
        self.telemetry.set_status("Paused", "warn")
        self._msg("Paused")

    def _resume(self):
        (self._sim or self._swarm).resume()
        self._paused = False
        self.btn_run.configure(text="Pause")
        self.telemetry.set_status("Flying", "good")
        self._msg("Running")

    def _launch(self):
        spec = self._build_spec()
        n_drones = self.inspector.n_drones()
        if spec.kind == "waypoints" and n_drones == 1 and len(spec.waypoints) < 2:
            self._msg("Add at least two waypoints on the Map first")
            self._show_tab("map")
            return
        if n_drones > 1 and any(len(w) < 1 for w in self.inspector.swarm_paths()):
            self._msg("Every drone needs at least one waypoint")
            self._show_tab("map")
            return
        self._stop(quiet=True)
        dp = self.sidebar.get_drone_params()
        env = self.sidebar.get_env_params()
        fused = self.sidebar.use_estimator()
        events = self.sidebar.get_events()
        key, vals = self.sidebar.get_controller()
        self._msg("Planning route…")
        self.root.update_idletasks()

        try:
            if n_drones > 1:
                self._launch_swarm(spec, dp, env, fused, key, vals)
            else:
                ctrl = build_controller(key, dp, vals)
                built = build_simulation(spec, ctrl, dp, env, fused, events, real_time=True)
                built.sim.config.real_time_factor = self._speed
                self._sim, self._swarm = built.sim, None
                self.map.set_rrt_paths(built.rrt_paths)
                self.map.set_poly_path(built.poly_path)
                wps = built.waypoints if spec.kind == "waypoints" else None
                self.scene.set_static(spec.obstacles, wps, built.rrt_paths, built.poly_path)
                self.scene.reset(spec.obstacles)
                self.scene.set_static(spec.obstacles, wps, built.rrt_paths, built.poly_path)
                built.sim.start()
        except Exception as e:  # noqa: BLE001
            self._msg(f"Could not start: {e}")
            return

        self._static_dirty = False
        self._run_info = dict(dp=dp, fused=fused, n_obs=len(spec.obstacles), duration=spec.duration,
                              marks=self._scheduled_marks(), key=key)
        self.signals.clear()
        self.map.clear_live()
        self.map.set_read_only(True)
        self._running, self._paused = True, False
        self.btn_run.configure(text="Pause")
        self.telemetry.set_status("Flying", "good")
        self._msg(f"Running {CATALOG[key].label}")
        self._tick()

    def _launch_swarm(self, spec, dp, env, fused, key, vals):
        drones = []
        sep_on, sep_dist = self.inspector.swarm_separation()
        for i, wps in enumerate(self.inspector.swarm_paths()):
            aug, _ = plan_waypoints(wps, spec.obstacles, spec.avoid)
            wspecs = [WaypointSpec(w.x, w.y, w.z, math.radians(w.yaw), w.hover_time) for w in aug]
            ds = SingleDroneSim(i, wspecs, dp, env, key, 0.01,
                                controller=build_controller(key, dp, vals), use_estimator=fused)
            ds._state[0:2] = (aug[0].x, aug[0].y)
            ds.obstacles = list(spec.obstacles)
            use_obs = spec.avoid.uses_cbf and spec.obstacles
            kp, kd = ds._controller.position_gains
            cbf = CBFParams(r_safe=spec.avoid.safety_margin, k0=spec.avoid.cbf_k0,
                            k1=spec.avoid.cbf_k1, influence=max(spec.avoid.cbf_influence, sep_dist + 1.0),
                            nominal_kp=kp, nominal_kd=kd)
            if sep_on:
                ds.separation_filter = CBFSafetyFilter(spec.obstacles if use_obs else [], cbf)
                ds.separation_dist = sep_dist
            elif use_obs:
                flt = CBFSafetyFilter(spec.obstacles, cbf)
                ds.setpoint_filter = lambda t, s, sp, _f=flt: _f.filter(s[0:3], s[3:6], sp)
            drones.append(ds)
        for ds in drones:
            ds.peers = [q for q in drones if q is not ds]
        self._swarm = SwarmSimulation(drones, real_time=True, real_time_factor=self._speed)
        self._sim = None
        self._active_drone = 0
        self.scene.set_static(spec.obstacles, None, [], None)
        self.scene.reset(spec.obstacles)
        self._swarm.start()

    def _stop(self, quiet=False):
        if self._sim:
            self._sim.stop()
        if self._swarm:
            self._swarm.stop()
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        was = self._running
        self._running = self._paused = False
        self.btn_run.configure(text="▶  Run")
        self.map.set_read_only(False)
        if was and not quiet:
            self.telemetry.set_status("Stopped", "neutral")
            self._msg("Stopped")

    def _reset(self):
        self._stop(quiet=True)
        self._sim = self._swarm = None
        self.signals.clear()
        self.map.clear_live()
        self.scene.reset(self.obs.obstacles)
        self.telemetry.reset()
        self.timeline.set_progress(0, self.sidebar.get_mission_basics()[3], self._scheduled_marks())
        self._msg("Reset. Setup is unchanged.")

    def _on_speed(self, factor):
        self._speed = float(factor)
        if self._sim:
            self._sim.config.real_time_factor = self._speed
        if self._swarm:
            self._swarm._rtf = self._speed

    def _select_drone(self, idx):
        self._active_drone = idx

    # ── Update loop ──────────────────────────────────────────────────────────

    def _tick(self):
        if not self._running:
            return
        info = self._run_info
        if self._swarm:
            self._tick_swarm(info)
            alive = self._swarm._running
        else:
            self._tick_single(info)
            alive = self._sim.is_running
        if not alive and not self._paused:
            self._finish()
            return
        self._job = self.root.after(80, self._tick)

    def _tick_single(self, info):
        sim = self._sim
        state, rotors, sp = sim.state, sim.rotor_speeds, sim.setpoint
        m = sim.metrics
        events = list(sim.event_log)
        self.telemetry.update(state, m, rotors, sim.rotor_effectiveness, events,
                              info["dp"].omega_max, info["n_obs"], info["fused"])
        self.timeline.set_progress(m.sim_time, info["duration"], info["marks"])
        if m.error_count and self.status_lbl.cget("text") != m.last_error:
            self._msg(m.last_error[:90])
        if self._tab == "scene":
            self.scene.update([state], [rotors], [sp])
        elif self._tab == "map":
            d = sim.logger.get_tail(1500)
            if "x" in d:
                self.map.set_live([(state[0], state[1], state[8])], [np.column_stack((d["x"], d["y"]))])
        elif self._tab == "plots":
            self.signals.update(sim.logger, events, m.has_disturbance_est, info["fused"])

    def _tick_swarm(self, info):
        sw = self._swarm
        states, rotors, sps = sw.get_states(), sw.get_rotor_speeds(), sw.get_setpoints()
        mets, logs = sw.get_metrics(), sw.get_loggers()
        i = min(self._active_drone, len(states) - 1)
        self.telemetry.update(states[i], mets[i], rotors[i], np.ones(4), [],
                              info["dp"].omega_max, info["n_obs"], info["fused"])
        self.timeline.set_progress(sw.time, info["duration"], info["marks"])
        if self._tab == "scene":
            self.scene.update(states, rotors, sps)
        elif self._tab == "map":
            trails = []
            for lg in logs:
                d = lg.get_tail(1500)
                trails.append(np.column_stack((d["x"], d["y"])) if "x" in d else np.zeros((0, 2)))
            self.map.set_live([(s[0], s[1], s[8]) for s in states], trails)
        elif self._tab == "plots":
            self.signals.update(logs[i], [], False, info["fused"])
        if sw.time >= info["duration"]:
            sw.stop()

    def _finish(self):
        m = (self._sim.metrics if self._sim else self._swarm.get_metrics()[0])
        self._running = False
        self.btn_run.configure(text="▶  Run")
        self.map.set_read_only(False)
        crashed = m.is_crashed
        self.telemetry.set_status("Crashed" if crashed else "Finished", "bad" if crashed else "accent")
        self._msg(f"{'Crashed' if crashed else 'Finished'} at {m.sim_time:.1f} s. RMSE {m.rmse_pos:.2f} m")
        self.timeline.set_progress(m.sim_time, self._run_info["duration"], self._run_info["marks"])

    # ── Export ───────────────────────────────────────────────────────────────

    def _current_logger(self):
        if self._sim:
            return self._sim.logger
        if self._swarm:
            return self._swarm.get_loggers()[min(self._active_drone, self._swarm.n_drones - 1)]
        return None

    def _export(self, kind):
        lg = self._current_logger()
        if lg is None or lg.n_samples < 2:
            self._msg("Nothing to export yet. Run a flight first.")
            return
        ext = ".csv" if kind == "csv" else ".mat"
        path = filedialog.asksaveasfilename(defaultextension=ext, initialfile=f"dronsim_flight{ext}",
                                            filetypes=[(kind.upper(), f"*{ext}")])
        if not path:
            return
        try:
            (lg.export_csv if kind == "csv" else lg.export_mat)(path)
            self._msg(f"Saved {lg.n_samples} samples to {path.split('/')[-1]}")
        except Exception as e:  # noqa: BLE001
            self._msg(f"Export failed: {e}")

    def _export_csv(self):
        self._export("csv")

    def _export_mat(self):
        self._export("mat")

    # ── About ────────────────────────────────────────────────────────────────

    def _show_about(self):
        win = tk.Toplevel(self.root)
        win.title("About DRONSIM")
        win.configure(bg=C["surface"])
        win.resizable(False, False)
        win.transient(self.root)
        body = tk.Frame(win, bg=C["surface"])
        body.pack(fill="both", expand=True, padx=28, pady=24)
        tk.Label(body, text="DRONSIM", bg=C["surface"], fg=C["text"], font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(body, text=f"Version {VERSION}  ·  by {AUTHOR}", bg=C["surface"], fg=C["accent"],
                 font=F["body"]).pack(anchor="w", pady=(0, 14))
        rows = [("Physics", "6-DOF Newton-Euler, RK4 at 100 Hz, wind, Dryden turbulence, ground effect"),
                ("Controllers", "PID, sliding mode, PID+SMC, MPC, ADRC, IFT, geometric SE(3), RL (PPO)"),
                ("Sensing", "IMU, GPS and barometer with a Kalman filter and complementary attitude filter"),
                ("Avoidance", "APF, RRT*, and a control barrier function safety filter"),
                ("Scenarios", "Rotor faults, wind gusts, swarm flights, controller comparison")]
        for k, v in rows:
            r = tk.Frame(body, bg=C["surface"])
            r.pack(fill="x", pady=3)
            r.grid_columnconfigure(1, weight=1)
            tk.Label(r, text=k, bg=C["surface"], fg=C["text3"], font=F["small_bold"], width=11,
                     anchor="nw").grid(row=0, column=0, sticky="nw")
            tk.Label(r, text=v, bg=C["surface"], fg=C["text"], font=F["body"], justify="left",
                     wraplength=420, anchor="w").grid(row=0, column=1, sticky="w")
        Btn(body, "Close", win.destroy, kind="secondary", padx=24).pack(anchor="e", pady=(18, 0))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + 120
        win.geometry(f"+{x}+{y}")

    def _on_close(self):
        self._stop(quiet=True)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DroneSimApp().run()
