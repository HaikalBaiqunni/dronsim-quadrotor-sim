"""
drone_sim/gui/plots.py
=======================
Matplotlib panels for the centre area:

  ScenePlot    3D view with the animated drone model, obstacles and routes
  SignalPlots  live time-series (position, attitude, rotors, effort,
               estimator error, disturbance observer)
  ComparePanel run several controllers on the same mission and rank them
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..core.catalog import CATALOG
from ..core.compare import CompareResult, run_comparison
from .drone_3d import DroneVisualizer3D, SwarmVisualizer3D
from .obstacle_render import Obstacle3DRenderer
from .theme import C, F, SERIES
from .widgets import (Btn, Card, Chip, ScrollFrame, Segmented, SliderField, WrapLabel,
                      caption, check, divider)
from ..core.catalog import Param

FIG_BG = C["surface"]
AX_BG = "#10151C"

plt.rcParams.update({
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG, "axes.edgecolor": C["border"],
    "axes.labelcolor": C["text2"], "axes.titlecolor": C["text"], "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": C["text3"], "ytick.color": C["text3"],
    "xtick.labelsize": 9, "ytick.labelsize": 9, "grid.color": "#232B36", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "text.color": C["text"], "legend.facecolor": C["surface2"],
    "legend.edgecolor": C["border"], "legend.fontsize": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
})


def _style_ax(ax, ylabel="", xlabel=""):
    ax.grid(True, alpha=0.6)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    for sp in ax.spines.values():
        sp.set_color(C["border"])


# ── 3D scene ─────────────────────────────────────────────────────────────────

class ScenePlot(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=FIG_BG)
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=FIG_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.set_facecolor(AX_BG)
        for pane in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            pane.set_pane_color((0.06, 0.08, 0.11, 1.0))
            pane._axinfo["grid"]["color"] = (0.2, 0.25, 0.32, 0.6)
        self.ax.set_xlabel("X (m)", labelpad=2)
        self.ax.set_ylabel("Y (m)", labelpad=2)
        self.ax.set_zlabel("Z (m)", labelpad=2)
        self.ax.tick_params(labelsize=8)
        self.vis = DroneVisualizer3D(self.ax)
        self.swarm_vis: Optional[SwarmVisualizer3D] = None
        self.obs_renderer = Obstacle3DRenderer()
        self._overlays = []
        self._lim = np.array([-4.0, 4.0, -4.0, 4.0, 0.0, 4.0])
        self._lim_applied = None
        self._ground = None
        self._start = (0.0, 0.0, 0.1)
        self._apply_limits(force=True)

    def _apply_limits(self, force=False):
        if force or self._lim_applied is None or not np.allclose(self._lim, self._lim_applied):
            x0, x1, y0, y1, z0, z1 = self._lim
            span = max(x1 - x0, y1 - y0, z1 - z0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.ax.set_xlim(cx - span / 2, cx + span / 2)
            self.ax.set_ylim(cy - span / 2, cy + span / 2)
            self.ax.set_zlim(0, span)
            self.ax.set_box_aspect((1, 1, 1))
            self._lim_applied = self._lim.copy()
            if self._ground is not None:
                try:
                    self._ground.remove()
                except Exception:
                    pass
            g = np.array([-1, 1]) * span / 2
            xx, yy = np.meshgrid(cx + g, cy + g)
            self._ground = self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.07, color=C["green"])

    def _grow(self, pts: np.ndarray, pad=1.5):
        pts = np.atleast_2d(pts)
        lo = pts.min(axis=0) - pad
        hi = pts.max(axis=0) + pad
        new = self._lim.copy()
        new[0], new[2], new[4] = min(new[0], lo[0]), min(new[2], lo[1]), 0.0
        new[1], new[3], new[5] = max(new[1], hi[0]), max(new[3], hi[1]), max(new[5], hi[2])
        self._lim = new

    def _hide_swarm(self):
        if self.swarm_vis:
            self.swarm_vis.hide()
            self.swarm_vis = None

    def show_idle(self):
        self._hide_swarm()
        st = np.zeros(12)
        st[0:3] = self._start
        self.vis.update(st, np.full(4, 500.0), np.array([*self._start, 0.0]), dt=0.0)
        self.vis.reset_trail()
        self.canvas.draw_idle()

    def set_static(self, obstacles, waypoints=None, rrt_paths=None, poly_path=None, start=(0.0, 0.0, 0.1)):
        self._start = tuple(start)
        for a in self._overlays:
            try:
                a.remove()
            except Exception:
                pass
        self._overlays.clear()
        self.obs_renderer.draw(self.ax, obstacles)
        pts = []
        if waypoints and len(waypoints) >= 2:
            xs, ys, zs = zip(*[(w.x, w.y, max(w.z, 0)) for w in waypoints])
            ln, = self.ax.plot(xs, ys, zs, "--", color=C["accent"], lw=1.2, alpha=0.7)
            mk, = self.ax.plot(xs, ys, zs, "D", color=C["accent"], ms=5, alpha=0.9)
            self._overlays += [ln, mk]
            pts += list(zip(xs, ys, zs))
        for path in rrt_paths or []:
            ln, = self.ax.plot(path[:, 0], path[:, 1], np.maximum(path[:, 2], 0), "--",
                               color=C["purple"], lw=1.8, alpha=0.9)
            self._overlays.append(ln)
        if poly_path is not None:
            ln, = self.ax.plot(poly_path[:, 0], poly_path[:, 1], poly_path[:, 2], "-",
                               color=C["green"], lw=1.8, alpha=0.7)
            self._overlays.append(ln)
            pts += [tuple(p) for p in poly_path[::20]]
        for o in obstacles:
            pts.append((o.cx, o.cy, getattr(o, "z_top", getattr(o, "cz", 1) * 2)))
        if pts:
            self._grow(np.array(pts))
        self._apply_limits()
        self.show_idle()

    def redraw_obstacles(self, obstacles):
        self.obs_renderer.draw(self.ax, obstacles)
        self.canvas.draw_idle()

    def reset(self, obstacles):
        self.vis.reset_trail()
        if self.swarm_vis:
            self.swarm_vis.reset_trails()
        self.obs_renderer.draw(self.ax, obstacles)
        self.show_idle()

    def update(self, states, rotors, setpoints, dt=0.08):
        if len(states) == 1:
            self._hide_swarm()
            self.vis.update(states[0], rotors[0], setpoints[0], dt=dt)
        else:
            self.vis.hide()                     # the idle single-drone model
            if self.swarm_vis is None or self.swarm_vis.n != len(states):
                self._hide_swarm()
                self.swarm_vis = SwarmVisualizer3D(self.ax, len(states))
            self.swarm_vis.update(states, rotors, setpoints, dt)
        self._grow(np.array([s[0:3] for s in states]), pad=1.0)
        self._apply_limits()
        self.canvas.draw_idle()


# ── Live signals ─────────────────────────────────────────────────────────────

class SignalPlots(tk.Frame):
    WINDOW = 1500          # samples shown (15 s at 100 Hz)
    VIEWS = [("pos", "Position"), ("att", "Attitude"), ("rot", "Rotors"),
             ("eff", "Effort"), ("est", "Estimator"), ("obs", "Observer")]

    def __init__(self, parent):
        super().__init__(parent, bg=FIG_BG)
        bar = tk.Frame(self, bg=FIG_BG)
        bar.pack(fill="x", padx=12, pady=(10, 4))
        self.bar = Segmented(bar, self.VIEWS, "pos", self._switch, bg=FIG_BG, font=F["small_bold"], pady=5)
        self.bar.pack(fill="x")
        self.note = WrapLabel(self, "", fg=C["amber"], bg=FIG_BG, font=F["small"])
        self.note.pack(fill="x", padx=14)
        self.host = tk.Frame(self, bg=FIG_BG)
        self.host.pack(fill="both", expand=True)
        self.view = "pos"
        self._figs: Dict[str, tuple] = {}
        for key, _ in self.VIEWS:
            self._build(key)
        self._switch("pos")
        self._events: List[tuple] = []

    def _fig(self, key, nrows, sharex=True):
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=FIG_BG)
        canvas = FigureCanvasTkAgg(fig, master=self.host)
        axes = fig.subplots(nrows, 1, sharex=sharex) if nrows > 1 else [fig.add_subplot(111)]
        axes = list(np.atleast_1d(axes))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.11, hspace=0.35)
        self._figs[key] = (fig, canvas, axes, [])
        return fig, canvas, axes

    def _build(self, key):
        if key == "pos":
            fig, cv, axes = self._fig(key, 3)
            lines = []
            for ax, lab, col in zip(axes, ["X (m)", "Y (m)", "Z (m)"], SERIES[:3]):
                _style_ax(ax, lab)
                a, = ax.plot([], [], color=col, label="Actual")
                b, = ax.plot([], [], "--", color=col, alpha=0.6, label="Setpoint")
                ax.legend(loc="upper right", ncol=2)
                lines.append((a, b))
            axes[-1].set_xlabel("Time (s)")
        elif key == "att":
            fig, cv, axes = self._fig(key, 3)
            lines = []
            for ax, lab, col in zip(axes, ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"], [C["amber"], C["purple"], C["green"]]):
                _style_ax(ax, lab)
                a, = ax.plot([], [], color=col)
                lines.append((a,))
            axes[-1].set_xlabel("Time (s)")
        elif key == "rot":
            fig, cv, axes = self._fig(key, 1)
            ax = axes[0]
            _style_ax(ax, "Rotor speed (rad/s)", "Time (s)")
            lines = [(ax.plot([], [], color=SERIES[i], label=f"Rotor {i + 1}")[0],) for i in range(4)]
            ax.legend(loc="upper right", ncol=4)
        elif key == "eff":
            fig, cv, axes = self._fig(key, 2)
            _style_ax(axes[0], "Tracking cost")
            _style_ax(axes[1], "Position error (m)", "Time (s)")
            lines = [(axes[0].plot([], [], color=C["coral"])[0],), (axes[1].plot([], [], color=C["amber"])[0],)]
        elif key == "est":
            fig, cv, axes = self._fig(key, 1)
            ax = axes[0]
            _style_ax(ax, "Estimate minus truth (m)", "Time (s)")
            lines = [(ax.plot([], [], color=SERIES[i], label=lab)[0],) for i, lab in enumerate(["X", "Y", "Z"])]
            ax.legend(loc="upper right", ncol=3)
        else:
            fig, cv, axes = self._fig(key, 1)
            ax = axes[0]
            _style_ax(ax, "Estimated disturbance (m/s²)", "Time (s)")
            lines = [(ax.plot([], [], color=SERIES[i], label=lab)[0],) for i, lab in enumerate(["X", "Y", "Z"])]
            ax.legend(loc="upper right", ncol=3)
        self._figs[key] = (fig, cv, axes, lines)

    def _switch(self, key):
        self.view = key
        for _, (fig, cv, _, _) in self._figs.items():
            cv.get_tk_widget().pack_forget()
        self._figs[key][1].get_tk_widget().pack(fill="both", expand=True)
        self.bar.set(key)
        self._figs[key][1].draw_idle()

    def clear(self):
        for key, (fig, cv, axes, lines) in self._figs.items():
            for group in lines:
                for ln in group:
                    ln.set_data([], [])
            for ax in axes:
                for v in getattr(ax, "_ev_lines", []):
                    v.remove()
                ax._ev_lines = []
            cv.draw_idle()
        self.note.configure(text="")

    def update(self, logger, events: List[tuple], controller_has_observer: bool, fused: bool):
        if logger.n_samples < 2:
            return
        d = logger.get_tail(self.WINDOW)
        t = d["time"]
        key = self.view
        fig, cv, axes, lines = self._figs[key]
        note = ""
        if key == "pos":
            for i, (k, sk) in enumerate([("x", "x_d"), ("y", "y_d"), ("z", "z_d")]):
                lines[i][0].set_data(t, d[k])
                lines[i][1].set_data(t, d[sk])
        elif key == "att":
            for i, k in enumerate(["phi", "theta", "psi"]):
                lines[i][0].set_data(t, d[k])
        elif key == "rot":
            for i, k in enumerate(["w1", "w2", "w3", "w4"]):
                lines[i][0].set_data(t, d[k])
        elif key == "eff":
            lines[0][0].set_data(t, d["control_cost"])
            lines[1][0].set_data(t, np.sqrt(d["ex"] ** 2 + d["ey"] ** 2 + d["ez"] ** 2))
        elif key == "est":
            for i, k in enumerate(["est_ex", "est_ey", "est_ez"]):
                lines[i][0].set_data(t, d[k])
            if not fused:
                note = "Turn on Sensor fusion in the World section to see estimator error."
        elif key == "obs":
            for i, k in enumerate(["dist_x", "dist_y", "dist_z"]):
                lines[i][0].set_data(t, d[k])
            if not controller_has_observer:
                note = "The selected controller has no disturbance observer. Try ADRC."
        self.note.configure(text=note)
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t[0], max(t[-1], t[0] + 1e-3))
            for v in getattr(ax, "_ev_lines", []):
                v.remove()
            ax._ev_lines = [ax.axvline(te, color=C["red"], ls=":", lw=1.2, alpha=0.9)
                            for te, _ in events if t[0] <= te <= t[-1]]
        cv.draw_idle()


# ── Compare ──────────────────────────────────────────────────────────────────

class ComparePanel(tk.Frame):
    DEFAULTS = ["PID", "SMC", "ADRC", "GEO"]

    def __init__(self, parent, scenario: Callable[[], dict]):
        """``scenario()`` returns kwargs for run_comparison (spec, dp, env, ...)."""
        super().__init__(parent, bg=FIG_BG)
        self.scenario = scenario
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancel = False
        self._results: List[CompareResult] = []

        side = tk.Frame(self, bg=C["surface"], width=250)
        side.pack(side="left", fill="y", padx=(10, 6), pady=10)
        side.pack_propagate(False)
        caption(side, "Controllers to compare", pady=(12, 4)).configure(padx=12)
        self._vars: Dict[str, tk.IntVar] = {}
        for key, spec in CATALOG.items():
            v = tk.IntVar(value=int(key in self.DEFAULTS))
            self._vars[key] = v
            check(side, spec.label + ("   NEW" if spec.new else "   EXPERIMENTAL" if spec.experimental else ""), v, bg=C["surface"]).pack(fill="x", padx=12)
        divider(side, pady=8)
        self.dur = SliderField(side, Param("d", "Fly for", 20, 5, 90, 5, unit="s"), 20, None, bg=C["surface"])
        self.dur.pack(fill="x", padx=12, pady=4)
        self.run_btn = Btn(side, "Run comparison", self._run, kind="primary")
        self.run_btn.pack(fill="x", padx=12, pady=(8, 4))
        self.status = WrapLabel(side, "Flies the mission from the Mission, World and Events sections "
                                      "with each controller, faster than real time.",
                                fg=C["text3"], bg=C["surface"])
        self.status.pack(fill="x", padx=12, pady=(4, 0))
        self.bar = ttk.Progressbar(side, mode="determinate")
        self.bar.pack(fill="x", padx=12, pady=8)

        main = tk.Frame(self, bg=FIG_BG)
        main.pack(side="left", fill="both", expand=True, pady=10, padx=(0, 10))
        WrapLabel(main, "RMSE and Max are position error. Effort is the sum of squared rotor speeds "
                        "times time, in millions of rad². Tilt is the peak roll or pitch. "
                        "Clear is the closest approach to an obstacle.",
                  fg=C["text3"], bg=FIG_BG).pack(side="bottom", fill="x", pady=(6, 0))
        self.fig = Figure(figsize=(7, 3.4), dpi=100, facecolor=FIG_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.16)
        _style_ax(self.ax, "Position error (m)", "Time (s)")
        self.ax.set_title("Tracking error")

        cols = ("rank", "ctrl", "rmse", "maxerr", "energy", "tilt", "clear", "result")
        heads = ("#", "Controller", "RMSE m", "Max m", "Effort", "Tilt °", "Clear m", "Result")
        widths = (34, 130, 70, 70, 70, 60, 70, 74)
        tf = tk.Frame(main, bg=FIG_BG)
        tf.pack(side="bottom", fill="x", pady=(8, 0))
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=6, selectmode="none")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, minwidth=int(w * 0.75), anchor="w" if c in ("ctrl", "result") else "e",
                             stretch=True)
        self.tree.pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def selected(self) -> List[str]:
        return [k for k, v in self._vars.items() if v.get()]

    def _run(self):
        if self._thread and self._thread.is_alive():
            self._cancel = True
            self.status.configure(text="Stopping…")
            return
        keys = self.selected()
        if len(keys) < 2:
            self.status.configure(text="Pick at least two controllers.")
            return
        kw = self.scenario()
        kw["keys"] = keys
        kw["duration"] = float(self.dur.get())
        self._cancel = False
        self.run_btn.configure(text="Stop")
        self.run_btn.restyle("danger")
        self.bar.configure(maximum=len(keys), value=0)
        self._results = []
        self.tree.delete(*self.tree.get_children())

        def work():
            try:
                res = run_comparison(
                    progress=lambda i, n, lbl: self._q.put(("progress", i, n, lbl)),
                    cancel=lambda: self._cancel, **kw)
                self._q.put(("done", res))
            except Exception as e:  # noqa: BLE001
                self._q.put(("error", str(e)))
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self.after(150, self._poll)

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "progress":
                    _, i, n, lbl = msg
                    self.bar.configure(value=i)
                    if lbl:
                        self.status.configure(text=f"Flying {lbl}  ({i + 1} of {n})")
                elif msg[0] == "done":
                    self._finish(msg[1])
                    return
                elif msg[0] == "error":
                    self.status.configure(text=f"Comparison failed: {msg[1]}")
                    self._reset_btn()
                    return
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _reset_btn(self):
        self.run_btn.configure(text="Run comparison")
        self.run_btn.restyle("primary")

    def _finish(self, results: List[CompareResult]):
        self._reset_btn()
        self.bar.configure(value=self.bar.cget("maximum"))
        self._results = results
        if not results:
            self.status.configure(text="No results.")
            return
        self.status.configure(text=f"Done. {len(results)} controllers flown.")
        self.ax.clear()
        _style_ax(self.ax, "Position error (m)", "Time (s)")
        self.ax.set_title("Tracking error")
        ranked = sorted(results, key=lambda r: (r.crashed, r.rmse))
        colour = {r.key: SERIES[i % len(SERIES)] for i, r in enumerate(results)}
        for r in results:
            self.ax.plot(r.time, r.error, color=colour[r.key], label=r.label)
        self.ax.legend(loc="upper right", ncol=2)
        self.canvas.draw_idle()
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(ranked):
            tag = f"c_{r.key}"
            self.tree.tag_configure(tag, foreground=colour[r.key])
            clear = "—" if r.min_clearance > 50 else f"{r.min_clearance:.2f}"
            self.tree.insert("", "end", tags=(tag,), values=(
                i + 1, r.label, f"{r.rmse:.3f}", f"{r.max_error:.3f}", f"{r.energy / 1e6:.1f}",
                f"{r.max_tilt:.0f}", clear, "Crashed" if r.crashed else "Flew"))
