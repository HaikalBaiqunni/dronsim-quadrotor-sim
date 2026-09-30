"""
drone_sim/gui/telemetry.py
===========================
Right-hand live telemetry column and the bottom timeline bar.
"""

import tkinter as tk
from typing import Callable, List, Optional

import numpy as np

from .theme import C, F, SERIES
from .widgets import (BarMeter, Btn, Card, Chip, KPI, ScrollFrame, Segmented, WrapLabel,
                      caption, combo, divider)


class TelemetryPanel(tk.Frame):
    def __init__(self, parent, on_select_drone: Optional[Callable] = None):
        super().__init__(parent, bg=C["surface"], width=272)
        self.pack_propagate(False)
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        p = self.scroll.inner

        head = tk.Frame(p, bg=C["surface"])
        head.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(head, text="Telemetry", bg=C["surface"], fg=C["text"], font=F["title"]).pack(side="left")
        self.status = Chip(head, "Ready", "neutral")
        self.status.pack(side="right")

        self._drone_row = tk.Frame(p, bg=C["surface"])
        self._drone_combo = combo(self._drone_row, ["Drone 1"], "Drone 1",
                                  lambda v: on_select_drone and on_select_drone(int(v.split()[1]) - 1), width=12)
        self._drone_combo.pack(fill="x", padx=12)

        grid = tk.Frame(p, bg=C["surface"])
        grid.pack(fill="x", padx=12, pady=(6, 0))
        grid.grid_columnconfigure(0, weight=1, uniform="k")
        grid.grid_columnconfigure(1, weight=1, uniform="k")
        self.k = {}
        specs = [("err", "Error", "m"), ("rmse", "RMSE", "m"),
                 ("speed", "Speed", "m/s"), ("alt", "Altitude", "m"),
                 ("tilt", "Tilt", "deg"), ("clear", "Clearance", "m"),
                 ("dist", "Disturbance", "m/s²"), ("est", "Est. error", "m")]
        for i, (key, label, unit) in enumerate(specs):
            card = KPI(grid, label, unit)
            card.grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3)
            self.k[key] = card

        caption(p, "Pose", pady=(12, 2)).configure(padx=12)
        pose = tk.Frame(p, bg=C["surface2"], highlightthickness=1, highlightbackground=C["border"])
        pose.pack(fill="x", padx=12)
        self._pose = {}
        for i, (name, unit) in enumerate([("X", "m"), ("Y", "m"), ("Z", "m"),
                                          ("Roll", "°"), ("Pitch", "°"), ("Yaw", "°")]):
            pose.grid_columnconfigure(i % 3, weight=1, uniform="pose")
            cell = tk.Frame(pose, bg=C["surface2"])
            cell.grid(row=i // 3, column=i % 3, sticky="ew", padx=6, pady=5)
            tk.Label(cell, text=f"{name} ({unit})", bg=C["surface2"], fg=C["text3"], font=F["small"],
                     anchor="w").pack(fill="x")
            v = tk.Label(cell, text="—", bg=C["surface2"], fg=C["text"], font=F["mono"], anchor="w")
            v.pack(fill="x")
            self._pose[name] = v

        caption(p, "Rotors", pady=(12, 2)).configure(padx=12)
        rot = tk.Frame(p, bg=C["surface"])
        rot.pack(fill="x", padx=12)
        self._rotors = []
        for i in range(4):
            m = BarMeter(rot, f"Ω{i + 1}", color=SERIES[i], bg=C["surface"])
            m.pack(fill="x", pady=1)
            self._rotors.append(m)

        caption(p, "Events", pady=(12, 2)).configure(padx=12)
        self._events_host = tk.Frame(p, bg=C["surface"])
        self._events_host.pack(fill="x", padx=12, pady=(0, 14))
        self._last_events = None
        self.reset()

    def set_drone_count(self, n: int):
        if n > 1:
            self._drone_combo.configure(values=[f"Drone {i + 1}" for i in range(n)])
            self._drone_combo.set("Drone 1")
            self._drone_row.pack(fill="x", pady=(0, 2), before=self.k["err"].master)
        else:
            self._drone_row.pack_forget()

    def reset(self):
        for k in self.k.values():
            k.set("—")
        for v in self._pose.values():
            v.configure(text="—")
        for m in self._rotors:
            m.set(0, "")
        self.status.set("Ready", "neutral")
        self._show_events([])

    def set_status(self, text: str, tone: str = "neutral"):
        self.status.set(text, tone)

    def _show_events(self, events):
        if events == self._last_events:
            return
        self._last_events = list(events)
        for w in self._events_host.winfo_children():
            w.destroy()
        if not events:
            WrapLabel(self._events_host, "Nothing has happened yet.", fg=C["text3"], bg=C["surface"]).pack(fill="x")
        for t, text in events[-6:]:
            row = tk.Frame(self._events_host, bg=C["surface"])
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(1, weight=1)
            tk.Label(row, text=f"{t:5.1f} s", bg=C["surface"], fg=C["amber"], font=F["mono_small"],
                     anchor="w").grid(row=0, column=0, sticky="nw", padx=(0, 8))
            WrapLabel(row, text, fg=C["text"], bg=C["surface"], font=F["small"]).grid(row=0, column=1, sticky="ew")

    def update(self, state, metrics, rotors, effectiveness, events, omega_max=1000.0,
               n_obstacles=0, fused=False):
        m = metrics
        speed = float(np.linalg.norm(state[3:6]))
        tone_err = "good" if m.pos_error < 0.3 else ("warn" if m.pos_error < 1.0 else "bad")
        self.k["err"].set(f"{m.pos_error:.2f}", tone_err)
        self.k["rmse"].set(f"{m.rmse_pos:.2f}")
        self.k["speed"].set(f"{speed:.1f}")
        self.k["alt"].set(f"{state[2]:.2f}")
        self.k["tilt"].set(f"{m.max_tilt_deg:.0f}", "bad" if m.max_tilt_deg > 45 else None)
        if n_obstacles and np.isfinite(m.clearance):
            self.k["clear"].set(f"{m.clearance:.2f}", "bad" if m.clearance < 0.15 else ("warn" if m.clearance < 0.5 else "good"))
        else:
            self.k["clear"].set("—")
        self.k["dist"].set(f"{m.disturbance_est:.2f}" if m.has_disturbance_est else "n/a")
        self.k["est"].set(f"{m.est_error:.3f}" if fused else "n/a")

        for name, val in zip(("X", "Y", "Z"), state[0:3]):
            self._pose[name].configure(text=f"{val:+.2f}")
        for name, val in zip(("Roll", "Pitch", "Yaw"), np.degrees(state[6:9])):
            self._pose[name].configure(text=f"{val:+.1f}")
        for i, mtr in enumerate(self._rotors):
            mtr.set(rotors[i] / omega_max, f"{rotors[i]:.0f}", dim=effectiveness[i] < 0.999)

        if m.error_count:
            self.status.set("Error", "bad")
        elif m.is_crashed:
            self.status.set("Crashed", "bad")
        elif not m.is_stable and m.sim_time > 1.0:
            self.status.set("Unstable", "warn")
        elif m.finished:
            self.status.set("Finished", "accent")
        self._show_events(events)


class Timeline(tk.Frame):
    """Bottom bar: progress with event markers, speed, export."""

    def __init__(self, parent, on_speed: Callable, on_export_csv: Callable, on_export_mat: Callable):
        super().__init__(parent, bg=C["surface2"], height=52)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)
        self.time_lbl = tk.Label(self, text="0.0 / 60 s", bg=C["surface2"], fg=C["text"],
                                 font=F["mono"], width=13, anchor="w")
        self.time_lbl.grid(row=0, column=0, padx=(14, 8), sticky="w")
        self.bar = tk.Canvas(self, height=24, bg=C["surface2"], highlightthickness=0)
        self.bar.grid(row=0, column=1, sticky="ew", pady=14)
        self.bar.bind("<Configure>", lambda e: self._draw())
        self._t, self._dur, self._marks = 0.0, 60.0, []
        self.speed = Segmented(self, [(0.5, "0.5×"), (1.0, "1×"), (2.0, "2×"), (4.0, "4×")], 1.0,
                               on_speed, bg=C["surface2"], font=F["small_bold"], pady=4)
        self.speed.grid(row=0, column=2, padx=10)
        Btn(self, "Export CSV", on_export_csv, kind="secondary", padx=10, pady=5,
            font=F["small_bold"]).grid(row=0, column=3, padx=(0, 6))
        Btn(self, "Export MAT", on_export_mat, kind="secondary", padx=10, pady=5,
            font=F["small_bold"]).grid(row=0, column=4, padx=(0, 14))

    def set_progress(self, t: float, duration: float, marks: List[tuple]):
        self._t, self._dur, self._marks = t, max(duration, 1e-6), marks
        self.time_lbl.configure(text=f"{t:5.1f} / {duration:.0f} s")
        self._draw()

    def _draw(self):
        c = self.bar
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            return
        c.create_rectangle(0, h / 2 - 3, w, h / 2 + 3, fill=C["surface3"], outline="")
        frac = min(1.0, self._t / self._dur)
        c.create_rectangle(0, h / 2 - 3, w * frac, h / 2 + 3, fill=C["accent"], outline="")
        for t, text in self._marks:
            x = min(w - 1, w * t / self._dur)
            c.create_line(x, 2, x, h - 2, fill=C["red"], width=2)
