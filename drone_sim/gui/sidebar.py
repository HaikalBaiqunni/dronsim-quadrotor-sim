"""
drone_sim/gui/sidebar.py
=========================
Left-hand configuration column. Collapsible sections (one open at a time):

  Vehicle · Mission · Controller · World · Events

Each section shows a one-line summary while collapsed, so the whole setup is
readable at a glance.
"""

import tkinter as tk
from typing import Callable, Dict, List

import numpy as np

from ..core.catalog import CATALOG, FAMILIES, Param, ControllerSpec, default_values
from ..core.dynamics import DroneParams
from ..core.environment import EnvironmentParams, WindCondition
from ..core.mission import PRESETS
from ..core.simulation import ScheduledEvent
from .theme import C, F
from .widgets import (Btn, Card, Chip, LabeledEntry, RadioList, ScrollFrame, Section,
                      Segmented, SliderField, WrapLabel, caption, check, combo, divider)

WIND_LEVELS = [("CALM", "Calm  (under 1 m/s)"), ("LIGHT", "Light  (1 to 3 m/s)"),
               ("MODERATE", "Moderate  (3 to 6 m/s)"), ("STRONG", "Strong  (6 to 10 m/s)"),
               ("STORM", "Storm  (over 10 m/s)")]
PRESET_LABELS = {"hover": "Hover in place", "circle": "Circle", "figure8": "Figure-8", "spiral": "Climbing spiral"}
SENSOR_MODES = [("ideal", "Ideal sensors", "The controller sees the exact state."),
                ("noisy", "Raw noisy sensors", "GPS and gyro noise goes straight into the controller."),
                ("fused", "Sensor fusion", "IMU + GPS + barometer through a Kalman filter. Most realistic.")]


class Sidebar(tk.Frame):
    ORDER = ["vehicle", "mission", "controller", "world", "events"]

    def __init__(self, parent, on_open_map: Callable, on_change: Callable,
                 on_section: Callable = None):
        super().__init__(parent, bg=C["surface"], width=350)
        self.pack_propagate(False)
        self.on_open_map, self.on_change = on_open_map, on_change
        self.on_section = on_section or (lambda key: None)
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        self.host = self.scroll.inner
        self.sections: Dict[str, Section] = {}

        self.values: Dict[str, dict] = {k: default_values(k) for k in CATALOG}
        self.ctrl_key = "PID"
        self.level = "basic"
        self.events: List[ScheduledEvent] = []
        self.waypoint_count = 0

        self._build_vehicle()
        self._build_mission()
        self._build_controller()
        self._build_world()
        self._build_events()
        self.sections["controller"].expand()

    # ── Section helpers ──────────────────────────────────────────────────────

    def _section(self, key, title, badge=""):
        s = Section(self.host, title, on_toggle=self._accordion, badge=badge)
        s.pack(fill="x", padx=12, pady=(10 if key == "vehicle" else 6, 0))
        s.body.configure(bg=C["surface2"])
        self.sections[key] = s
        return s

    def _accordion(self, section, opened):
        if opened:
            for s in self.sections.values():
                if s is not section:
                    s.collapse()
            self.on_section(next(k for k, v in self.sections.items() if v is section))

    def open(self, key):
        self.sections[key].expand()
        self.after(50, lambda: self.scroll.scroll_to(self.sections[key]))

    def _changed(self, *_):
        self.refresh_summaries()
        self.on_change()

    # ── Vehicle ──────────────────────────────────────────────────────────────

    def _build_vehicle(self):
        s = self._section("vehicle", "Vehicle")
        b = s.body
        WrapLabel(b, "250 mm research quadrotor, X configuration. Inertia scales with mass.",
                  bg=C["surface2"], fg=C["text2"]).pack(fill="x", pady=(0, 6))
        self.mass = SliderField(b, Param("mass", "Mass", 0.5, 0.3, 1.2, 0.05, unit="kg"), 0.5,
                                lambda k, v: self._changed(), bg=C["surface2"])
        self.mass.pack(fill="x", pady=4)
        self._veh_info = WrapLabel(b, "", bg=C["surface2"], fg=C["text3"])
        self._veh_info.pack(fill="x", pady=(4, 0))

    def get_drone_params(self) -> DroneParams:
        m = float(self.mass.get())
        k = m / 0.5
        d = DroneParams()
        d.mass = m
        d.Ixx *= k
        d.Iyy *= k
        d.Izz *= k
        return d

    # ── Mission ──────────────────────────────────────────────────────────────

    def _build_mission(self):
        s = self._section("mission", "Mission")
        b = s.body
        self.kind = Segmented(b, [("preset", "Preset"), ("waypoints", "Waypoints")], "preset",
                              self._kind_changed, bg=C["surface2"])
        self.kind.pack(fill="x", pady=(0, 8))

        self._kind_host = tk.Frame(b, bg=C["surface2"])
        self._kind_host.pack(fill="x")
        self._preset_frame = tk.Frame(self._kind_host, bg=C["surface2"])
        self._preset_combo = combo(self._preset_frame, list(PRESET_LABELS.values()),
                                   PRESET_LABELS["hover"], lambda v: self._changed(), width=16)
        self._preset_combo.pack(fill="x")
        caption(self._preset_frame, "Hover point", bg=C["surface2"], pady=(10, 2))
        self.hover_fields = {}
        for label, val, unit in [("X", 0, "m"), ("Y", 0, "m"), ("Height Z", 2, "m"), ("Yaw", 0, "deg")]:
            f = LabeledEntry(self._preset_frame, label, val, unit=unit, bg=C["surface2"],
                             on_change=self._changed)
            f.pack(fill="x", pady=2)
            self.hover_fields[label] = f

        self._wp_frame = tk.Frame(self._kind_host, bg=C["surface2"])
        self._wp_info = WrapLabel(self._wp_frame, "", bg=C["surface2"], fg=C["text2"], font=F["body"])
        self._wp_info.pack(fill="x", pady=(0, 8))
        Btn(self._wp_frame, "Edit route on map", self.on_open_map, kind="primary").pack(fill="x")

        divider(b, pady=10)
        self.duration = SliderField(b, Param("duration", "Run time", 60, 10, 300, 5, unit="s",
                                             hint="The simulation stops by itself after this time."),
                                    60, lambda k, v: self._changed(), bg=C["surface2"])
        self.duration.pack(fill="x")
        self._kind_changed("preset", notify=False)

    def _kind_changed(self, kind, notify=True):
        self._preset_frame.pack_forget()
        self._wp_frame.pack_forget()
        (self._preset_frame if kind == "preset" else self._wp_frame).pack(fill="x")
        if notify:
            self._changed()

    def set_mission_kind(self, kind: str):
        if self.kind.get() != kind:
            self.kind.set(kind)
            self._kind_changed(kind, notify=False)

    def set_waypoint_summary(self, count: int, n_drones: int = 1):
        self.waypoint_count = count
        extra = f" for each of {n_drones} drones" if n_drones > 1 else ""
        self._wp_info.configure(
            text=f"{count} waypoint{'s' if count != 1 else ''}{extra}. Draw and edit them on the map.")
        self.refresh_summaries()

    def get_mission_basics(self):
        preset = next(k for k, v in PRESET_LABELS.items() if v == self._preset_combo.get())
        hf = self.hover_fields
        hover = (hf["X"].get_float(0), hf["Y"].get_float(0), hf["Height Z"].get_float(2),
                 hf["Yaw"].get_float(0))
        return self.kind.get(), preset, hover, float(self.duration.get())

    # ── Controller ───────────────────────────────────────────────────────────

    def _build_controller(self):
        s = self._section("controller", "Controller")
        b = s.body
        entries = []
        self._label_to_key = {}
        for fam in FAMILIES:
            for spec in CATALOG.values():
                if spec.family == fam:
                    label = f"{fam}  ·  {spec.label}"
                    entries.append(label)
                    self._label_to_key[label] = spec.key
        self._ctrl_combo = combo(b, entries, entries[0], self._controller_picked, width=22)
        self._ctrl_combo.pack(fill="x")
        row = tk.Frame(b, bg=C["surface2"])
        row.pack(fill="x", pady=(6, 0))
        self._new_chip = Chip(row, "New in v2", "accent")
        self._blurb = WrapLabel(b, "", bg=C["surface2"], fg=C["text2"], font=F["body"])
        self._blurb.pack(fill="x", pady=(2, 8))
        self._level = Segmented(b, [("basic", "Basic"), ("adv", "Advanced")], "basic",
                                self._level_changed, bg=C["surface2"], pady=5)
        self._level.pack(fill="x", pady=(0, 4))
        self._param_host = tk.Frame(b, bg=C["surface2"])
        self._param_host.pack(fill="x")
        Btn(b, "Reset to defaults", self._reset_defaults, kind="ghost", padx=10, pady=5,
            font=F["small_bold"]).pack(fill="x", pady=(10, 0))
        self._controller_picked(entries[0], notify=False)

    def _controller_picked(self, label, notify=True):
        self.ctrl_key = self._label_to_key[label]
        spec = CATALOG[self.ctrl_key]
        self._blurb.configure(text=spec.blurb)
        if spec.experimental:
            self._new_chip.set("Experimental", "warn")
            self._new_chip.pack(side="left")
        elif spec.new:
            self._new_chip.set("New in v2", "accent")
            self._new_chip.pack(side="left")
        else:
            self._new_chip.pack_forget()
        self._rebuild_params()
        if notify:
            self._changed()

    def _level_changed(self, level):
        self.level = level
        self._rebuild_params()

    def _rebuild_params(self):
        for w in self._param_host.winfo_children():
            w.destroy()
        spec: ControllerSpec = CATALOG[self.ctrl_key]
        vals = self.values[self.ctrl_key]
        group = None
        shown = 0
        for p in spec.params:
            if p.level == "adv" and self.level == "basic":
                continue
            if p.group != group:
                group = p.group
                if group:
                    caption(self._param_host, group, bg=C["surface2"], pady=(10, 2))
            SliderField(self._param_host, p, vals[p.key], self._param_changed,
                        bg=C["surface2"]).pack(fill="x", pady=4)
            shown += 1

    def _param_changed(self, key, value):
        self.values[self.ctrl_key][key] = value
        self._changed()

    def _reset_defaults(self):
        self.values[self.ctrl_key] = default_values(self.ctrl_key)
        self._rebuild_params()
        self._changed()

    def get_controller(self):
        return self.ctrl_key, dict(self.values[self.ctrl_key])

    def all_values(self) -> Dict[str, dict]:
        return {k: dict(v) for k, v in self.values.items()}

    # ── World ────────────────────────────────────────────────────────────────

    def _build_world(self):
        s = self._section("world", "World")
        b = s.body
        tk.Label(b, text="Wind", bg=C["surface2"], fg=C["text2"], font=F["body"],
                 anchor="w").pack(fill="x")
        self._wind = combo(b, [lbl for _, lbl in WIND_LEVELS], WIND_LEVELS[0][1],
                           lambda v: self._changed(), width=22)
        self._wind.pack(fill="x", pady=(2, 6))
        self.wind_dir = SliderField(b, Param("wind_dir", "Wind direction", 0, 0, 355, 5, unit="deg",
                                             hint="0 blows toward +X, 90 toward +Y"), 0,
                                    lambda k, v: self._changed(), bg=C["surface2"])
        self.wind_dir.pack(fill="x", pady=4)
        self.turb = tk.IntVar(value=0)
        check(b, "Turbulence (Dryden model)", self.turb, self._changed, bg=C["surface2"]).pack(fill="x", pady=(6, 0))
        self.turb_level = SliderField(b, Param("turb", "Turbulence intensity", 0.3, 0.1, 1.0, 0.05), 0.3,
                                      lambda k, v: self._changed(), bg=C["surface2"])
        self.turb_level.pack(fill="x", pady=4)
        self.ground = tk.IntVar(value=1)
        check(b, "Ground effect", self.ground, self._changed, bg=C["surface2"]).pack(fill="x")
        divider(b, pady=8)
        caption(b, "Sensors", bg=C["surface2"], pady=(0, 4))
        self.sensors = RadioList(b, SENSOR_MODES, "fused", lambda v: self._changed(), bg=C["surface2"])
        self.sensors.pack(fill="x")

    def get_env_params(self) -> EnvironmentParams:
        key = next(k for k, lbl in WIND_LEVELS if lbl == self._wind.get())
        mode = self.sensors.get()
        return EnvironmentParams(
            wind_condition=WindCondition[key], wind_direction=np.radians(float(self.wind_dir.get())),
            turbulence_enabled=bool(self.turb.get()), turbulence_intensity=float(self.turb_level.get()),
            ground_effect_enabled=bool(self.ground.get()), noise_enabled=(mode == "noisy"))

    def use_estimator(self) -> bool:
        return self.sensors.get() == "fused"

    # ── Events ───────────────────────────────────────────────────────────────

    def _build_events(self):
        s = self._section("events", "Events")
        b = s.body
        WrapLabel(b, "Schedule things going wrong during the flight. Rotor faults reduce that "
                     "rotor's thrust; gusts add a burst of wind.", bg=C["surface2"],
                  fg=C["text2"]).pack(fill="x", pady=(0, 6))

        caption(b, "Rotor fault", bg=C["surface2"], pady=(6, 2))
        row = tk.Frame(b, bg=C["surface2"])
        row.pack(fill="x")
        row.grid_columnconfigure(0, weight=1)
        tk.Label(row, text="Rotor", bg=C["surface2"], fg=C["text2"], font=F["body"],
                 anchor="w").grid(row=0, column=0, sticky="w")
        self._f_rotor = combo(row, ["1", "2", "3", "4"], "2", width=4)
        self._f_rotor.grid(row=0, column=1, sticky="e")
        self._f_time = LabeledEntry(b, "At time", 20, unit="s", bg=C["surface2"])
        self._f_time.pack(fill="x", pady=3)
        self._f_loss = SliderField(b, Param("loss", "Thrust lost", 50, 10, 100, 5, unit="%",
                                            hint="100 means the rotor stops completely."), 50,
                                   None, bg=C["surface2"])
        self._f_loss.pack(fill="x", pady=3)
        Btn(b, "Add rotor fault", self._add_fault, kind="secondary", padx=10, pady=6,
            font=F["small_bold"]).pack(fill="x", pady=(4, 0))

        divider(b, pady=10)
        caption(b, "Wind gust", bg=C["surface2"], pady=(0, 2))
        self._g_time = LabeledEntry(b, "At time", 10, unit="s", bg=C["surface2"])
        self._g_time.pack(fill="x", pady=3)
        self._g_speed = LabeledEntry(b, "Speed", 6, unit="m/s", bg=C["surface2"])
        self._g_speed.pack(fill="x", pady=3)
        self._g_dir = LabeledEntry(b, "Direction", 0, unit="deg", bg=C["surface2"])
        self._g_dir.pack(fill="x", pady=3)
        self._g_dur = LabeledEntry(b, "Duration", 2, unit="s", bg=C["surface2"])
        self._g_dur.pack(fill="x", pady=3)
        Btn(b, "Add gust", self._add_gust, kind="secondary", padx=10, pady=6,
            font=F["small_bold"]).pack(fill="x", pady=(4, 0))

        divider(b, pady=10)
        caption(b, "Scheduled", bg=C["surface2"], pady=(0, 4))
        self._ev_host = tk.Frame(b, bg=C["surface2"])
        self._ev_host.pack(fill="x")
        self._rebuild_events()

    def _add_fault(self):
        loss = float(self._f_loss.get()) / 100.0
        self.events.append(ScheduledEvent(t=self._f_time.get_float(20), kind="rotor_fault",
                                          rotor=int(self._f_rotor.get()) - 1,
                                          effectiveness=1.0 - loss))
        self._rebuild_events()
        self._changed()

    def _add_gust(self):
        sp = self._g_speed.get_float(6)
        d = np.radians(self._g_dir.get_float(0))
        self.events.append(ScheduledEvent(t=self._g_time.get_float(10), kind="gust",
                                          vec=(sp * np.cos(d), sp * np.sin(d), 0.0),
                                          duration=max(0.2, self._g_dur.get_float(2))))
        self._rebuild_events()
        self._changed()

    def _rebuild_events(self):
        for w in self._ev_host.winfo_children():
            w.destroy()
        if not self.events:
            WrapLabel(self._ev_host, "Nothing scheduled.", bg=C["surface2"],
                      fg=C["text3"]).pack(fill="x")
            return
        for i, ev in enumerate(sorted(self.events, key=lambda e: e.t)):
            row = tk.Frame(self._ev_host, bg=C["surface3"])
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(1, weight=1)
            tone = C["red"] if ev.kind == "rotor_fault" else C["amber"]
            tk.Label(row, text=f"{ev.t:.0f} s", bg=C["surface3"], fg=tone, font=F["mono"],
                     width=6, anchor="w").grid(row=0, column=0, padx=(8, 4), pady=6)
            WrapLabel(row, ev.describe(), bg=C["surface3"], fg=C["text"], font=F["body"]).grid(
                row=0, column=1, sticky="ew")
            tk.Button(row, text="✕", bg=C["surface3"], fg=C["text2"], relief="flat", bd=0,
                      font=F["small_bold"], cursor="hand2", highlightthickness=0,
                      command=lambda e=ev: self._remove_event(e)).grid(row=0, column=2, padx=8)

    def _remove_event(self, ev):
        self.events.remove(ev)
        self._rebuild_events()
        self._changed()

    def get_events(self) -> List[ScheduledEvent]:
        return list(self.events)

    # ── Summaries ────────────────────────────────────────────────────────────

    def refresh_summaries(self):
        d = self.get_drone_params()
        twr = 4 * d.kT * d.omega_max ** 2 / (d.mass * d.g)
        hover = np.sqrt(d.mass * d.g / (4 * d.kT))
        self._veh_info.configure(text=f"Thrust-to-weight {twr:.1f}. Hover needs about {hover:.0f} rad/s per rotor.")
        self.sections["vehicle"].set_summary(f"250 mm quad, {d.mass:.2f} kg, thrust/weight {twr:.1f}")

        kind, preset, hover_sp, dur = self.get_mission_basics()
        if kind == "waypoints":
            what = f"{self.waypoint_count} waypoints"
        elif preset == "hover":
            what = f"Hover at ({hover_sp[0]:g}, {hover_sp[1]:g}, {hover_sp[2]:g})"
        else:
            what = PRESET_LABELS[preset]
        self.sections["mission"].set_summary(f"{what}, {dur:.0f} s")

        spec = CATALOG[self.ctrl_key]
        self.sections["controller"].set_summary(f"{spec.label}  ·  {spec.family}")

        env = self.get_env_params()
        wind = env.wind_condition.name.capitalize()
        bits = [f"{wind} wind"]
        if env.turbulence_enabled:
            bits.append("turbulence")
        bits.append({"ideal": "ideal sensors", "noisy": "noisy sensors", "fused": "sensor fusion"}[self.sensors.get()])
        self.sections["world"].set_summary(", ".join(bits))

        n = len(self.events)
        self.sections["events"].set_summary("None scheduled" if not n else f"{n} scheduled")
