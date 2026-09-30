"""
drone_sim/web/studio.py
=======================
DRONSIM Studio (prototype): NiceGUI front end on top of the unchanged core.

    pip install nicegui
    python -m drone_sim.web.studio          # http://localhost:8080
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from nicegui import ui  # noqa: E402

from drone_sim.core.catalog import CATALOG  # noqa: E402
from drone_sim.core.mission import PRESETS, ALGORITHMS  # noqa: E402
from drone_sim.web.common import chrome, rail  # noqa: E402
from drone_sim.web.session import SHARED, StudioSession, WINDS, euler_to_R  # noqa: E402

ARM = 0.23 * 2.5      # visual scale: real size is unreadable at room-scale views
# X-frame rotor layout: (x, y, spin direction)
ROTORS = [(ARM * .707, -ARM * .707, 1), (ARM * .707, ARM * .707, -1),
          (-ARM * .707, ARM * .707, 1), (-ARM * .707, -ARM * .707, -1)]

def build_page():
    sess = StudioSession(SHARED)
    cfg = sess.cfg
    rotor_nodes, state = [], {"angle": [0.0] * 4, "static": []}

    chrome("/")

    # ---- top bar -------------------------------------------------------
    with ui.header().classes("items-center gap-3 bg-[#161b22] px-4 py-2 border-b border-[#30363d]"):
        ui.icon("flight", size="sm").classes("text-blue-400")
        ui.label("DRONSIM Studio").classes("text-lg font-medium")
        ui.badge("prototype", color="orange").props("outline")
        ui.space()
        run_btn = ui.button("Run", icon="play_arrow", on_click=lambda: start()).props("color=positive no-caps")
        pause_btn = ui.button("Pause", icon="pause", on_click=lambda: toggle()).props("outline no-caps")
        stop_btn = ui.button("Stop", icon="stop", on_click=lambda: halt()).props("outline color=negative no-caps")

    with ui.row().classes("w-full no-wrap gap-0").style("height: calc(100vh - 64px)"):
        # ---- left rail -------------------------------------------------
        rail("/")

        # ---- centre: viewport + timeline --------------------------------
        with ui.column().classes("grow gap-0 h-full").style("min-width:0"):
            with ui.element("div").classes("relative w-full grow").style("min-height:0"):
                scene = ui.scene(grid=(10, 10), background_color="#0d1117").classes("w-full h-full")
                with scene:
                    drone = scene.group()
                    with drone:
                        scene.box(0.4, 0.4, 0.12).material("#58a6ff")
                        for i, (rx, ry, d) in enumerate(ROTORS):
                            scene.box(ARM * 1.45, 0.04, 0.03).rotate(0, 0, math.atan2(ry, rx)).material("#8b949e")
                            g = scene.group().move(rx, ry, 0.08)
                            with g:
                                scene.cylinder(0.02, 0.02, 0.08).rotate(math.pi / 2, 0, 0).material("#484f58")
                                scene.box(0.5, 0.03, 0.01).move(0, 0, 0.05).material("#f0883e")
                                scene.box(0.03, 0.5, 0.01).move(0, 0, 0.05).material("#f0883e")
                            rotor_nodes.append(g)
                    trail = scene.point_cloud([[0, 0, 0]], point_size=0.05).material("#58a6ff")
                    target = scene.sphere(0.12).material("#3fb950").move(0, 0, 0)
                scene.move_camera(x=-4.5, y=-5.5, z=4, look_at_x=0, look_at_y=0, look_at_z=1.2, up_z=1, duration=0)

                def cam(x, y, z, lx=0, ly=0, lz=1):
                    scene.move_camera(x=x, y=y, z=z, look_at_x=lx, look_at_y=ly, look_at_z=lz, up_z=1)

                with ui.row().classes("absolute gap-1 bg-[#161b22] rounded-lg p-1 border border-[#30363d]") \
                        .style("top:12px; left:12px; z-index:10"):
                    ui.button("Iso", on_click=lambda: cam(-4.5, -5.5, 4, 0, 0, 1.2)).props("flat dense no-caps")
                    ui.button("Top", on_click=lambda: cam(0, 0.01, 12, 0, 0, 0)).props("flat dense no-caps")
                    ui.button("Side", on_click=lambda: cam(0, -8, 1.5, 0, 0, 1.5)).props("flat dense no-caps")

            with ui.column().classes("w-full gap-1 px-4 py-2 bg-[#161b22] border-t border-[#30363d]"):
                with ui.row().classes("w-full items-center gap-4 tele"):
                    t_lbl = ui.label("t = 0.0 s").classes("text-sm w-24")
                    e_lbl = ui.label("error 0.00 m").classes("text-sm w-32")
                    c_lbl = ui.label("clearance —").classes("text-sm w-36")
                    pos_lbl = ui.label("pos (0.00, 0.00, 0.00)").classes("text-sm grow")
                    warn = ui.label("").classes("text-sm text-orange-400")
                track = ui.element("div").classes("tl-track w-full")
                with track:
                    fill = ui.element("div").classes("tl-fill").style("width:0%")
                    marks = ui.element("div").classes("absolute inset-0")
                event_col = ui.label("No events yet").classes("text-xs text-gray-400")

        # ---- right drawer: property cards -------------------------------
        with ui.scroll_area().classes("h-full bg-[#0d1117] border-l border-[#30363d]").style("width:320px;flex:none"):
            with ui.column().classes("gap-2 p-3 w-full"):
                def card(title, icon, opened=False):
                    c = ui.expansion(title, icon=icon, value=opened).classes("dcard w-full rounded-lg") \
                        .props("dense header-class=text-weight-medium")
                    return c

                with card("Controller", "settings_suggest", True):
                    opts = {k: f"{k}  ({v.family})" + (" — experimental" if v.experimental else "")
                            for k, v in CATALOG.items()}
                    with ui.column().classes("p-3 w-full gap-2"):
                        ui.select(opts, label="Method").bind_value(cfg, "controller").classes("w-full")
                        blurb = ui.label("").classes("text-xs text-gray-400")
                        blurb.bind_text_from(cfg, "controller", lambda k: CATALOG[k].blurb)
                        ui.switch("Kalman + complementary estimator").bind_value(cfg, "estimator")
                with card("Mission", "route", True):
                    with ui.column().classes("p-3 w-full gap-2"):
                        ui.toggle({False: "Preset", True: "Waypoints"}).bind_value(cfg, "use_waypoints")
                        ui.select(PRESETS, label="Trajectory").bind_value(cfg, "preset").classes("w-full") \
                            .bind_visibility_from(cfg, "use_waypoints", lambda w: not w)
                        ui.select(ALGORITHMS, label="Avoidance").bind_value(cfg, "avoid").classes("w-full")
                        ui.switch("Obstacles (box + pillar)").bind_value(cfg, "obstacles")
                        ui.number("Duration [s]", min=5, max=300, step=5).bind_value(cfg, "duration").classes("w-full")
                with card("World", "air"):
                    with ui.column().classes("p-3 w-full gap-2"):
                        ui.select(list(WINDS), label="Wind").bind_value(cfg, "wind").classes("w-full")
                        ui.switch("Turbulence").bind_value(cfg, "turbulence")
                        ui.switch("Sensor noise").bind_value(cfg, "noise")
                with card("Events", "bolt"):
                    with ui.column().classes("p-3 w-full gap-2"):
                        ui.label("Set time to 0 to disable").classes("text-xs text-gray-400")
                        ui.number("Rotor fault at [s]", min=0, step=1).bind_value(cfg, "fault_t").classes("w-full")
                        ui.select({0: "Rotor 1", 1: "Rotor 2", 2: "Rotor 3", 3: "Rotor 4"}, label="Rotor") \
                            .bind_value(cfg, "fault_rotor").classes("w-full")
                        ui.slider(min=0, max=1, step=0.05).props("label-always").bind_value(cfg, "fault_eff")
                        ui.label("remaining thrust (0 = dead)").classes("text-xs text-gray-400")
                        ui.number("Gust at [s]", min=0, step=1).bind_value(cfg, "gust_t").classes("w-full")
                        ui.number("Gust speed [m/s]", min=0, step=0.5).bind_value(cfg, "gust_speed").classes("w-full")
                with card("Vehicle", "precision_manufacturing"):
                    with ui.column().classes("p-3 w-full gap-1 text-sm"):
                        dp = sess.dp
                        ui.label(f"Mass {dp.mass} kg · arm {dp.arm_length} m")
                        ui.label("Vehicle editing comes with the guided setup screen.").classes("text-xs text-gray-400")

    # ---- behaviour -------------------------------------------------------
    def draw_static():
        for n in state["static"]:
            n.delete()
        state["static"] = []
        b, spec = sess.built, sess.spec
        with scene:
            for o in spec.obstacles:
                if hasattr(o, "lx"):
                    n = scene.box(o.lx, o.ly, o.lz).move(o.cx, o.cy, o.cz).material(o.color, 0.6)
                else:
                    n = scene.cylinder(o.radius, o.radius, o.z_top - o.z_bot).rotate(math.pi / 2, 0, 0) \
                        .move(o.cx, o.cy, (o.z_top + o.z_bot) / 2).material(o.color, 0.6)
                state["static"].append(n)
            for w in b.waypoints if spec.kind == "waypoints" else []:
                state["static"].append(scene.sphere(0.06).move(w.x, w.y, w.z).material("#d29922"))
            if b.poly_path is not None:
                for p, q in zip(b.poly_path[:-1:4], b.poly_path[4::4]):
                    state["static"].append(scene.line(list(map(float, p)), list(map(float, q))).material("#d29922"))
        marks.clear()
        with marks:
            for ev in sess.planned_events:
                ui.element("div").classes("tl-mark").style(f"left:{100 * ev.t / spec.duration:.2f}%") \
                    .tooltip(ev.describe())

    def start():
        sess.start()
        draw_static()
        event_col.text = "Running…"

    def toggle():
        paused = sess.toggle_pause()
        pause_btn.text = "Resume" if paused else "Pause"

    def halt():
        sess.stop()
        pause_btn.text = "Pause"

    def tick():
        snap = sess.snapshot()
        live = snap is not None
        run_btn.enabled = not (live and snap.running)
        pause_btn.enabled = stop_btn.enabled = live and snap.running
        if not live:
            return
        drone.move(*map(float, snap.pos))
        drone.rotate_R(euler_to_R(*map(float, snap.euler)))
        target.move(*map(float, snap.setpoint))
        for i, g in enumerate(rotor_nodes):
            state["angle"][i] += ROTORS[i][2] * snap.rotors[i] * 0.05 * 0.15
            g.rotate(0, 0, state["angle"][i])
        if len(snap.trail) >= 2:
            trail.set_points(snap.trail[::3].tolist())
        t_lbl.text = f"t = {snap.t:5.1f} s"
        e_lbl.text = f"error {snap.error:.2f} m"
        c_lbl.text = "clearance —" if snap.clearance > 50 else f"clearance {snap.clearance:.2f} m"
        pos_lbl.text = "pos (%.2f, %.2f, %.2f)" % tuple(snap.pos)
        fill.style(f"width:{min(100, 100 * snap.t / sess.spec.duration):.1f}%")
        bad = [f"R{i + 1} {int(100 * (1 - e))}% loss" for i, e in enumerate(snap.effectiveness) if e < 0.999]
        warn.text = ("⚠ " + ", ".join(bad)) if bad else ""
        if snap.events:
            event_col.text = " · ".join(f"{t:.1f}s {txt}" for t, txt in snap.events)
        elif not snap.running:
            event_col.text = "Finished — no events fired"

    ui.timer(0.05, tick)


@ui.page("/")
def index():
    build_page()


from drone_sim.web import setup  # noqa: E402,F401  (registers /setup)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="DRONSIM Studio", port=int(os.environ.get("PORT", 8080)), reload=False, show=False)
