"""
drone_sim/web/analysis_page.py
==============================
Analysis screen (layout F): fly several controllers on one scenario, then compare
them: toggle runs, error curves with event markers, leaderboard, insight cards,
replay scrubber with a marker per run in a 3D view.
"""

from nicegui import run, ui

from drone_sim.core.catalog import CATALOG
from drone_sim.core.environment import WindCondition
from drone_sim.web import analysis as A
from drone_sim.web.common import chrome, rail

PALETTE = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#ff7b72", "#e3b341", "#79c0ff", "#a5d6ff"]
WINDS = {w.name.title(): w for w in WindCondition}
KIND_ICON = {"good": ("check_circle", "text-green-400"), "warn": ("warning", "text-yellow-400"),
             "bad": ("error", "text-red-400"), "info": ("info", "text-blue-300")}


def build_analysis():
    chrome("/analysis")
    st = {"runs": [], "visible": set(), "busy": False, "playing": False}
    form = {"keys": ["PID", "SMC", "ADRC", "GEO"], "scenario": "circle", "wind": "Calm",
            "turb": False, "fault": 8.0, "gust": 0.0, "dur": 20.0}
    color = {}

    with ui.header().classes("items-center gap-3 bg-[#161b22] px-4 py-2 border-b border-[#30363d]"):
        ui.icon("flight", size="sm").classes("text-blue-400")
        ui.label("DRONSIM · Analysis").classes("text-lg font-medium")
        ui.badge("prototype", color="orange").props("outline")

    with ui.row().classes("w-full no-wrap gap-0").style("height: calc(100vh - 64px)"):
        rail("/analysis")
        with ui.scroll_area().classes("grow h-full"):
            with ui.column().classes("w-full p-4 gap-3").style("max-width:1300px;margin:0 auto"):
                # ---- setup card ------------------------------------------------
                with ui.card().classes("dcard w-full p-3 gap-2"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.select({k: k for k in CATALOG if not CATALOG[k].experimental},
                                  multiple=True, label="Controllers").props("use-chips dense") \
                            .bind_value(form, "keys").classes("grow").style("min-width:260px")
                        ui.select(A.SCENARIOS, label="Trajectory").props("dense").bind_value(form, "scenario") \
                            .style("width:140px")
                        ui.select(list(WINDS), label="Wind").props("dense").bind_value(form, "wind").style("width:120px")
                        ui.number("Fault at [s]", min=0, step=1).props("dense").bind_value(form, "fault").style("width:110px")
                        ui.number("Gust at [s]", min=0, step=1).props("dense").bind_value(form, "gust").style("width:110px")
                        ui.number("Length [s]", min=8, max=60, step=2).props("dense").bind_value(form, "dur").style("width:100px")
                        ui.switch("Turbulence").bind_value(form, "turb")
                        fly_btn = ui.button("Fly", icon="play_arrow", on_click=lambda: fly()).props("no-caps color=positive")
                        spin = ui.spinner(size="sm")
                        spin.visible = False
                    status = ui.label("Set up a scenario and press Fly. 0 disables an event. A run takes a few seconds "
                                      "per controller.").classes("text-xs text-gray-400")
                    chips = ui.row().classes("gap-2")

                # ---- curves + replay -------------------------------------------
                with ui.row().classes("w-full no-wrap gap-3"):
                    with ui.card().classes("dcard p-3 gap-1 grow").style("min-width:0"):
                        ui.label("Position error (axis fitted to finished runs; crashed runs run off the top)") \
                            .classes("text-sm font-medium")
                        chart = ui.echart({
                            "animation": False, "legend": {"show": False},
                            "tooltip": {"trigger": "axis"},
                            "grid": {"left": 44, "right": 14, "top": 14, "bottom": 28},
                            "xAxis": {"type": "value", "name": "s", "nameLocation": "end"},
                            "yAxis": {"type": "value", "name": "m"},
                            "series": [],
                        }).classes("w-full").style("height:260px")
                        with ui.row().classes("items-center gap-2 w-full no-wrap"):
                            play_btn = ui.button(icon="play_arrow", on_click=lambda: toggle_play()).props("round dense flat")
                            slider = ui.slider(min=0, max=1, step=0.05, value=0,
                                               on_change=lambda e: seek(e.value)).classes("grow")
                            t_lbl = ui.label("0.0 s").classes("text-xs w-14 tele")
                    with ui.card().classes("dcard p-0").style("width:380px;flex:none"):
                        scene = ui.scene(grid=(10, 10), background_color="#0d1117").classes("w-full").style("height:320px")
                        scene.move_camera(x=-4.5, y=-5.5, z=4, look_at_x=0, look_at_y=0, look_at_z=1.2, up_z=1, duration=0)

                # ---- leaderboard + insights --------------------------------------
                with ui.row().classes("w-full no-wrap gap-3"):
                    with ui.card().classes("dcard p-3 gap-1 grow").style("min-width:0"):
                        ui.label("Leaderboard").classes("text-sm font-medium")
                        table = ui.table(columns=[
                            {"name": "rank", "label": "#", "field": "rank", "align": "left"},
                            {"name": "label", "label": "Controller", "field": "label", "align": "left"},
                            {"name": "rmse", "label": "RMSE [m]", "field": "rmse", "align": "right"},
                            {"name": "max", "label": "Max err [m]", "field": "max", "align": "right"},
                            {"name": "energy", "label": "Effort [×10⁶]", "field": "energy", "align": "right"},
                            {"name": "tilt", "label": "Max tilt [°]", "field": "tilt", "align": "right"},
                            {"name": "status", "label": "Status", "field": "status", "align": "left"},
                        ], rows=[], row_key="label").props("dense flat").classes("w-full bg-transparent")
                    with ui.column().classes("gap-2").style("width:380px;flex:none") as ins_col:
                        pass

    # ---- behaviour ---------------------------------------------------------------
    def render():
        runs = st["runs"]
        color.clear()
        for i, r in enumerate(runs):
            color[r.key] = PALETTE[i % len(PALETTE)]
        chips.clear()
        with chips:
            for r in runs:
                on = r.key in st["visible"]
                ui.chip(r.label + (" ✕" if r.crashed else ""), selectable=True, selected=on,
                        on_selection_change=lambda e, k=r.key: toggle_run(k, e.value)) \
                    .props(f'outline color="{"red" if r.crashed else "white"}" text-color="{color[r.key]}"')
        draw_chart()
        draw_scene()
        rows = []
        for n, r in enumerate(A.leaderboard(runs), 1):
            rows.append({"rank": n, "label": r.label, "rmse": f"{r.rmse:.3f}", "max": f"{r.max_error:.2f}",
                         "energy": f"{r.energy / 1e6:.1f}", "tilt": f"{r.max_tilt:.0f}",
                         "status": "crashed" if r.crashed else "finished"})
        table.rows = rows
        table.update()
        ins_col.clear()
        with ins_col:
            ui.label("Insights").classes("text-sm font-medium")
            for ins in A.insights(runs):
                icon, cls = KIND_ICON[ins.kind]
                with ui.card().classes("dcard p-3 gap-1 w-full"):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon(icon).classes(cls)
                        ui.label(ins.title).classes("text-sm font-medium")
                    ui.label(ins.detail).classes("text-xs text-gray-400")
            if not runs:
                ui.label("Nothing to analyse yet.").classes("text-xs text-gray-400")
        T = max((r.time[-1] for r in runs), default=1.0)
        slider._props["max"] = round(float(T), 2)
        slider.update()
        slider.set_value(0)

    def draw_chart():
        series = []
        for r in st["runs"]:
            if r.key not in st["visible"]:
                continue
            step = max(1, len(r.time) // 400)
            series.append({"name": r.label, "type": "line", "showSymbol": False, "sync": True,
                           "data": [[round(float(t), 3), round(float(e), 4)] for t, e in
                                    zip(r.time[::step], r.error[::step])],
                           "lineStyle": {"color": color[r.key], "width": 2}, "itemStyle": {"color": color[r.key]}})
        if series and st["runs"]:
            marks = [{"xAxis": ev.t, "label": {"formatter": ev.describe(), "color": "#f85149", "fontSize": 10},
                      "lineStyle": {"color": "#f85149", "type": "dashed"}} for ev in st["runs"][0].events]
            series[0]["markLine"] = {"silent": True, "symbol": "none", "data": marks}
        chart.options["series"] = series
        # a crashed run can reach tens of metres and flatten everything else: fit the axis to finished runs
        fin = [float(r.error.max()) for r in st["runs"] if r.key in st["visible"] and not r.crashed]
        chart.options["yAxis"]["max"] = round(max(fin) * 1.15, 2) if fin else None
        chart.update()

    markers = {}

    def draw_scene():
        scene.clear()
        markers.clear()
        with scene:
            for r in st["runs"]:
                if r.key not in st["visible"]:
                    continue
                step = max(1, len(r.time) // 300)
                scene.point_cloud(r.pos[::step].tolist(), point_size=0.03).material(color[r.key])
                markers[r.key] = scene.sphere(0.1).material(color[r.key])
        seek(slider.value)

    def toggle_run(k, on):
        (st["visible"].add if on else st["visible"].discard)(k)
        draw_chart()
        draw_scene()

    def seek(t):
        t_lbl.text = f"{t:.1f} s"
        for r in st["runs"]:
            m = markers.get(r.key)
            if m is not None:
                i = r.at(t)
                m.move(*map(float, r.pos[i]))

    def toggle_play():
        st["playing"] = not st["playing"]
        play_btn.props(f'icon={"pause" if st["playing"] else "play_arrow"}')

    def advance():
        if not st["playing"] or not st["runs"]:
            return
        T = max(r.time[-1] for r in st["runs"])
        v = slider.value + 0.1
        if v >= T:
            v = 0.0
        slider.set_value(v)

    ui.timer(0.1, advance)

    async def fly():
        if st["busy"]:
            return
        if not form["keys"]:
            status.text = "Pick at least one controller."
            return
        st["busy"] = True
        fly_btn.enabled = False
        spin.visible = True
        st["playing"] = False
        status.text = "Flying " + ", ".join(form["keys"]) + " ..."
        try:
            runs = await run.io_bound(
                A.fly, list(form["keys"]), form["scenario"], WINDS[form["wind"]], bool(form["turb"]),
                float(form["fault"] or 0), float(form["gust"] or 0), float(form["dur"] or 20))
            st["runs"] = runs
            st["visible"] = {r.key for r in runs}
            status.text = f"{len(runs)} runs. Toggle the chips to show or hide a run."
            render()
        except Exception as exc:  # surfaced to the user instead of dying silently
            status.text = f"Run failed: {exc}"
        finally:
            st["busy"] = False
            fly_btn.enabled = True
            spin.visible = False
            play_btn.props("icon=play_arrow")


@ui.page("/analysis")
def analysis_page():
    build_analysis()
