"""
drone_sim/web/setup.py
======================
Guided setup (layout E): pick what matters, choose a controller from cards whose
ratings are measured (see ratings.py), review, open in Studio.
"""

import asyncio

from nicegui import run, ui

from drone_sim.core.catalog import CATALOG
from drone_sim.web import ratings as R
from drone_sim.web.common import chrome, rail
from drone_sim.web.session import SHARED

PRIORITIES = {
    "robust": ("Survive wind and faults", "shield", lambda r: (-r.robust_stars, r.robust_rmse if r.robust_rmse == r.robust_rmse else 9e9)),
    "precise": ("Track a path tightly", "my_location", lambda r: r.precision_rmse),
    "cpu": ("Run cheap and fast", "speed", lambda r: r.cpu_rel),
    "all": ("No preference", "balance", None),
}


def stars_text(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def build_setup():
    chrome("/setup")
    cfg = SHARED
    st = {"ratings": R.load_cached(), "prio": "all", "busy": False}
    prio_cards = {}

    with ui.header().classes("items-center gap-3 bg-[#161b22] px-4 py-2 border-b border-[#30363d]"):
        ui.icon("flight", size="sm").classes("text-blue-400")
        ui.label("DRONSIM · Guided setup").classes("text-lg font-medium")
        ui.badge("prototype", color="orange").props("outline")

    with ui.row().classes("w-full no-wrap gap-0").style("height: calc(100vh - 64px)"):
        rail("/setup")
        with ui.scroll_area().classes("grow h-full"):
            with ui.column().classes("w-full p-4 gap-3").style("max-width:1200px;margin:0 auto"):
                with ui.stepper().props("flat animated").classes("w-full bg-transparent") as stepper:
                    # ---- step 1 -------------------------------------------------
                    with ui.step("Goal", icon="flag"):
                        ui.label("What matters most for this flight?").classes("text-base")
                        ui.label("This only sorts the cards on the next step.").classes("text-xs text-gray-400")
                        with ui.row().classes("gap-3 mt-2"):
                            for k, (title, icon, _) in PRIORITIES.items():
                                with ui.card().classes("dcard pick p-4 items-center gap-1").style("width:190px") as c:
                                    ui.icon(icon, size="md").classes("text-blue-400")
                                    ui.label(title).classes("text-sm text-center")
                                c.on("click", lambda k=k: choose_prio(k))
                                prio_cards[k] = c
                        with ui.stepper_navigation():
                            ui.button("Next", on_click=stepper.next).props("no-caps color=primary")

                    # ---- step 2 -------------------------------------------------
                    with ui.step("Controller", icon="settings_suggest"):
                        status = ui.row().classes("items-center gap-3 w-full")
                        with ui.row().classes("w-full no-wrap gap-4"):
                            cards_box = ui.column().classes("gap-2 grow").style("min-width:0")
                            with ui.card().classes("dcard p-3 gap-2").style("width:360px;flex:none;align-self:flex-start"):
                                ui.label("Preview").classes("text-sm font-medium")
                                ui.label("Position error over the circle task (measured).").classes("text-xs text-gray-400")
                                chart = ui.echart({
                                    "animation": False, "grid": {"left": 40, "right": 10, "top": 10, "bottom": 28},
                                    "xAxis": {"type": "value", "name": "s", "nameLocation": "end"},
                                    "yAxis": {"type": "value", "name": "m"},
                                    "series": [{"type": "line", "showSymbol": False, "data": [],
                                                "lineStyle": {"color": "#58a6ff"}}],
                                }).classes("w-full").style("height:200px")
                                prev_note = ui.label("").classes("text-xs text-gray-400")
                        with ui.stepper_navigation():
                            ui.button("Next", on_click=stepper.next).props("no-caps color=primary")
                            ui.button("Back", on_click=stepper.previous).props("flat no-caps")

                    # ---- step 3 -------------------------------------------------
                    with ui.step("Review", icon="fact_check"):
                        review = ui.column().classes("gap-1")
                        with ui.stepper_navigation():
                            ui.button("Open in Studio", icon="view_in_ar", on_click=lambda: ui.navigate.to("/")) \
                                .props("no-caps color=positive")
                            ui.button("Back", on_click=stepper.previous).props("flat no-caps")

    # ---- behaviour -----------------------------------------------------------
    def choose_prio(k):
        st["prio"] = k
        for kk, c in prio_cards.items():
            if kk == k:
                c.classes(add="sel")
            else:
                c.classes(remove="sel")
        render_cards()

    def card_for(key, spec, r):
        sel = cfg.controller == key
        with cards_box:
            c = ui.card().classes("dcard pick p-3 w-full gap-1" + (" sel" if sel else ""))
            with c:
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(spec.label).classes("text-base font-medium")
                    ui.badge(spec.family, color="grey-8")
                    if spec.new:
                        ui.badge("new", color="blue")
                    if spec.experimental:
                        ui.badge("experimental", color="orange")
                    if r is not None and r.crashed:
                        ui.badge("crashes on 50% rotor loss", color="red")
                ui.label(spec.blurb).classes("text-xs text-gray-400")
                if r is None:
                    ui.label("Not rated" + (" — untrained, known to crash." if spec.experimental
                                            else " — run the measurement."))\
                        .classes("text-xs text-orange-400")
                else:
                    with ui.grid(columns="110px 90px 1fr").classes("gap-x-3 gap-y-0 items-center"):
                        ui.label("Robustness").classes("text-xs")
                        ui.label(stars_text(r.robust_stars)).classes("stars text-sm")
                        ui.label("crashed" if r.crashed else f"wind + fault RMSE {r.robust_rmse:.2f} m") \
                            .classes("text-xs text-gray-400")
                        ui.label("Tracking").classes("text-xs")
                        ui.label(stars_text(r.precision_stars)).classes("stars text-sm")
                        ui.label(f"circle RMSE {r.precision_rmse:.2f} m").classes("text-xs text-gray-400")
                        ui.label("CPU cost").classes("text-xs")
                        ui.label(stars_text(r.cpu_stars)).classes("stars text-sm")
                        ui.label(f"{r.cpu_rel:.1f}× the fastest").classes("text-xs text-gray-400")
            c.on("click", lambda k=key: select(k))

    def render_cards():
        cards_box.clear()
        rs = st["ratings"] or {}
        keys = list(CATALOG)
        fn = PRIORITIES[st["prio"]][2]
        if fn and rs:
            keys.sort(key=lambda k: (k not in rs, fn(rs[k]) if k in rs else 0))
        for k in keys:
            card_for(k, CATALOG[k], rs.get(k))
        refresh_side()

    def refresh_side():
        r = (st["ratings"] or {}).get(cfg.controller)
        if r:
            chart.options["series"][0]["data"] = [[t, e] for t, e in zip(r.curve_t, r.curve_err)]
            prev_note.text = "Includes the first 5 s (climb from the ground)."
        else:
            chart.options["series"][0]["data"] = []
            prev_note.text = "No measurement for this controller."
        chart.update()
        spec = CATALOG[cfg.controller]
        review.clear()
        with review:
            ui.label(f"Controller: {spec.label} ({spec.family})").classes("text-base")
            ui.label(spec.blurb).classes("text-xs text-gray-400")
            if r:
                ui.label(f"Measured: robustness {stars_text(r.robust_stars)} · tracking {stars_text(r.precision_stars)}"
                         f" · CPU {stars_text(r.cpu_stars)}").classes("text-sm stars")
            if spec.experimental:
                ui.label("Experimental: this controller is untrained and is expected to crash.") \
                    .classes("text-sm text-orange-400")
            ui.label("Ratings come from 20 s simulated flights: a circle in calm air, and a hover in moderate "
                     "wind with a 50% rotor loss at 8 s. Tracking stars are relative to the best controller.") \
                .classes("text-xs text-gray-400 mt-2")

    def select(key):
        cfg.controller = key
        render_cards()

    async def measure():
        if st["busy"]:
            return
        st["busy"] = True
        btn.enabled = False
        spin.visible = True
        try:
            st["ratings"] = await run.io_bound(R.measure_and_save, lambda s: None)
        finally:
            st["busy"] = False
            spin.visible = False
            btn.enabled = True
            note()
            render_cards()

    def note():
        msg.text = ("Ratings measured from simulation runs." if st["ratings"]
                    else "No ratings yet. Measuring takes about a minute.")
        btn.text = "Re-measure" if st["ratings"] else "Measure ratings"

    with status:
        msg = ui.label("").classes("text-sm")
        btn = ui.button("Measure ratings", icon="science", on_click=measure).props("no-caps outline")
        spin = ui.spinner(size="sm")
        spin.visible = False
        ui.label("(takes ~1 min, cached afterwards)").classes("text-xs text-gray-400")
    note()
    choose_prio("all")


@ui.page("/setup")
def setup_page():
    build_setup()
