"""Shared page chrome for the web UI."""

from nicegui import ui

CSS = """
body { background: #0d1117; }
.q-card.dcard { background: #161b22; border: 1px solid #30363d; box-shadow: none; }
.tl-track { position: relative; height: 26px; background: #21262d; border-radius: 6px; overflow: hidden; }
.tl-fill { position: absolute; left: 0; top: 0; bottom: 0; background: #1f6feb55; }
.tl-mark { position: absolute; top: 0; bottom: 0; width: 3px; background: #f85149; }
.tele { font-variant-numeric: tabular-nums; }
.stars { color: #d29922; letter-spacing: 1px; }
.pick { cursor: pointer; }
.pick.sel { border-color: #58a6ff !important; }
"""

PAGES = (("/", "view_in_ar", "Studio"), ("/setup", "tune", "Guided setup"),
         ("/analysis", "insights", "Analysis"))


def chrome(active: str):
    """Dark theme, CSS and the left navigation rail. Call inside a page."""
    ui.add_css(CSS)
    ui.dark_mode(True)


def rail(active: str):
    with ui.column().classes("items-center gap-2 py-3 px-1 bg-[#0d1117] border-r border-[#30363d]") \
            .style("width:56px;flex:none"):
        for path, icon, tip in PAGES:
            soon = "soon" in tip
            b = ui.button(icon=icon, on_click=(lambda p=path: ui.navigate.to(p))) \
                .props("round " + ("unelevated color=primary" if path == active else "flat")
                       + (" disable" if soon else ""))
            b.tooltip(tip)
