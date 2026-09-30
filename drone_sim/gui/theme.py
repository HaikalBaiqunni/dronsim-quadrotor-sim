"""
drone_sim/gui/theme.py
=======================
Colours, fonts and ttk styling shared by every GUI module.

Rules the rest of the GUI follows so text never gets clipped:
  * body text is at least 9 pt, controls use 10 pt
  * labels that can be long use ``WrapLabel`` (wraps to the available width)
  * controls sit in grid/pack cells that stretch (``fill='x'`` / ``sticky='ew'``)
    instead of fixed character widths
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

C = {
    "bg":       "#0E1116",   # window background
    "surface":  "#151A21",   # panels
    "surface2": "#1B222B",   # cards, section headers
    "surface3": "#252D39",   # inputs, idle buttons
    "border":   "#2D3644",
    "text":     "#E6EDF3",
    "text2":    "#A3B0BE",
    "text3":    "#7C8896",
    "accent":   "#4C9AFF",
    "accent_bg": "#16304F",
    "accent_fg": "#0B1220",
    "green":    "#3DDC97",
    "green_bg": "#12352A",
    "amber":    "#F5B942",
    "amber_bg": "#3A2E12",
    "red":      "#FF6B6B",
    "red_bg":   "#3D1B1F",
    "purple":   "#B392F0",
    "coral":    "#FF9E7A",
    "teal":     "#5CD6D6",
}

# distinct series colours (controllers, drones)
SERIES = ["#4C9AFF", "#3DDC97", "#FF9E7A", "#B392F0", "#F5B942", "#5CD6D6", "#FF7AB6", "#A5D6A7"]

_family = "Segoe UI"


def init_fonts(root: tk.Misc):
    """Pick the UI font family once a Tk root exists."""
    global _family
    families = set(tkfont.families(root))
    for cand in ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Arial"):
        if cand in families:
            _family = cand
            break
    mono = "Consolas" if "Consolas" in families else "Courier New"
    F["body"] = (_family, 10)
    F["bold"] = (_family, 10, "bold")
    F["small"] = (_family, 9)
    F["small_bold"] = (_family, 9, "bold")
    F["title"] = (_family, 11, "bold")
    F["h1"] = (_family, 13, "bold")
    F["kpi"] = (mono, 15, "bold")
    F["mono"] = (mono, 10)
    F["mono_small"] = (mono, 9)


F = {
    "body": ("Segoe UI", 10), "bold": ("Segoe UI", 10, "bold"),
    "small": ("Segoe UI", 9), "small_bold": ("Segoe UI", 9, "bold"),
    "title": ("Segoe UI", 11, "bold"), "h1": ("Segoe UI", 13, "bold"),
    "kpi": ("Consolas", 15, "bold"), "mono": ("Consolas", 10),
    "mono_small": ("Consolas", 9),
}


def apply_style(root: tk.Misc):
    """Configure ttk widgets (Combobox, Scale, Scrollbar, Treeview, Progressbar)."""
    init_fonts(root)
    root.option_add("*Font", F["body"])
    root.option_add("*TCombobox*Listbox.background", C["surface3"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", C["accent_fg"])
    root.option_add("*TCombobox*Listbox.font", F["body"])

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure("TCombobox", fieldbackground=C["surface3"], background=C["surface3"],
                foreground=C["text"], arrowcolor=C["text2"], bordercolor=C["border"],
                lightcolor=C["surface3"], darkcolor=C["surface3"], padding=(6, 5),
                selectbackground=C["surface3"], selectforeground=C["text"])
    s.map("TCombobox", fieldbackground=[("readonly", C["surface3"])],
          foreground=[("readonly", C["text"])],
          bordercolor=[("focus", C["accent"])])
    s.configure("Horizontal.TScale", background=C["surface"], troughcolor=C["surface3"],
                bordercolor=C["surface"], lightcolor=C["accent"], darkcolor=C["accent"])
    s.configure("Vertical.TScrollbar", background=C["surface3"], troughcolor=C["surface"],
                bordercolor=C["surface"], arrowcolor=C["text2"], lightcolor=C["surface3"],
                darkcolor=C["surface3"], gripcount=0)
    s.map("Vertical.TScrollbar", background=[("active", C["border"])])
    s.configure("Horizontal.TProgressbar", troughcolor=C["surface3"], background=C["accent"],
                bordercolor=C["surface3"], lightcolor=C["accent"], darkcolor=C["accent"])
    s.configure("Treeview", background=C["surface2"], fieldbackground=C["surface2"],
                foreground=C["text"], bordercolor=C["border"], rowheight=28, font=F["body"])
    s.configure("Treeview.Heading", background=C["surface3"], foreground=C["text2"],
                relief="flat", font=F["small_bold"], padding=(8, 6))
    s.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    s.map("Treeview", background=[("selected", C["accent_bg"])],
          foreground=[("selected", C["text"])])
    s.map("Treeview.Heading", background=[("active", C["border"])])
