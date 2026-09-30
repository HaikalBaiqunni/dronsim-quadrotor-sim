"""
drone_sim/gui/widgets.py
=========================
Reusable, consistently sized widgets.

Design contract (so text is never clipped):
  * every widget stretches with its container; nothing relies on a fixed
    character width for text that can vary
  * long text uses ``WrapLabel``, which re-wraps whenever its width changes
  * buttons have generous padding and a minimum width
"""

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Sequence, Tuple

from .theme import C, F


# ── Buttons ──────────────────────────────────────────────────────────────────

_BTN_KINDS = {
    #            bg            fg            hover bg
    "primary":   (C["accent"], C["accent_fg"], "#6BB0FF"),
    "success":   (C["green"],  "#06241A",      "#63E8B0"),
    "secondary": (C["surface3"], C["text"],    "#313B49"),
    "danger":    (C["red_bg"], C["red"],       "#52242A"),
    "ghost":     (C["surface"], C["text2"],    C["surface3"]),
    "chip":      (C["surface3"], C["text2"],   "#313B49"),
}


class Btn(tk.Button):
    """Flat button with hover state. ``kind``: primary/success/secondary/danger/ghost."""

    def __init__(self, parent, text: str, command: Optional[Callable] = None,
                 kind: str = "secondary", padx: int = 14, pady: int = 7,
                 font=None, **kw):
        bg, fg, hover = _BTN_KINDS[kind]
        super().__init__(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=hover, activeforeground=fg,
                         relief="flat", bd=0, padx=padx, pady=pady,
                         font=font or F["bold"], cursor="hand2",
                         highlightthickness=0, **kw)
        self._colors = (bg, hover)
        self.bind("<Enter>", lambda e: self.configure(bg=self._colors[1]))
        self.bind("<Leave>", lambda e: self.configure(bg=self._colors[0]))

    def restyle(self, kind: str):
        bg, fg, hover = _BTN_KINDS[kind]
        self._colors = (bg, hover)
        self.configure(bg=bg, fg=fg, activebackground=hover, activeforeground=fg)


# ── Text ─────────────────────────────────────────────────────────────────────

class WrapLabel(tk.Label):
    """Label that wraps to whatever width it is given (never clips text)."""

    def __init__(self, parent, text: str = "", fg: str = None, bg: str = None,
                 font=None, **kw):
        super().__init__(parent, text=text, fg=fg or C["text2"],
                         bg=bg or parent.cget("bg"), font=font or F["small"],
                         justify="left", anchor="w", **kw)
        self._pad = 2 * int(self.cget("padx") or 0) + 6
        self.bind("<Configure>", self._rewrap)

    def _rewrap(self, e):
        w = max(e.width - self._pad, 60)
        if abs(w - int(self.cget("wraplength") or 0)) > 3:
            self.configure(wraplength=w)


class Tooltip:
    """Small hover tooltip."""

    def __init__(self, widget: tk.Misc, text: str, delay: int = 450):
        self.widget, self.text, self.delay = widget, text, delay
        self._tip, self._job = None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _):
        self._job = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, bg=C["surface3"], fg=C["text"],
                 font=F["small"], justify="left", wraplength=280, padx=10, pady=6,
                 highlightthickness=1, highlightbackground=C["border"]).pack()

    def _hide(self, _=None):
        if self._job:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


def divider(parent, pady: int = 8, bg: str = None) -> tk.Frame:
    f = tk.Frame(parent, bg=C["border"], height=1)
    f.pack(fill="x", pady=pady)
    return f


def caption(parent, text: str, bg: str = None, pady=(10, 4)) -> tk.Label:
    """Small upper-case group caption."""
    lbl = tk.Label(parent, text=text.upper(), bg=bg or parent.cget("bg"),
                   fg=C["text3"], font=F["small_bold"], anchor="w")
    lbl.pack(fill="x", pady=pady)
    return lbl


class Card(tk.Frame):
    """Bordered container."""

    def __init__(self, parent, bg: str = None, pad: int = 10, **kw):
        super().__init__(parent, bg=bg or C["surface2"], highlightthickness=1,
                         highlightbackground=C["border"], **kw)
        self.inner = tk.Frame(self, bg=self.cget("bg"))
        self.inner.pack(fill="both", expand=True, padx=pad, pady=pad)


class Chip(tk.Label):
    """Coloured status pill."""

    def __init__(self, parent, text: str = "", tone: str = "neutral", **kw):
        super().__init__(parent, text=text, font=F["small_bold"], padx=8, pady=2, **kw)
        self.set(text, tone)

    def set(self, text: str, tone: str = "neutral"):
        bg, fg = {"neutral": (C["surface3"], C["text2"]),
                  "good": (C["green_bg"], C["green"]),
                  "warn": (C["amber_bg"], C["amber"]),
                  "bad": (C["red_bg"], C["red"]),
                  "accent": (C["accent_bg"], C["accent"])}[tone]
        self.configure(text=text, bg=bg, fg=fg)


# ── Scrolling ────────────────────────────────────────────────────────────────

class ScrollFrame(tk.Frame):
    """Vertically scrolling container. Put children in ``.inner``."""

    def __init__(self, parent, bg: str = None):
        bg = bg or C["surface"]
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", self._bind_wheel)
        self.bind("<Leave>", self._unbind_wheel)
        self._scroll_needed = False

    def _on_scroll_set(self, lo, hi):
        self.vsb.set(lo, hi)
        need = not (float(lo) <= 0.0 and float(hi) >= 1.0)
        if need != self._scroll_needed:
            self._scroll_needed = need
            if need:
                self.vsb.pack(side="right", fill="y", before=self.canvas)
            else:
                self.vsb.pack_forget()

    def _on_inner(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _unbind_wheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, e):
        if self._scroll_needed:
            self.canvas.yview_scroll(-int(e.delta / 120), "units")

    def scroll_to(self, widget: tk.Widget):
        self.update_idletasks()
        top = widget.winfo_y()
        total = max(self.inner.winfo_height(), 1)
        self.canvas.yview_moveto(max(0.0, (top - 8) / total))


# ── Collapsible section ──────────────────────────────────────────────────────

class Section(tk.Frame):
    """Collapsible card: title + one-line summary in the header, content in ``.body``."""

    def __init__(self, parent, title: str, summary: str = "", expanded: bool = False,
                 on_toggle: Optional[Callable] = None, badge: str = ""):
        super().__init__(parent, bg=C["surface2"], highlightthickness=1,
                         highlightbackground=C["border"])
        self.on_toggle = on_toggle
        self.expanded = False
        self.header = tk.Frame(self, bg=C["surface2"], cursor="hand2")
        self.header.pack(fill="x")
        top = tk.Frame(self.header, bg=C["surface2"])
        top.pack(fill="x", padx=12, pady=(9, 0))
        self._chev = tk.Label(top, text="▸", bg=C["surface2"], fg=C["text2"], font=F["bold"], width=2, anchor="w")
        self._chev.pack(side="left")
        self._title = tk.Label(top, text=title, bg=C["surface2"], fg=C["text"], font=F["title"], anchor="w")
        self._title.pack(side="left")
        if badge:
            self._badge = Chip(top, badge, "accent")
            self._badge.pack(side="left", padx=8)
        self._summary = WrapLabel(self.header, summary, bg=C["surface2"], fg=C["text2"])
        self._summary.pack(fill="x", padx=(36, 12), pady=(0, 9))
        self.body = tk.Frame(self, bg=C["surface"])
        for w in (self.header, top, self._chev, self._title, self._summary):
            w.bind("<Button-1>", lambda e: self.toggle())
        if expanded:
            self.expand()

    def set_summary(self, text: str):
        self._summary.configure(text=text)

    def expand(self):
        if self.expanded:
            return
        self.expanded = True
        self._chev.configure(text="▾", fg=C["accent"])
        self.body.pack(fill="x", padx=12, pady=(4, 12))
        if self.on_toggle:
            self.on_toggle(self, True)

    def collapse(self):
        if not self.expanded:
            return
        self.expanded = False
        self._chev.configure(text="▸", fg=C["text2"])
        self.body.pack_forget()
        if self.on_toggle:
            self.on_toggle(self, False)

    def toggle(self):
        self.collapse() if self.expanded else self.expand()


# ── Choice widgets ───────────────────────────────────────────────────────────

class Segmented(tk.Frame):
    """Row of equal-width toggle buttons. Use for short labels only."""

    def __init__(self, parent, options: Sequence, value=None,
                 command: Optional[Callable] = None, bg: str = None, font=None,
                 pady: int = 6):
        super().__init__(parent, bg=bg or parent.cget("bg"))
        self.command = command
        self._opts = [(o, o) if isinstance(o, str) else o for o in options]
        self._btns = {}
        self._value = value if value is not None else self._opts[0][0]
        for i, (val, lbl) in enumerate(self._opts):
            self.grid_columnconfigure(i, weight=1, uniform="seg")
            b = tk.Button(self, text=lbl, relief="flat", bd=0, padx=6, pady=pady,
                          font=font or F["bold"], cursor="hand2", highlightthickness=0,
                          command=lambda v=val: self.set(v, notify=True))
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 2, 0))
            self._btns[val] = b
        self._paint()

    def _paint(self):
        for val, b in self._btns.items():
            on = val == self._value
            b.configure(bg=C["accent"] if on else C["surface3"],
                        fg=C["accent_fg"] if on else C["text2"],
                        activebackground=C["accent"] if on else "#313B49",
                        activeforeground=C["accent_fg"] if on else C["text"])

    def set(self, value, notify: bool = False):
        self._value = value
        self._paint()
        if notify and self.command:
            self.command(value)

    def get(self):
        return self._value


class RadioList(tk.Frame):
    """Vertical list of options, each with a title and an optional description."""

    def __init__(self, parent, options: Sequence[Tuple], value=None,
                 command: Optional[Callable] = None, bg: str = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, bg=bg)
        self.command = command
        self.var = tk.StringVar(value=value if value is not None else options[0][0])
        for opt in options:
            val, title = opt[0], opt[1]
            desc = opt[2] if len(opt) > 2 else ""
            row = tk.Frame(self, bg=bg)
            row.pack(fill="x", pady=2)
            rb = tk.Radiobutton(row, text=title, variable=self.var, value=val, bg=bg,
                                fg=C["text"], selectcolor=C["surface3"],
                                activebackground=bg, activeforeground=C["text"],
                                font=F["body"], anchor="w", highlightthickness=0,
                                command=self._changed)
            rb.pack(fill="x")
            if desc:
                WrapLabel(row, desc, bg=bg, fg=C["text3"]).pack(fill="x", padx=(24, 0))

    def _changed(self):
        if self.command:
            self.command(self.var.get())

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)



class Slider(tk.Canvas):
    """Thin custom slider (track, filled part, round thumb). Reports raw values."""

    R = 8

    def __init__(self, parent, from_: float, to: float, variable: tk.DoubleVar,
                 command: Optional[Callable] = None, bg: str = None):
        super().__init__(parent, height=24, bg=bg or parent.cget("bg"), highlightthickness=0,
                         bd=0, cursor="hand2")
        self.lo, self.hi, self.var, self.command = from_, to, variable, command
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<ButtonPress-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        self._trace = self.var.trace_add("write", lambda *a: self._draw())

    def _frac(self) -> float:
        span = self.hi - self.lo
        return 0.0 if span <= 0 else min(1.0, max(0.0, (float(self.var.get()) - self.lo) / span))

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 * self.R:
            return
        x0, x1 = self.R, w - self.R
        x = x0 + (x1 - x0) * self._frac()
        self.create_line(x0, h / 2, x1, h / 2, fill=C["surface3"], width=4, capstyle="round")
        self.create_line(x0, h / 2, x, h / 2, fill=C["accent"], width=4, capstyle="round")
        self.create_oval(x - self.R, h / 2 - self.R, x + self.R, h / 2 + self.R,
                         fill=C["accent"], outline=C["surface2"], width=2)

    def _drag(self, e):
        w = self.winfo_width()
        x0, x1 = self.R, w - self.R
        f = min(1.0, max(0.0, (e.x - x0) / max(1, x1 - x0)))
        self.var.set(self.lo + f * (self.hi - self.lo))
        if self.command:
            self.command(None)


# ── Numeric field: label + entry + slider ────────────────────────────────────

def _decimals(step: float) -> int:
    if step >= 1:
        return 0
    return min(5, max(1, int(round(-math.log10(step)))))


class SliderField(tk.Frame):
    """
    Label (wraps) + value entry on one row, slider underneath.
    Also handles ``choice`` (combobox) and ``bool`` (checkbutton) parameters.
    """

    def __init__(self, parent, p, value=None, on_change: Optional[Callable] = None,
                 bg: str = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, bg=bg)
        self.p = p
        self.on_change = on_change
        self._bg = bg
        value = p.default if value is None else value
        text = p.label + (f"  ({p.unit})" if p.unit else "")

        if p.kind == "bool":
            self.var = tk.IntVar(value=int(bool(value)))
            cb = tk.Checkbutton(self, text=p.label, variable=self.var, bg=bg, fg=C["text"],
                                selectcolor=C["surface3"], activebackground=bg,
                                activeforeground=C["text"], font=F["body"], anchor="w",
                                highlightthickness=0, command=self._notify)
            cb.pack(fill="x")
            if p.hint:
                WrapLabel(self, p.hint, bg=bg, fg=C["text3"]).pack(fill="x", padx=(24, 0))
            return

        self.grid_columnconfigure(0, weight=1)
        lbl = WrapLabel(self, text, bg=bg, fg=C["text2"], font=F["body"])
        lbl.grid(row=0, column=0, sticky="ew")

        if p.kind == "choice":
            self.var = tk.StringVar(value=p.options[int(value)])
            cb = ttk.Combobox(self, textvariable=self.var, values=list(p.options),
                              state="readonly", width=8, font=F["body"])
            cb.grid(row=0, column=1, sticky="e", padx=(8, 0))
            cb.bind("<<ComboboxSelected>>", lambda e: self._notify())
            cb.bind("<MouseWheel>", lambda e: "break")
            return

        self._dec = _decimals(p.step)
        self.var = tk.DoubleVar(value=float(value))
        self.entry_var = tk.StringVar(value=self._fmt(float(value)))
        ent = tk.Entry(self, textvariable=self.entry_var, width=8, justify="right",
                       bg=C["surface3"], fg=C["accent"], insertbackground=C["text"],
                       relief="flat", font=F["mono"], highlightthickness=1,
                       highlightbackground=C["border"], highlightcolor=C["accent"])
        ent.grid(row=0, column=1, sticky="e", padx=(8, 0), ipady=3)
        ent.bind("<Return>", self._from_entry)
        ent.bind("<FocusOut>", self._from_entry)
        self.scale = Slider(self, p.lo, p.hi, self.var, self._from_scale, bg=bg)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        if p.hint:
            WrapLabel(self, p.hint, bg=bg, fg=C["text3"]).grid(
                row=2, column=0, columnspan=2, sticky="ew")

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._dec}f}"

    def _quantize(self, v: float) -> float:
        p = self.p
        v = min(max(v, p.lo), p.hi)
        q = round((v - p.lo) / p.step) * p.step + p.lo
        q = round(q, self._dec + 2)
        return int(round(q)) if p.kind == "int" else q

    def _from_scale(self, _):
        v = self._quantize(float(self.var.get()))
        self.var.set(v)
        self.entry_var.set(self._fmt(v))
        self._notify()

    def _from_entry(self, _=None):
        try:
            v = self._quantize(float(self.entry_var.get()))
        except ValueError:
            v = self._quantize(float(self.var.get()))
        self.var.set(v)
        self.entry_var.set(self._fmt(v))
        self._notify()

    def _notify(self):
        if self.on_change:
            self.on_change(self.p.key, self.get())

    def get(self):
        p = self.p
        if p.kind == "bool":
            return int(self.var.get())
        if p.kind == "choice":
            return list(p.options).index(self.var.get())
        v = self._quantize(float(self.var.get()))
        return v

    def set(self, value):
        p = self.p
        if p.kind == "bool":
            self.var.set(int(bool(value)))
        elif p.kind == "choice":
            self.var.set(p.options[int(value)])
        else:
            self.var.set(float(value))
            self.entry_var.set(self._fmt(float(value)))


class LabeledEntry(tk.Frame):
    """Label + right-aligned entry on one row (for plain numbers)."""

    def __init__(self, parent, label: str, value, width: int = 8, unit: str = "",
                 bg: str = None, on_change: Optional[Callable] = None):
        bg = bg or parent.cget("bg")
        super().__init__(parent, bg=bg)
        self.grid_columnconfigure(0, weight=1)
        WrapLabel(self, label + (f"  ({unit})" if unit else ""), bg=bg,
                  fg=C["text2"], font=F["body"]).grid(row=0, column=0, sticky="ew")
        self.var = tk.StringVar(value=str(value))
        self.entry = tk.Entry(self, textvariable=self.var, width=width, justify="right",
                              bg=C["surface3"], fg=C["accent"], insertbackground=C["text"],
                              relief="flat", font=F["mono"], highlightthickness=1,
                              highlightbackground=C["border"], highlightcolor=C["accent"])
        self.entry.grid(row=0, column=1, sticky="e", padx=(8, 0), ipady=3)
        if on_change:
            self.entry.bind("<Return>", lambda e: on_change())
            self.entry.bind("<FocusOut>", lambda e: on_change())

    def get_float(self, default: float = 0.0) -> float:
        try:
            return float(self.var.get())
        except ValueError:
            return default

    def set(self, v):
        self.var.set(str(v))


def combo(parent, values: Sequence[str], value: str, command: Optional[Callable] = None,
          width: int = 12) -> ttk.Combobox:
    cb = ttk.Combobox(parent, values=list(values), state="readonly", font=F["body"], width=width)
    cb.set(value)
    cb.bind("<MouseWheel>", lambda e: "break")
    if command:
        cb.bind("<<ComboboxSelected>>", lambda e: command(cb.get()))
    return cb


def check(parent, text: str, var: tk.Variable, command: Optional[Callable] = None,
          bg: str = None) -> tk.Checkbutton:
    bg = bg or parent.cget("bg")
    return tk.Checkbutton(parent, text=text, variable=var, bg=bg, fg=C["text"],
                          selectcolor=C["surface3"], activebackground=bg,
                          activeforeground=C["text"], font=F["body"], anchor="w",
                          highlightthickness=0, command=command)


# ── Telemetry widgets ────────────────────────────────────────────────────────

class KPI(tk.Frame):
    """Metric card: small label on top, large monospaced value below."""

    def __init__(self, parent, label: str, unit: str = "", bg: str = None):
        super().__init__(parent, bg=bg or C["surface2"], highlightthickness=1,
                         highlightbackground=C["border"])
        tk.Label(self, text=label, bg=self.cget("bg"), fg=C["text3"], font=F["small"],
                 anchor="w").pack(fill="x", padx=10, pady=(7, 0))
        row = tk.Frame(self, bg=self.cget("bg"))
        row.pack(fill="x", padx=10, pady=(0, 7))
        self._val = tk.Label(row, text="—", bg=self.cget("bg"), fg=C["text"], font=F["kpi"], anchor="w")
        self._val.pack(side="left")
        self._unit = tk.Label(row, text=unit, bg=self.cget("bg"), fg=C["text3"], font=F["small"])
        self._unit.pack(side="left", padx=(5, 0), pady=(5, 0))

    def set(self, text: str, tone: Optional[str] = None):
        col = {"good": C["green"], "warn": C["amber"], "bad": C["red"],
               None: C["text"]}.get(tone, C["text"])
        self._val.configure(text=text, fg=col)


class BarMeter(tk.Canvas):
    """Thin horizontal meter with a text label on the left and value on the right."""

    def __init__(self, parent, label: str, color: str = None, height: int = 22, bg: str = None):
        super().__init__(parent, height=height, bg=bg or parent.cget("bg"),
                         highlightthickness=0, bd=0)
        self.label, self.color = label, color or C["accent"]
        self._frac, self._text, self._dim = 0.0, "", False
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, frac: float, text: str = "", dim: bool = False):
        self._frac, self._text, self._dim = max(0.0, min(1.0, frac)), text, dim
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20:
            return
        lab_w, val_w = 26, 62
        self.create_text(0, h / 2, text=self.label, anchor="w", fill=C["text2"], font=F["small"])
        x0, x1 = lab_w, w - val_w
        self.create_rectangle(x0, h / 2 - 4, x1, h / 2 + 4, fill=C["surface3"], outline="")
        col = C["red"] if self._dim else self.color
        self.create_rectangle(x0, h / 2 - 4, x0 + (x1 - x0) * self._frac, h / 2 + 4, fill=col, outline="")
        self.create_text(w, h / 2, text=self._text, anchor="e", fill=C["text"], font=F["mono_small"])
