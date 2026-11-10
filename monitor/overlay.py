# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Corner overlay for Windows: VRAM, context, speed, always on top.

tkinter only. Frameless, optionally click-through, never steals focus, sits in a screen corner
and drags with the mouse when click-through is off. All drawing lives in monitor.panel so the
exact same code path is exercised by tests that need no display; this module owns only the
window, the fonts and the poll loop.

Run:  python -m monitor.overlay            (default bottom-right)
      python -m monitor.overlay --corner top-left --scale 1.2 --no-click-through
"""

from __future__ import annotations

import argparse
import ctypes
import tkinter as tk
from tkinter import font as tkfont

from monitor import panel
from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP, Sample, fetch_engine, fetch_swap
from monitor.economics import context_economics, speculation
from monitor.vram import Calibration, growth_projection, headroom

CORNERS = {
    "bottom-right": (1, 1),
    "bottom-left": (0, 1),
    "top-right": (1, 0),
    "top-left": (0, 0),
}

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
GA_ROOT = 0


class Fonts:
    """Real tkinter fonts, plus the role accessors panel.draw uses."""

    def __init__(self, scale: float) -> None:
        base = max(9, int(10 * scale))
        self._mono = tkfont.Font(family="Consolas", size=base)
        self._bold = tkfont.Font(family="Consolas", size=base, weight="bold")
        self._small = tkfont.Font(family="Consolas", size=max(8, base - 2))

    def bold(self) -> tkfont.Font:
        return self._bold

    def small(self) -> tkfont.Font:
        return self._small

    def mono(self) -> tkfont.Font:
        return self._mono


class _Canvas:
    """Adapts a tkinter Canvas to panel.draw's duck-typed interface."""

    def __init__(self, cv: tk.Canvas, fonts: Fonts) -> None:
        self.cv, self.fonts = cv, fonts

    def create_text(self, x: int, y: int, anchor: str = "", fill: str = "", text: str = "", font: object = None) -> None:
        role = getattr(font, "role", None)
        real = {"bold": self.fonts.bold, "small": self.fonts.small}.get(role, self.fonts.mono)() if role else (font or self.fonts.mono())
        self.cv.create_text(x, y, anchor=anchor or "w", fill=fill, text=text, font=real)

    def create_rectangle(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        self.cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline or "")

    def create_oval(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        self.cv.create_oval(x0, y0, x1, y1, fill=fill, outline=outline or "")

    def delete(self, tag: str) -> None:
        self.cv.delete(tag)


def toplevel_hwnd(root: tk.Tk) -> int:
    """Handle of the real top-level window for an overrideredirect Tk root."""
    child = int(root.winfo_id() or 0)
    if not child:
        return 0
    root_hwnd = ctypes.windll.user32.GetAncestor(child, GA_ROOT)
    return root_hwnd or child


class Overlay:
    def __init__(self, engine: str, swap_base: str, corner: str, scale: float, interval: float,
                 click_through: bool = True) -> None:
        self.engine, self.swap_base, self.interval = engine, swap_base, interval
        self.calib = Calibration.load()

        self.root = tk.Tk()
        self.root.title("Inference Monitor")
        self.root.overrideredirect(True)          # no title bar, no taskbar entry
        self.root.attributes("-topmost", True)    # above every window
        self.root.configure(bg=panel.BG)

        self.fonts = Fonts(scale)
        self.W, self.H = int(panel.PANEL_W * scale), int(panel.PANEL_H * scale)
        self._place(corner)

        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H, bg=panel.BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._grab)
        self.canvas.bind("<B1-Motion>", self._drag)

        # Click-through means the panel cannot receive mouse events, so dragging is only wired
        # up when it is off. Otherwise the panel is positioned by --corner alone.
        self._drag_enabled = not click_through
        if click_through:
            self._enable_click_through()

        self._tick()

    def _enable_click_through(self) -> bool:
        try:
            self.root.update_idletasks()
            hwnd = toplevel_hwnd(self.root)
            if not hwnd:
                return False
            user32 = ctypes.windll.user32
            styles = user32.GetWindowLongW(hwnd, GWL_EXSTYLE) or 0
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, styles | WS_EX_LAYERED | WS_EX_TRANSPARENT)
            return True
        except Exception:
            return False

    def _place(self, corner: str) -> None:
        ax, ay = CORNERS.get(corner, CORNERS["bottom-right"])
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        margin = 24
        x = (sw - self.W - margin) if ax else margin
        y = (sh - self.H - margin) if ay else margin
        self.root.geometry(f"{self.W}x{self.H}+{x}+{y}")

    def _grab(self, ev: tk.Event) -> None:
        if not self._drag_enabled:
            return
        self._ox, self._oy = ev.x_root - self.root.winfo_x(), ev.y_root - self.root.winfo_y()

    def _drag(self, ev: tk.Event) -> None:
        if not self._drag_enabled:
            return
        self.root.geometry(f"+{ev.x_root - self._ox}+{ev.y_root - self._oy}")

    def _tick(self) -> None:
        self.render(fetch_engine(self.engine), fetch_swap(self.swap_base))
        self.root.after(int(self.interval * 1000), self._tick)

    def render(self, s: Sample, swap: dict) -> None:
        spec = speculation(s)
        econ = context_economics(s)
        vram = headroom(s, self.calib)
        proj = growth_projection(s.hist_gpu_mem_used, window_s=60.0)

        self.canvas.delete("all")
        panel.draw(
            _Canvas(self.canvas, self.fonts), s, swap,
            {"spare_context_tokens": vram["spare_context_tokens"]},
            {"acceptance_rate": spec.acceptance_rate},
            {
                "cache_hit_rate": econ.cache_hit_rate,
                "io_ratio": econ.io_ratio,
                "mean_prompt_per_request": econ.mean_prompt_per_request,
            },
            proj,
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--swap", default=DEFAULT_SWAP)
    ap.add_argument("--corner", choices=sorted(CORNERS), default="bottom-right")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--no-click-through", action="store_true",
                    help="let the panel take mouse input so it can be dragged (blocks clicks under it)")
    args = ap.parse_args()
    Overlay(args.engine, args.swap, args.corner, args.scale, args.interval,
            click_through=not args.no_click_through).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
