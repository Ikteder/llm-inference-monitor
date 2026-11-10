# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Always-on-top desktop panel for the live engine figures.

    python -m monitor.panel [--engine URL] [--swap URL] [--interval 1.5]

The window is a thin shell: rendering lives in monitor.panel's draw() so it can be verified headless,
and this module only owns the tkinter loop and the refresh timer. Requires tkinter, which ships with
the standard Windows CPython installer but not with every minimal or embedded build.
"""

from __future__ import annotations

import argparse
import sys

from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP, fetch_engine, fetch_swap
from monitor.economics import context_economics, speculation
from monitor.panel import PANEL_H, PANEL_W, draw
from monitor.vram import Calibration, growth_projection, headroom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--swap", default=DEFAULT_SWAP)
    ap.add_argument("--interval", type=float, default=1.5, help="refresh seconds")
    args = ap.parse_args()

    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except ImportError:
        print("tkinter is not available in this interpreter; use 'python -m monitor' for the terminal dashboard")
        return 1

    root = tk.Tk(className="llm-inference-monitor")
    root.title("Inference Monitor")
    root.geometry(f"{PANEL_W}x{PANEL_H}")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.configure(bg="#12141a")

    canvas = tk.Canvas(root, width=PANEL_W, height=PANEL_H, highlightthickness=0, bg="#12141a")
    canvas.pack()

    bold = tkfont.Font(family="Consolas", size=12, weight="bold")
    mono = tkfont.Font(family="Consolas", size=10)
    small = tkfont.Font(family="Consolas", size=8)
    fonts = {"bold": bold, "mono": mono, "small": small}

    class _TkCanvas:
        """Adapts the real canvas to the role-based font API draw() expects."""

        def create_text(self, x, y, anchor="", fill="", text="", font=None):
            canvas.create_text(x, y, anchor=anchor, fill=fill, text=text, font=fonts.get(getattr(font, "role", "mono"), mono))

        def create_rectangle(self, x0, y0, x1, y1, fill="", outline=""):
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline or "", width=0)

        def create_oval(self, x0, y0, x1, y1, fill="", outline=""):
            canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline or "", width=0)

        def delete(self, tag):
            canvas.delete(tag)

    calib = Calibration.load()

    def refresh() -> None:
        s = fetch_engine(args.engine)
        swap = fetch_swap(args.swap)
        if not s.ok:
            econ = spec = {}
            vram = {"spare_context_tokens": None}
            proj = {}
        else:
            econ = _as_dict(context_economics(s))
            spec = _as_dict(speculation(s))
            vram = headroom(s, calib)
            proj = growth_projection(list(s.hist_gpu_mem_used or []), window_s=60.0)

        def g(name, dflt=None):
            return getattr(s, name, dflt)

        class _S:
            pass

        view = _S()
        for k in ("ok", "error", "state", "phase", "model", "max_context", "hist_tok_s", "live_tok_s",
                  "gpu_mem_used", "gpu_mem_total", "vram_free_mib", "gpu_util", "gpu_temp", "gpu_power",
                  "gpu_power_limit", "cpu"):
            setattr(view, k, g(k))

        canvas.delete("all")
        draw(_TkCanvas(), view, swap, vram, spec, econ, proj)
        root.after(int(args.interval * 1000), refresh)

    def _as_dict(obj) -> dict:
        from dataclasses import asdict, is_dataclass

        return asdict(obj) if is_dataclass(obj) else {}

    refresh()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
