# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Deterministic renderer for the overlay, shared by the live app and the test suite.

The drawing code is a pure function over an abstract canvas object so it can be exercised
without a display: the real app passes a tkinter Canvas, the tests pass a recorder that logs
every primitive. Screen capture is deliberately not used for verification because a grabbed
frame depends on window stacking, DWM composition and whether the window is on screen at all;
those are environment facts, not properties of this code.

Colors and layout constants live here so overlay.py and the tests cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BG = "#12141a"
FG = "#e6e9ef"
DIM = "#8b93a7"
OK = "#5ec97f"
WARN = "#e8b04b"
BAD = "#e5675f"
ACCENT = "#6ea8fe"
TRACK = "#262b36"

PANEL_W, PANEL_H = 300, 176
PAD = 12


@dataclass
class Prim:
    kind: str            # "text" | "rect" | "oval"
    color: str
    x: int
    y: int
    text: str = ""
    font: str = ""       # "bold" | "small" | "mono"
    x1: int = 0
    y1: int = 0


@dataclass
class Recorder:
    """Stands in for a tkinter Canvas and records what would have been drawn."""

    prims: list[Prim] = field(default_factory=list)

    def create_text(self, x: int, y: int, anchor: str = "", fill: str = "", text: str = "", font: object = None) -> None:
        self.prims.append(Prim("text", fill, x, y, str(text), _font_name(font)))

    def create_rectangle(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        self.prims.append(Prim("rect", fill, x0, y0, "", "", x1, y1))

    def create_oval(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        self.prims.append(Prim("oval", fill, x0, y0, "", "", x1, y1))

    def delete(self, _tag: str) -> None:
        self.prims.clear()


def _font_name(font: object) -> str:
    """Classify a font by role, whether it is a panel._F stand-in or a real tkinter font."""
    role = getattr(font, "role", None)
    if role:
        return str(role)
    actual = getattr(font, "actual", None)
    if actual is None:
        return "mono"
    try:
        info = actual()
    except Exception:
        return "mono"
    weight = str(info.get("weight", "")).lower()
    size = info.get("size", 0)
    try:
        size = abs(int(size))
    except (TypeError, ValueError):
        size = 10
    if weight == "bold":
        return "bold"
    if size < 10:
        return "small"
    return "mono"


def color_for(frac: float | None) -> str:
    """Traffic-light ramp used for every utilization bar in the panel."""
    if frac is None:
        return DIM
    if frac >= 0.92:
        return BAD
    if frac >= 0.78:
        return WARN
    return OK


def draw(rec: Recorder, s: object, swap: dict, vram: dict, spec: dict, econ: dict, proj: dict) -> None:
    """Render one frame onto anything with canvas-like methods.

    ``s`` is duck-typed so a Sample or a plain stand-in works; every field access goes through
    getattr with a default so a partially populated sample still renders instead of raising.
    """
    W = PANEL_W
    pad = PAD
    y = pad

    g = lambda name, dflt=None: getattr(s, name, dflt)  # noqa: E731 - terse local accessor

    if not g("ok", False):
        rec.create_text(pad, y, fill=BAD, text="engine offline", font=_F("bold"))
        rec.create_text(pad, y + 20, fill=DIM, text=str(g("error") or "")[:46], font=_F("small"))
        return

    busy = str(g("state", "idle")) in ("generating", "prefilling", "prompt_eval")

    hist = g("hist_tok_s", []) or []
    speed = g("live_tok_s") if g("live_tok_s") is not None else (hist[-1] if hist else 0.0)
    rec.create_oval(pad, y + 6, pad + 10, y + 16, fill=(ACCENT if busy else DIM))
    rec.create_text(pad + 18, y, fill=FG, text=f"{speed or 0:5.1f} tok/s", font=_F("bold"))
    phase = g("phase") or g("state", "idle")
    rec.create_text(W - pad, y, anchor="ne", fill=DIM, text=str(phase)[:22], font=_F("small"))
    y += 26

    used, total = g("gpu_mem_used"), g("gpu_mem_total")
    if used is not None and total:
        frac = used / total
        rec.create_text(pad, y, fill=DIM, text="VRAM", font=_F("small"))
        rec.create_text(
            W - pad, y, anchor="e", fill=color_for(frac),
            text=f"{used / (1024 * 1024):.0f}/{total / (1024 * 1024):.0f} MiB", font=_F("mono"),
        )
        y += 15
        _bar(rec, pad, y, W - 2 * pad, frac, color_for(frac))
        y += 9
        free = g("vram_free_mib")
        if free is not None:
            rec.create_text(pad, y, fill=DIM, text=f"engine free {free} MiB", font=_F("small"))

    y += 18
    ctx_used = int(econ.get("mean_prompt_per_request") or 0)
    ceiling = g("max_context", 0) or 1
    frac_ctx = min(ctx_used / ceiling, 1.0)
    rec.create_text(pad, y, fill=DIM, text="CONTEXT", font=_F("small"))
    spare = vram.get("spare_context_tokens")
    spare_txt = f"+{spare:,} spare" if spare is not None else "calibrate for spare"
    rec.create_text(W - pad, y, anchor="e", fill=FG, text=f"{ctx_used:,}/{ceiling:,}", font=_F("mono"))
    y += 15
    _bar(rec, pad, y, W - 2 * pad, frac_ctx, color_for(frac_ctx))
    y += 9
    rec.create_text(pad, y, fill=DIM, text=spare_txt, font=_F("small"))

    y += 20
    hit = econ.get("cache_hit_rate")
    io_ = econ.get("io_ratio")
    rec.create_text(pad, y, fill=DIM, text="KV RESIDENT", font=_F("small"))
    rec.create_text(pad + 96, y, fill=color_for(hit), text=(f"{hit * 100:4.0f}%" if hit is not None else " n/a"), font=_F("mono"))
    if io_ is not None:
        rec.create_text(W - pad, y, anchor="e", fill=FG, text=f"I/O {io_:.0f}", font=_F("mono"))

    y += 17
    acc = spec.get("acceptance_rate")
    rec.create_text(pad, y, fill=DIM, text="DRAFTS", font=_F("small"))
    rec.create_text(pad + 96, y, fill=color_for(acc), text=(f"{acc * 100:4.0f}%" if acc is not None else " n/a"), font=_F("mono"))
    trend = ""
    if proj.get("available") and proj.get("projectable"):
        trend = f"VRAM {proj['slope_mib_per_min']:+.0f} MiB/min"
    rec.create_text(W - pad, y, anchor="e", fill=(WARN if trend else DIM), text=trend, font=_F("small"))

    y += 19
    gpu = swap.get("gpu") or {}
    util = g("gpu_util") if g("gpu_util") is not None else gpu.get("util")
    temp = g("gpu_temp") if g("gpu_temp") is not None else gpu.get("temp_celsius")
    power = g("gpu_power") if g("gpu_power") is not None else gpu.get("power_w")
    limit = g("gpu_power_limit") or gpu.get("power_limit_w")
    bits: list[str] = []
    hot = False
    if util is not None:
        bits.append(f"GPU {util:.0f}%")
    if temp is not None:
        over = float(temp) >= 80
        hot = hot or over
        bits.append(f"{temp:.0f}C" + ("!" if over else ""))
    if power is not None and limit:
        near = float(power) / float(limit) >= 0.9
        hot = hot or near
        bits.append(f"{power:.0f}/{limit:.0f}W" + ("!" if near else ""))
    if g("cpu") is not None:
        bits.append(f"CPU {g('cpu'):.0f}%")
    rec.create_text(pad, y, fill=(BAD if hot else DIM), text=" ".join(bits), font=_F("small"))

    rec.create_text(pad, PANEL_H - pad, anchor="sw", fill=DIM, text=str(g("model") or "?")[:40], font=_F("small"))


def _bar(rec: Recorder, x: int, y: int, w: int, frac: float | None, color: str) -> None:
    rec.create_rectangle(x, y, x + w, y + 5, fill=TRACK)
    if frac and frac > 0:
        rec.create_rectangle(x, y, x + max(1, int(w * min(frac, 1.0))), y + 5, fill=color)


class _F:
    """Font stand-in that carries a role name; the tkinter path swaps in real tkfont objects."""

    def __init__(self, role: str) -> None:
        self.role = role

    def actual(self, **_kw: object) -> dict[str, object]:
        return {"weight": "bold" if self.role == "bold" else "normal", "size": 8 if self.role == "small" else 10}
