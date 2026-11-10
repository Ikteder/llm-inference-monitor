# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Render the overlay to a PNG without a screen, using only the standard library.

The panel is drawn by ``monitor.panel.draw``, the same code path the live app uses, onto a small
software canvas that implements just the primitives that function calls. The result is written as
a PNG with zlib and struct. No display, no window manager, no Pillow, so this runs headless in CI
and produces byte-identical output for a given sample.

Screen capture was the previous approach here and it is not verifiable: what comes back depends on
window stacking, DWM composition and whether the window happens to be on screen at all. A grab of
an off-screen window returns an all-black frame on this machine, which looks like a rendering bug
and is not one. Drawing into our own buffer removes that class of false result.

Usage: python tools/render_overlay.py [--out docs/assets/overlay-preview.png]
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitor import panel  # noqa: E402
from monitor.collect import Sample  # noqa: E402
from monitor.economics import context_economics, speculation  # noqa: E402
from monitor.vram import Calibration, growth_projection  # noqa: E402
from tools.render_sample import canned_sample  # noqa: E402

RGB = tuple[int, int, int]


def _parse_hex(value: str) -> RGB:
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))

COLORS = {name: _parse_hex(getattr(panel, name))
          for name in ("BG", "FG", "DIM", "OK", "WARN", "BAD", "ACCENT", "TRACK")}

# 5x7 bitmap set, enough to draw legible text at panel scale and small enough to keep this file
# readable. The overlay itself uses real Consolas; these glyphs exist only so the verification
# render has pixels that correspond one-to-one with the primitives being asserted on.
GLYPHS: dict[str, tuple[str, ...]] = {
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    "!": (".###.", ".###.", ".###.", ".###.", ".###.", ".....", ".###."),
    ",": (".....", ".....", ".....", ".....", ".###.", ".###.", "..#.."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "/": ("....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."),
    "%": ("##..#", "##.#.", "..#..", ".#...", "#..##", "#.###", "....."),
    ":": (".....", ".###.", ".###.", ".....", ".###.", ".###.", "....."),
    ".": (".....", ".....", ".....", ".....", ".###.", ".###.", "....."),
    "=": (".....", "#####", "#####", ".....", "#####", ".....", "....."),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#...#", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "#...#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", "####.", "....#", "#...#", ".###."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "...#.", "..#..", ".#...", "#....", "#....", "#####"),
    "_": (".....", ".....", ".....", ".....", ".....", ".....", "#####"),
}


class Canvas:
    """Software canvas: the primitive set panel.draw uses, rasterized into an RGB buffer."""

    def __init__(self, w: int = panel.PANEL_W, h: int = panel.PANEL_H) -> None:
        self.w, self.h = w, h
        self.px: list[list[RGB]] = [[COLORS["BG"] for _ in range(w)] for _ in range(h)]

    # -- primitives -------------------------------------------------------- #

    def create_rectangle(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        self._fill_rect(int(x0), int(y0), int(x1), int(y1), _parse_hex(fill) if fill else COLORS["BG"])

    def create_oval(self, x0: int, y0: int, x1: int, y1: int, fill: str = "", outline: str = "") -> None:
        col = _parse_hex(fill) if fill else COLORS["BG"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = max(0.5, (x1 - x0) / 2.0), max(0.5, (y1 - y0) / 2.0)
        for yy in range(int(y0), int(y1) + 1):
            for xx in range(int(x0), int(x1) + 1):
                if ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0:
                    self._set(xx, yy, col)

    def create_text(self, x: int, y: int, anchor: str = "", fill: str = "", text: str = "", font: object = None) -> None:
        col = _parse_hex(fill) if fill else COLORS["FG"]
        self._text(int(x), int(y), str(text), col, anchor or "w")

    def delete(self, _tag: str = "") -> None:
        for row in self.px:
            for i in range(self.w):
                row[i] = COLORS["BG"]

    # -- rasterizer -------------------------------------------------------- #

    def _set(self, x: int, y: int, col: RGB) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y][x] = col

    def _fill_rect(self, x0: int, y0: int, x1: int, y1: int, col: RGB) -> None:
        for yy in range(min(y0, y1), max(y0, y1) + 1):
            for xx in range(min(x0, x1), max(x0, x1) + 1):
                self._set(xx, yy, col)

    def _text(self, x: int, y: int, text: str, col: RGB, anchor: str) -> None:
        width = len(text) * 6 - 1
        if "e" in anchor:
            x -= width
        elif "n" in anchor or "w" in anchor:
            pass
        else:  # center-ish default for "e"/"w" combos we do not use
            x -= width // 2
        top = y - 3
        for ch in text:
            glyph = GLYPHS.get(ch.upper(), GLYPHS.get(ch))
            if glyph is None:
                # Unknown codepoint: draw a visible box so a missing glyph shows up as ink rather
                # than silently vanishing and making an assertion pass on nothing.
                glyph = ("#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####")
            for row_i, row in enumerate(glyph):
                for col_i, bit in enumerate(row):
                    if bit == "#":
                        self._set(x + col_i, top + row_i, col)
            x += 6

    # -- output ------------------------------------------------------------ #

    def to_png(self) -> bytes:
        raw = bytearray()
        for row in self.px:
            raw.append(0)  # filter type 0 (None) per scanline
            for r, g, b in row:
                raw += bytes((r, g, b))

        def chunk(tag: bytes, body: bytes) -> bytes:
            return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b"")
        )


def render_frame(s: Sample) -> Canvas:
    """Draw one frame from a sample through the real panel code path."""
    calib = Calibration.load()
    econ = context_economics(s)
    spec = speculation(s)
    vram = panel_headroom(s, calib)
    proj = growth_projection(list(s.hist_gpu_mem_used or []), window_s=60.0)

    cv = Canvas()
    panel.draw(
        cv, s, {"ok": False, "error": "render check: swap not used", "gpu": {}, "models": []},
        {"spare_context_tokens": vram["spare_context_tokens"]},
        {"acceptance_rate": spec.acceptance_rate},
        {
            "cache_hit_rate": econ.cache_hit_rate,
            "io_ratio": econ.io_ratio,
            "mean_prompt_per_request": econ.mean_prompt_per_request,
        },
        proj,
    )
    return cv


def panel_headroom(s: Sample, calib: Calibration) -> dict[str, object]:
    """Headroom for the render path.

    A calibrated record is forced here so the screenshot exercises the spare-tokens branch rather than the
    "calibrate me" placeholder. The coefficient is a render-check stand-in; only ``monitor.calibrate``
    produces one that may be quoted. The live app reads whatever is on disk.
    """
    from monitor.vram import headroom

    if not calib.calibrated:
        calib = Calibration(
            bytes_per_token=704.0,
            reserved_floor_bytes=300 * 1024 * 1024,
            calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            engine_version=s.engine_version,
            gpu_name=s.gpu_name,
            notes=["render-check placeholder; not a measurement"],
        )
    return headroom(s, calib)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/assets/overlay-preview.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_frame(canned_sample()).to_png())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
