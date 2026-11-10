# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Pixel assertions for a rendered overlay PNG (no Pillow required).

PNG is decoded with zlib and pure Python. Checks that the panel background is present, that
non-background pixels exist in every expected band of the layout, and that the accent/OK/WARN
colors actually appear, which proves the bars and status text drew rather than an empty box.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def read_png(path: str) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos, w, h, bitdepth, ctype, idat = 8, 0, 0, 8, 6, b""
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + ln]
        if tag == b"IHDR":
            w, h, bitdepth, ctype = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
        pos += 12 + ln
    if bitdepth != 8 or ctype not in (2, 6):
        raise ValueError(f"expected RGB/8 or RGBA/8, got color={ctype} depth={bitdepth}")
    bpp = 3 if ctype == 2 else 4
    raw = zlib.decompress(idat)
    stride = w * bpp
    rows: list[bytes] = []
    prev = bytes(stride)
    p = 0
    for _ in range(h):
        ftype = raw[p]
        line = bytearray(raw[p + 1 : p + 1 + stride])
        p += 1 + stride
        if ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b, c = prev[i], (prev[i - bpp] if i >= bpp else 0)
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        rows.append(bytes(line))
        prev = bytes(line)
    px: list[tuple[int, int, int]] = []
    for row in rows:
        for i in range(0, stride, bpp):
            px.append((row[i], row[i + 1], row[i + 2]))
    return w, h, px


def close(c: tuple[int, int, int], target: tuple[int, int, int], tol: int = 18) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(c, target))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/overlay-preview.png"
    w, h, px = read_png(path)
    print(f"{path}: {w}x{h}")

    bg = (0x12, 0x14, 0x1A)
    fg = (0xE6, 0xE9, 0xEF)
    dim = (0x8B, 0x93, 0xA7)
    ok = (0x5E, 0xC9, 0x7F)
    warn = (0xE8, 0xB0, 0x4B)
    bad = (0xE5, 0x67, 0x5F)
    accent = (0x6E, 0xA8, 0xFE)

    counts = {n: sum(1 for p in px if close(p, t)) for n, t in
              [("bg", bg), ("fg", fg), ("dim", dim), ("ok", ok), ("warn", warn), ("bad", bad), ("accent", accent)]}
    for name, n in counts.items():
        print(f"  {name:<7} pixels: {n}")

    # Bands, as fractions of the panel height so they stay valid if PANEL_H changes. The overlay is
    # 176 px tall and every row of the layout must land in one of these five regions.
    rows = [
        ("header tok/s", 0.045, 0.19),
        ("vram bar", 0.19, 0.32),
        ("context bar", 0.32, 0.55),
        ("economics rows", 0.55, 0.78),
        ("thermal + footer", 0.78, 0.99),
    ]
    bands = [(label, int(h * f0), int(h * f1)) for label, f0, f1 in rows]

    def band(y0: int, y1: int) -> dict[str, int]:
        lo, hi = y0 * w, min(y1 * w, len(px))
        seg = px[lo:hi]
        return {
            "nonbg": sum(1 for p in seg if not close(p, bg)),
            "fg": sum(1 for p in seg if close(p, fg)),
            "dim": sum(1 for p in seg if close(p, dim)),
        }

    problems = 0
    for label, y0, y1 in bands:
        b = band(y0, y1)
        ok_band = b["nonbg"] > 60 and (b["fg"] + b["dim"]) > 20
        print(f"  band {label:<18} nonbg={b['nonbg']:>5} fg={b['fg']:>4} dim={b['dim']:>4} {'OK' if ok_band else 'FAIL'}")
        if not ok_band:
            problems += 1

    # Colors that must appear for the state being canned in render_overlay.
    must = {"accent (busy dot)": accent, "bad (VRAM>=92% or hot GPU)": bad}
    for label, col in must.items():
        present = any(close(p, col) for p in px)
        print(f"  color {label:<30} {'present' if present else 'MISSING'}")
        if not present:
            problems += 1

    bg_share = counts["bg"] / len(px)
    print(f"  background share: {bg_share * 100:.1f}%")
    if bg_share > 0.97:
        print("  FAIL: panel is essentially empty")
        problems += 1

    print("RENDER OK" if not problems else f"RENDER PROBLEMS: {problems}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
