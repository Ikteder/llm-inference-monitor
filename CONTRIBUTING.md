# Contributing

The point of this monitor is that its numbers can be trusted, so the bar for a change is "the offline
suite still says the same thing on a clean machine with nothing listening".

## Ground rules

- **Runtime stays stdlib only.** The `monitor` package imports nothing outside the standard library.
  tkinter is allowed only in the optional desktop panel. If a change needs a new runtime import, argue
  for it in an issue first.
- **No paid APIs, no model calls in tests.** Timing probes use a fake transport with pinned sleeps. A
  test that needs a key or a GPU cannot be merged, because CI has neither and the result would not be
  reproducible.
- **Measurement and derivation are separate.** `collect.py` and `prom.py` only read; `economics.py` and
  `vram.py` only compute. A function that does both is hard to test at the boundary where unit errors
  hide, so keep the split.
- **Never invent a number.** If a field is missing, render "n/a" and say so in the verdict. A dashboard
  that quietly substitutes zero looks healthy while reporting nothing, which is worse than a gap.
- **Distinguish what a probe cannot measure from a failure.** The cache-truth probe reports `inconclusive`
  when its passes are too cheap for wall-clock timing to resolve a cache claim. Do not lower that floor to
  make a verdict sharper; raise the prompt size instead.
- **Live tests self-skip.** Anything touching a running engine or llama-swap is marked so CI on a clean
  runner stays green while the same test exercises the real stack on a dev box.

## Before opening a PR

```bash
python -m pytest -q                 # expect 48 passed, no skips on a box with the stack running
python -m monitor.selfcheck         # README claims and entry points, no services needed
python -m monitor.selfcheck --live  # also asserts each live field the dashboard reads exists
python tools/render_sample.py       # deterministic render of one panel frame to PNG
python tools/check_render.py        # asserts every band of that PNG has content and sane color mix
```

All five must pass. CI runs the pytest, selfcheck and render legs across Python 3.10 to 3.14. The
selfcheck exits non-zero on a clean runner only for real breakage: checks that need a live stack are
reported as skipped rather than passed, so a check that never ran cannot look like one that agreed.

The render checks exist because a visual layer is otherwise unverifiable in CI. Screen capture is not
used: a grabbed frame depends on window stacking, composition and whether the window is on screen at
all, which are environment facts rather than properties of this code. `monitor/panel.py` therefore draws
to an abstract canvas, and `tools/render_overlay.py` rasterizes that same draw path into a PNG with pure
Python so the geometry can be asserted numerically.

## Reporting a problem

Include the exact command, full output, your Python version, which server exposes metrics on your engine
and what it returns for `/metrics`, plus whether llama-swap is in front of the engine. Metrics fields vary
by build, and "the panel shows n/a" is only actionable once the missing field is named. `selfcheck --live`
prints exactly which expected fields a given build lacks.

## Scope

This tool reads and derives. It does not tune inference servers, patch engines, or benchmark model
quality. Model switching exists only as a guarded action against llama-swap with a preview of what a swap
costs; anything beyond that belongs upstream.

## License

This project is licensed under the Apache License, Version 2.0. Contributions are accepted under the same
terms: anything you submit here ships as Apache 2.0 and carries an `SPDX-License-Identifier: Apache-2.0`
header line.
