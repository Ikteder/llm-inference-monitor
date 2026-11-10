# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Calibration probe: measure real VRAM growth per resident context token.

The headroom figure in economics.py needs a bytes-per-token coefficient. A KV geometry formula can
predict it, but only this probe measures what the build actually does, including whatever allocator
slack and expert cache it adds on top. Run it against a live engine, then feed the printed coefficient
to ``Calibration.save()`` so headroom stops being an estimate.

Sends prompts of increasing size and reads /metrics before and after each one. Growth is regressed over
pairs of probes rather than against the requested token count, because engines tokenize their own way:
the number in the request and the number the engine counts are not the same number, and using the
request-side figure biases the slope.

Usage:
    python -m monitor.calibrate [--engine URL] [--steps 512,4096,16384,49152]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request

MB = 1024 * 1024

from monitor.calibration import Calibration, fit, slopes
from monitor.collect import DEFAULT_ENGINE, http_json
from monitor.prom import as_float, as_int, first_present

# Roughly 11 characters per token for this filler. Only used to size a request, never to compute a rate,
# so the imprecision cannot leak into any reported number.
_CHARS_PER_TOKEN = 11.0
_INSTRUCTION_TOKENS = 60


def post_chat(base: str, prompt: str, max_tokens: int = 8, timeout: float = 300.0) -> dict:
    """One chat completion. Output is capped so decode time cannot dominate the probe."""
    body = json.dumps(
        {
            "model": "calibration",
            "messages": [{"role": "user", "content": prompt + "\nReply with the single word: ok"}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed localhost URL)
        return json.loads(resp.read().decode("utf-8"))


def snapshot(base: str) -> dict[str, float]:
    """The subset of /metrics a calibration step needs, normalized to bytes and MiB.

    Field presence varies by build, so missing values come back as 0.0 rather than None: the caller
    subtracts them, and a probe that reads nothing produces a zero slope instead of a crash mid-run.
    """
    payload = http_json(f"{base.rstrip('/')}/metrics")
    hw = payload.get("hardware") or {}
    eng = payload.get("engine") or {}
    return {
        "used_bytes": as_float(first_present(hw, "gpu_mem_used", default=0.0)) or 0.0,
        "free_mib": as_float(first_present(eng, "vram_free_mib", default=0.0)) or 0.0,
    }


def prompt_for(target_tokens: int) -> str:
    """Filler text sized to reach roughly ``target_tokens`` after tokenization."""
    words = max(int(target_tokens) - _INSTRUCTION_TOKENS, 0) // 11 + 1
    filler = "The quick brown fox jumps over the lazy dog while the kernel logs a warning. "
    return filler * words


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure VRAM growth per resident context token.")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--steps", default="512,4096,16384,49152", help="comma separated prompt token targets")
    ap.add_argument(
        "--save",
        action="store_true",
        help="write the fitted coefficient to the calibration file used by monitor.headroom",
    )
    args = ap.parse_args()

    sizes = [int(x) for x in args.steps.split(",") if x.strip()]
    print(f"engine: {args.engine}")

    rows: list[tuple[int, float]] = []
    for target in sizes:
        before = snapshot(args.engine)
        t0 = time.time()
        try:
            out = post_chat(args.engine, prompt_for(target))
        except Exception as exc:  # noqa: BLE001 - a probe run should report any failure and keep going
            print(f"  {target:>6} tokens -> request failed: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        after = snapshot(args.engine)
        usage = out.get("usage") or {}
        ptok = as_int(first_present(usage, "prompt_tokens", default=0))
        cached = as_int((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        d_bytes = after["used_bytes"] - before["used_bytes"]
        rows.append((ptok, after["used_bytes"]))
        print(
            f"  prompt_tokens={ptok:>6} cached={cached:>6} d_vram={d_bytes / MB:>+9.1f} MiB "
            f"free_after={after['free_mib']:>5.0f} MiB wall={dt:6.2f}s"
        )

    if len(rows) < 2:
        print("not enough successful probes to regress; engine unreachable?")
        return 1

    print("\nslope per probe pair:")
    for label, value in slopes(rows):
        print(f"  {label}: {value:+8.1f} bytes/token")

    got = fit(rows)
    coef = got["bytes_per_token"]
    if coef is None:
        print("\nno positive slope across any pair: VRAM did not grow with context on this run.")
        print("The engine may be evicting between probes, or gpu_mem_used is not a per-process figure here.")
        print("Headroom keeps its default coefficient instead of adopting a meaningless one.")
        return 1
    print(f"\nmedian coefficient: {coef:.1f} bytes/token (spread {got['spread']:.1f}, {int(got['pairs'])} pairs)")

    if args.save:
        cal = Calibration.load()
        cal.bytes_per_token = float(coef)
        cal.calibrated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        try:
            meta = http_json(f"{args.engine.rstrip('/')}/metrics") or {}
            eng = meta.get("engine") or {}
            hw = meta.get("hardware") or {}
            cal.engine_version = str(eng.get("version") or eng.get("engine") or "")
            cal.gpu_name = str(hw.get("gpu_name") or "")
        except Exception:  # noqa: BLE001, S112 - stamping metadata is optional; the coefficient is what matters
            pass
        path = cal.save()
        print(f"saved to {path}")
    else:
        print("re-run with --save to store this for monitor.headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
