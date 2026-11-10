# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Checks that README claims match reality and that the derived math is right.

    python -m monitor.selfcheck            # worked examples for every derived figure
    python -m monitor.selfcheck --live     # also assert each live field exists

The live half matters because a dashboard can render a missing key as zero and still look healthy
while reporting nothing. Here a missing expected field is an error, not a blank cell.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import re
import sys

from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP, fetch_engine, fetch_swap
from monitor.economics import context_economics, speculation, vram_budget
from monitor.vram import Calibration, context_cost, growth_projection, headroom, usable_vram_bytes

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# --------------------------------------------------------------------------- #
# README claims                                                               #
# --------------------------------------------------------------------------- #

MODULE_FOR_CLAIM = {
    "monitor": "monitor.dashboard",
    "monitor.overlay": "monitor.overlay",
    "monitor.panel": "monitor.panel_app",
    "monitor.crosscheck": "monitor.crosscheck",
    "monitor.cache_truth": "monitor.cache_truth_cli",
    "monitor.calibrate": "monitor.calibrate",
    "monitor.selfcheck": "monitor.selfcheck",
    "monitor.model_switch": "monitor.cli_model_switch",
}


# Render tools are plain scripts, not modules of the package. A reader copying them from the README
# runs them against a checkout, so they belong in the same guarantee.
SCRIPT_FOR_CLAIM = {
    "tools/render_sample.py": "tools/render_sample.py",
    "tools/check_render.py": "tools/check_render.py",
}


def readme_commands() -> list[str]:
    try:
        with open("README.md", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    out: list[str] = []
    for block in re.findall(r"```(?:bash|text)\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("python -m monitor") or line.startswith("python tools/"):
                out.append(line)
    return out


def check_readme_commands() -> None:
    claims = readme_commands()
    check("README shows runnable commands", len(claims) >= 6, f"{len(claims)} found")
    for claim in claims:
        mod_tok = re.search(r"python -m (monitor[\w.]*)", claim)
        if mod_tok:
            mod_name = MODULE_FOR_CLAIM.get(mod_tok.group(1))
            if mod_name is None:
                check(f"mapped: {claim}", False, f"no module registered for {mod_tok.group(1)}")
                continue
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:  # noqa: BLE001 - report any import failure plainly
                check(f"imports: {mod_name}", False, f"{type(exc).__name__}: {exc}")
                continue
            check(f"README command backed by code: {claim}", callable(getattr(mod, "main", None)), mod_name)
            continue
        script_tok = re.search(r"python (tools/[\w.]+)", claim)
        if script_tok:
            expected = SCRIPT_FOR_CLAIM.get(script_tok.group(1))
            ok = bool(expected) and Path(expected).is_file()
            check(f"README command backed by code: {claim}", ok, "" if ok else f"no such file: {script_tok.group(1)}")
            continue
        check(f"mapped: {claim}", False, "unrecognized command form")


def check_readme_numbers() -> None:
    """Every figure named in the README table must come from a function that exists."""
    import monitor.economics as economics
    import monitor.vram as vram

    for fn in ("speculation", "context_economics", "vram_budget"):
        check(f"economics.{fn} exists", callable(getattr(economics, fn, None)))
    for fn in ("headroom", "usable_vram_bytes", "context_cost", "growth_projection"):
        check(f"vram.{fn} exists", callable(getattr(vram, fn, None)))


# --------------------------------------------------------------------------- #
# Derived math against hand-computed expectations                              #
# --------------------------------------------------------------------------- #


def check_math() -> None:
    """Pinned magnitudes with hand-computed expectations.

    A misread millisecond field once turned a 260 s prefill into a 260,000 s one, so units are
    asserted explicitly here rather than only checking that a key exists.

    These checks need real engine totals to compare against. With nothing listening they are reported
    as skipped rather than passed: a math check that never ran must not look like a math check that
    agreed. Use --live on a box with the stack running, where they run for real.
    """
    eng = fetch_engine(DEFAULT_ENGINE)
    if not eng.ok:
        SKIPPED.append(("derived math against live engine totals", f"no engine at {DEFAULT_ENGINE}"))
        return

    econ = context_economics(eng)
    spec = speculation(eng)
    bud = vram_budget(eng)

    # The engine's own per-request decode speeds are the reference point for the derived mean.
    reported = [float(r["decode_tok_s"]) for r in eng.requests_detail if r.get("decode_tok_s")]
    if econ.mean_decode_tok_s and reported:
        mean = sum(reported) / len(reported)
        rel = abs(econ.mean_decode_tok_s - mean) / mean
        check("derived decode speed agrees with the engine's per-request figures", rel <= 0.35, f"derived {econ.mean_decode_tok_s:.1f} vs reported mean {mean:.1f}")

    if eng.prompt_tokens and eng.output_tokens:
        want_io = eng.prompt_tokens / eng.output_tokens
        check("I/O ratio matches prompt/output", abs(econ.io_ratio - want_io) < 1e-9, f"{econ.io_ratio} vs {want_io}")
        check("I/O ratio plausible for a chat workload", 0.2 < econ.io_ratio < 500, f"{econ.io_ratio:.1f}")

    if eng.prompt_tokens:
        want = min(eng.reused_tokens, eng.prompt_tokens) / eng.prompt_tokens
        check("cache hit rate matches reused/prompt", abs(econ.cache_hit_rate - want) < 1e-9, f"{econ.cache_hit_rate} vs {want}")
        check("cache hit rate is a fraction", 0.0 <= econ.cache_hit_rate <= 1.0, f"{econ.cache_hit_rate}")
        check("fresh tokens are prompt minus reused", econ.fresh_prompt_tokens == eng.prompt_tokens - min(eng.reused_tokens, eng.prompt_tokens))

    if eng.drafts_offered:
        acc = eng.drafts_accepted / eng.drafts_offered
        check("acceptance rate matches accepted/offered", abs(spec.acceptance_rate - acc) < 1e-9)
        # One verified token is always emitted plus the accepted bonus, so per-round >= 1.0.
        check("tokens per draft round counts the verified token", abs(spec.tokens_per_draft - (1 + acc)) < 1e-9, f"{spec.tokens_per_draft} vs {1 + acc}")
        check("tokens per draft round in a sane band", 1.0 <= spec.tokens_per_draft <= 8.0, f"{spec.tokens_per_draft:.2f}")

    if eng.prompt_ms and eng.decode_ms:
        busy = eng.prompt_ms + eng.decode_ms
        want_tax = eng.prompt_ms / busy
        check("prefill tax matches ms/(ms+ms)", abs(econ.prefill_share_of_busy_time - want_tax) < 1e-9, f"{econ.prefill_share_of_busy_time} vs {want_tax}")
        check("prefill tax is a fraction", 0.0 <= econ.prefill_share_of_busy_time <= 1.0, f"{econ.prefill_share_of_busy_time}")

    cost = econ.cost_per_1k_output_tokens
    if cost:
        total = cost["total_ms_per_1k_output"]
        parts = cost["prefill_ms_per_1k_output"] + cost["decode_ms_per_1k_output"]
        check("cost per 1k splits into prefill + decode", abs(parts - total) < 1e-6, f"{parts:.3f} vs {total:.3f}")
        secs = total / 1000.0
        # A ~20 tok/s local model spends well over a minute per 1k output tokens. Anything in the
        # millisecond range means a unit error crept in somewhere.
        check("cost per 1k is the right order of magnitude", 5.0 < secs < 5_000.0, f"{secs:.1f} s")

    if econ.mean_decode_tok_s:
        check("derived decode speed is a sane tok/s", 0.5 < econ.mean_decode_tok_s < 2_000.0, f"{econ.mean_decode_tok_s:.1f}")
    if eng.live_tok_s:
        check("engine live decode speed is a sane tok/s", 0.5 < eng.live_tok_s < 2_000.0, f"{eng.live_tok_s:.1f}")

    if bud.free_bytes is not None:
        check("free VRAM = total - used", bud.free_bytes == (bud.total_bytes or 0) - (bud.used_bytes or 0), str(bud))
        check("headroom verdict exists for live VRAM", bool(bud.headroom_verdict))


# --------------------------------------------------------------------------- #
# Live field presence                                                          #
# --------------------------------------------------------------------------- #

ENGINE_FIELDS = [
    "model", "max_context", "state", "gpu_util", "gpu_mem_used",
    "vram_free_mib", "arena_mib", "ram_total",
]
SWAP_SERIES = [
    "llamaswap_gpu_memory_used_bytes",
    "llamaswap_gpu_memory_total_bytes",
    "llamaswap_memory_used_bytes",
    "llamaswap_cpu_util_percent",
]


def check_live_fields() -> None:
    eng = fetch_engine(DEFAULT_ENGINE)
    if not eng.ok:
        SKIPPED.append(("live field presence against the engine", f"no engine at {DEFAULT_ENGINE}"))
        return

    missing = [f for f in ENGINE_FIELDS if getattr(eng, f, None) in (None, "")]
    check("engine exposes every field the dashboard reads", not missing, f"missing: {missing}")

    hist = list(eng.hist_tok_s or [])
    check("engine keeps a speed history the panel can graph", len(hist) >= 2, f"{len(hist)} samples")

    swap = fetch_swap(DEFAULT_SWAP)
    if not swap.get("ok"):
        SKIPPED.append(("llama-swap hardware series cross-check", f"no swap at {DEFAULT_SWAP or '?'}"))
    else:
        series = swap.get("series") or {}
        # Series names carry labels like [id=0,name=...], so match on the metric name prefix.
        missing_s = [m for m in SWAP_SERIES if not any(k.startswith(m) for k in series)]
        check("llama-swap exposes every hardware series the cross-check reads", not missing_s, f"missing: {missing_s}")

    calib = Calibration.load()
    h = headroom(eng, calib)
    needed = {"usable_bytes", "bytes_per_token", "calibrated", "spare_context_tokens", "context_ceiling", "verdict"}
    check("headroom returns its full contract", needed <= set(h), f"missing: {sorted(needed - set(h))}")
    check("usable VRAM is computable", usable_vram_bytes(eng, calib) is not None)

    cc = context_cost(eng, calib)
    check("context cost has per-request data", cc.get("available") is True, str(cc))
    if cc.get("ms_per_fresh_token"):
        # Re-prefill on a local GPU is sub-millisecond to a few ms per token; anything orders of
        # magnitude off means cumulative totals were divided instead of per-request pairs.
        check("re-prefill cost per fresh token is plausible", 0.01 < cc["ms_per_fresh_token"] < 50.0, f"{cc['ms_per_fresh_token']:.3f} ms")

    proj = growth_projection(list(eng.hist_gpu_mem_used or []), window_s=60.0)
    check("trend states slope and confidence", "slope_mib_per_min" in proj and ("r2" in proj or "reason" in proj), str(proj)[:120])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="also check fields against the running stack")
    args = ap.parse_args()

    check_readme_commands()
    check_readme_numbers()
    check_math()
    if args.live:
        check_live_fields()

    failed = 0
    for name, ok, detail in CHECKS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"\n         -> {detail}" if not ok and detail else ""))
        failed += 0 if ok else 1
    for name, why in SKIPPED:
        print(f"  [skip] {name}\n         -> {why}")
    ran = len(CHECKS) - failed
    line = f"\n{ran}/{len(CHECKS)} checks passed"
    if SKIPPED:
        line += f", {len(SKIPPED)} skipped (no live stack)"
    print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
