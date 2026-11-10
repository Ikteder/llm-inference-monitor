# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Terminal dashboard view.

Stdlib tkinter only, no third-party packages, so it runs on any stock Python on any
machine that can reach an OpenAI-compatible endpoint. Three panes: the live strip (what is
happening right now), the economics strip (what it costs), and the hardware strip (VRAM,
context headroom, thermals).

Run:  python -m monitor            (TUI)
      python -m monitor --json     (one snapshot as JSON, for scripting and CI)
"""

from __future__ import annotations

import argparse
import json
import time

from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP, Sample, fetch_engine, fetch_swap
from monitor.economics import ContextEconomics, Speculation, context_economics, speculation
from monitor.vram import Calibration, context_cost, growth_projection, headroom


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _mib(nbytes: object) -> str:
    if nbytes is None:
        return "n/a"
    return f"{int(nbytes) / (1024 * 1024):.0f} MiB"


def _num(value: object, width: int = 8, digits: int = 1) -> str:
    if value is None:
        return "n/a".rjust(width)
    try:
        return f"{float(value):>{width}.{digits}f}"
    except (TypeError, ValueError):
        return str(value).rjust(width)


def _spark(values: list[float], width: int = 30) -> str:
    vals = [v for v in values if isinstance(v, (int, float))][-width:]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    ramp = " .:-=+*#%@"
    return "".join(ramp[min(len(ramp) - 1, int((v - lo) / span * (len(ramp) - 1)))] for v in vals)


def render(sample: Sample, swap: dict, calib: Calibration) -> str:
    lines: list[str] = []
    bar = "=" * 78

    if not sample.ok:
        lines.append(f"engine unreachable: {sample.error}")
        return "\n".join(lines)

    spec = speculation(sample)
    econ = context_economics(sample)
    vram = headroom(sample, calib)
    cost = context_cost(sample, calib)

    state = sample.state.upper() + (f"/{sample.phase}" if sample.phase else "")
    lines.append(bar)
    lines.append(f"{sample.model}   engine {sample.engine_version}   ctx {sample.max_context:,}   [{state}]")
    lines.append(bar)

    # --- live strip: the speedometer, A ---
    hist = sample.hist_tok_s
    live_tok = sample.live_tok_s if sample.live_tok_s is not None else (hist[-1] if hist else None)
    lines.append(f"  decode now      {_num(live_tok)} tok/s   {_spark(hist)}")
    if sample.live_generated is not None:
        lines.append(
            f"  generating      {sample.live_generated:>8} tokens in {(sample.live_elapsed_s or 0):6.1f}s"
            f"   queued {sample.queued}"
        )

    # --- economics strip: B ---
    lines.append("")
    lines.append("  CONTEXT ECONOMICS (since engine boot)")
    lines.append(
        f"    cache hit       {_pct(econ.cache_hit_rate)} of prompt tokens reused"
        f"   ({econ.reused_tokens:,} / {econ.prompt_tokens:,})"
    )
    lines.append(f"    fresh prefill   {econ.fresh_prompt_tokens:>12,} tokens")
    if econ.io_ratio is not None:
        lines.append(f"    I/O ratio       {econ.io_ratio:>12.1f} in-tokens per out-token")
    if econ.prefill_share_of_busy_time is not None:
        lines.append(
            f"    prefill tax     {_pct(econ.prefill_share_of_busy_time)} of busy time"
            f"   ({econ.mean_prefill_tok_s or 0:,.0f} tok/s prefill vs {econ.mean_decode_tok_s or 0:.1f} tok/s decode)"
        )
    if econ.cost_per_1k_output_tokens:
        c = econ.cost_per_1k_output_tokens
        lines.append(
            f"    cost /1k out    {c['total_ms_per_1k_output'] / 1000:>12.1f} s total"
            f"   ({c['prefill_ms_per_1k_output'] / 1000:.1f} s prefill + {c['decode_ms_per_1k_output'] / 1000:.1f} s decode)"
        )
    # --- speculation: D ---
    lines.append("")
    lines.append(f"  SPECULATIVE DECODING ({spec.drafting}, depth {spec.spec_depth})")
    lines.append(
        f"    acceptance      {_pct(spec.acceptance_rate)}"
        f"   ({spec.drafts_accepted:,} of {spec.drafts_offered:,} drafts accepted)"
    )
    if spec.tokens_per_draft is not None:
        lines.append(f"    tokens / draft  {spec.tokens_per_draft:>12.2f}  mean bonus {spec.mean_bonus_tokens or 0:.2f}")
    lines.append(f"    verdict         {spec.verdict}")

    # --- VRAM + context headroom: the overlay's headline numbers ---
    lines.append("")
    lines.append("  VRAM AND CONTEXT BUDGET")
    if sample.gpu_mem_used is not None:
        pct = (sample.gpu_mem_used / sample.gpu_mem_total * 100) if sample.gpu_mem_total else 0
        lines.append(
            f"    driver VRAM     {_mib(sample.gpu_mem_used)} / {_mib(sample.gpu_mem_total)}"
            f"   ({pct:4.1f}%)   engine free {sample.vram_free_mib if sample.vram_free_mib is not None else 'n/a'} MiB"
        )
    lines.append(
        f"    usable now      {_mib(vram['usable_bytes']):>12}"
        f"   bytes/token {vram['bytes_per_token'] if vram['bytes_per_token'] else 'uncalibrated'}"
    )
    spare = vram["spare_context_tokens"]
    lines.append(
        f"    spare context   {(f'{spare:,}' if spare is not None else 'unknown'):>12}"
        f"   tokens (ceiling {vram['context_ceiling']:,})"
    )
    lines.append(f"    verdict         {vram['verdict']}")
    if cost.get("available"):
        mft = cost.get("ms_per_fresh_token")
        rate = cost.get("prefill_tok_s_from_requests")
        lines.append(
            f"    mean prompt     {cost['mean_prompt_tokens']:>12,.0f} tokens"
            f"   re-prefill {mft:.1f} ms/fresh token ({rate:,.0f} tok/s)"
            if mft else
            f"    mean prompt     {cost['mean_prompt_tokens']:>12,.0f} tokens"
            "   (turns were cache hits, no fresh-token slope)"
        )

    proj = growth_projection(sample.hist_gpu_mem_used, window_s=60.0)
    if proj.get("available"):
        flag = "leak risk" if proj["projectable"] else "flat/noisy"
        lines.append(
            f"    VRAM trend      {proj['slope_mib_per_min']:>+12.1f} MiB/min"
            f"   R2={proj['r2']:.2f} ({flag}, {proj['horizon_min']:.0f} min horizon"
            f" -> {proj['projected_delta_mib_over_horizon']:+.0f} MiB)"
        )
    else:
        lines.append(f"    VRAM trend      {proj.get('reason')}")

    # --- hardware strip ---
    lines.append("")
    gpu = swap.get("gpu") or {}
    lines.append(f"  {sample.gpu_name or 'GPU'}")
    util = sample.gpu_util if sample.gpu_util is not None else gpu.get("util")
    temp = sample.gpu_temp if sample.gpu_temp is not None else gpu.get("temp_celsius")
    power = sample.gpu_power if sample.gpu_power is not None else gpu.get("power_w")
    limit = sample.gpu_power_limit or gpu.get("power_limit_w")
    pwr = f"{power:.0f}/{limit:.0f} W" if (power is not None and limit) else f"{power if power is not None else 'n/a'} W"
    lines.append(
        f"    util {_num(util, 5)}%   temp {_num(temp, 4)}C   power {pwr}"
    )
    lines.append(
        f"    cpu  {_num(sample.cpu, 5)}%   RAM {_mib(sample.ram_used)} / {_mib(sample.ram_total)}"
        f"   {_spark(sample.hist_gpu_util)}"
    )

    loaded = [mid for mid, st in (swap.get("models") or []) if st and st != "unloaded"]
    if loaded:
        lines.append(f"    swap models     {', '.join(loaded)}")
    if not swap.get("ok"):
        lines.append(f"    swap unreachable: {swap.get('error')}")

    lines.append(bar)
    return "\n".join(lines)


def loop(engine: str, swap_base: str, interval: float, once: bool, as_json: bool) -> int:
    calib = Calibration.load()
    while True:
        sample = fetch_engine(engine)
        swap = fetch_swap(swap_base)
        if as_json:
            payload = {
                "sample": sample.__dict__,
                "speculation": speculation(sample).__dict__,
                "economics": context_economics(sample).__dict__,
                "vram": headroom(sample, calib),
                "cost": context_cost(sample, calib),
            }
            print(json.dumps(payload, indent=2, default=str))
        else:
            print("\033[2J\033[H" + render(sample, swap, calib), flush=True)
        if once:
            return 0 if sample.ok else 1
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=DEFAULT_ENGINE, help="inference engine base URL")
    ap.add_argument("--swap", default=DEFAULT_SWAP, help="llama-swap base URL (optional context)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    return loop(args.engine, args.swap, args.interval, args.once, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
