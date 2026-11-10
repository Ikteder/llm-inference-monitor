# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""VRAM headroom model, pinned to measurements instead of guessed constants.

The engine reports free VRAM and its own context ceiling, but nothing on the box answers the question
that actually matters before a swap or a long session: how many more tokens of context fit in the memory
that is genuinely available? Answering it needs two numbers that are exposed nowhere:

1. bytes per resident token for this exact build, since KV quantization, expert cache and arena
   allocations all vary between builds and are measured from observed VRAM growth;
2. the reserved floor below which the driver refuses new loads, observed as the free-VRAM value when a
   model is idle-loaded rather than mid-generation.

Both come from ``python -m monitor.calibrate`` runs against a live engine and are stored in the record
defined by monitor.calibration. With no calibration on disk the model still runs, but reports
``calibrated=False`` and refuses to print a spare-token figure: an uncalibrated guess at headroom is
worse than admitting the number is unknown.
"""

from __future__ import annotations

import math

from monitor.calibration import Calibration
from monitor.collect import Sample


def usable_vram_bytes(s: Sample, calib: Calibration) -> int | None:
    """VRAM that new work can actually claim right now.

    Prefers the engine's own ``vram_free_mib``, which already accounts for fragmentation inside the
    process, and falls back to driver-reported totals when that field is absent. The calibrated reserved
    floor is subtracted because loads below it fail outright rather than degrading.
    """
    if s.vram_free_mib is not None:
        free = int(s.vram_free_mib) * 1024 * 1024
    elif s.gpu_mem_used is not None and s.gpu_mem_total:
        free = int(s.gpu_mem_total - s.gpu_mem_used)
    else:
        return None
    if calib.reserved_floor_bytes:
        free -= int(calib.reserved_floor_bytes)
    return max(free, 0)


def headroom(s: Sample, calib: Calibration) -> dict[str, object]:
    """Spare context capacity for the current build.

    Returns a plain dict so the TUI, overlay and panel can render it without another dataclass layer.
    ``spare_context_tokens`` is deliberately None rather than a guess when calibration is missing: a
    plausible-looking spare-token number from an unmeasured coefficient is how a user ends up choosing a
    context size that fails to load.
    """
    usable = usable_vram_bytes(s, calib)
    bpt = calib.bytes_per_token
    calibrated = bool(bpt and bpt > 0)
    spare: int | None = int(usable // bpt) if (usable is not None and calibrated) else None

    ceiling = s.max_context
    verdict = "uncalibrated: run `python -m monitor.calibrate` to get a spare-token figure"
    if spare is not None and ceiling:
        # Thresholds are relative to the configured ceiling, not absolute token counts: 4000 spare tokens
        # is dire at an 8K context and generous at 128K, and only the ratio carries across builds.
        if spare < 0.15 * ceiling:
            verdict = "tight: context will be the constraint before VRAM is"
        elif spare < ceiling:
            verdict = "ok for the configured context, no room to double it"
        else:
            verdict = "room to raise context or load a second model"

    return {
        "usable_bytes": usable,
        "bytes_per_token": bpt if calibrated else None,
        "calibrated": calibrated,
        "spare_context_tokens": spare,
        "context_ceiling": ceiling,
        "verdict": verdict,
        "source_model": calib.engine_version or calib.gpu_name,
        "note": "; ".join(calib.notes),
    }


def context_cost(s: Sample, calib: Calibration) -> dict[str, object]:
    """What a context turn costs, using per-request records rather than cumulative sums.

    The engine keeps the most recent request records with their own prompt sizes, reuse and
    timings. Slopes must come from those per-request pairs: dividing cumulative prefill ms by
    cumulative fresh tokens mixes requests of wildly different sizes and reports a number that
    belongs to no real turn. Each pair here is one actual request, so ``ms_per_fresh_token``
    is the measured re-prefill cost for the kind of turn actually being run.
    """
    recs = [r for r in s.requests_detail if isinstance(r, dict) and r.get("prompt_ms")]
    if not recs:
        return {"available": False, "reason": "no per-request prompt timings yet"}

    slopes: list[float] = []
    prompts: list[float] = []
    for r in recs:
        ptok = float(r.get("prompt_tokens") or 0)
        reused = min(float(r.get("reused") or 0), ptok)
        fresh = ptok - reused
        prompts.append(ptok)
        if fresh >= 256:  # below this the request is essentially all cache hits; slope is noise
            slopes.append(float(r["prompt_ms"]) / fresh)

    mean_prompt = sum(prompts) / len(prompts) if prompts else 0.0
    ms_per_fresh_token = sum(slopes) / len(slopes) if slopes else None

    return {
        "available": True,
        "requests_sampled": len(recs),
        "mean_prompt_tokens": mean_prompt,
        "ms_per_fresh_token": ms_per_fresh_token,
        "prefill_tok_s_from_requests": (
            (1000.0 / ms_per_fresh_token) if ms_per_fresh_token else None
        ),
        "context_bytes_for_mean_prompt": (
            int(mean_prompt * calib.bytes_per_token)
            if calib.bytes_per_token
            else None
        ),
        "spare_context_tokens": headroom(s, calib)["spare_context_tokens"],
    }


def growth_projection(history: list[float], window_s: float, horizon_min: float = 30.0) -> dict[str, object]:
    """Linear projection of a monotonic resource (VRAM used) over a horizon.

    Least-squares on the engine's own rolling history array, which is one sample per second
    and covers roughly ``window_s`` seconds. Returns slope in MiB/min plus an R2 gate: below
    0.5 the series is noise and projecting it would be theater, so ``projectable`` stays False
    regardless of how large the fitted slope looks.
    """
    MIB = 1024 * 1024
    vals = [v for v in history if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(vals)
    if n < 8:
        return {"available": False, "reason": f"only {n} samples in history", "projectable": False}
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(vals) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if not sxx:
        return {"available": False, "reason": "degenerate x range", "projectable": False}
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals))
    slope_bytes_per_sample = sxy / sxx
    syy = sum((y - mean_y) ** 2 for y in vals)
    r2 = (sxy * sxy) / (sxx * syy) if syy else 0.0

    samples = max(n, 2)
    seconds_per_sample = max(window_s / (samples - 1), 1e-9)
    slope_mib_per_min = (slope_bytes_per_sample / MIB) * (60.0 / seconds_per_sample)
    horizon_samples = (horizon_min * 60.0) / seconds_per_sample

    return {
        "available": True,
        "slope_bytes_per_sample": slope_bytes_per_sample,
        "slope_mib_per_min": slope_mib_per_min,
        "seconds_per_sample": seconds_per_sample,
        "r2": r2,
        "projected_delta_mib_over_horizon": (slope_bytes_per_sample * horizon_samples) / MIB,
        "horizon_min": horizon_min,
        "projectable": r2 >= 0.5 and abs(slope_bytes_per_sample) > 0,
    }
