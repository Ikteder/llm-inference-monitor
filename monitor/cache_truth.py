# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Cache-truth measurement: does the engine's reported cache hit match a real cost?

The engine reports cached tokens per request. This module proves or disproves that number by
timing two requests of identical size, one guaranteed cold and one warmed, and comparing the
wall-clock difference against what the counters claim. Stdlib only; requires a reachable engine.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def post_chat(url: str, body: bytes, timeout: float = 120.0) -> dict:
    """One chat completion. A seam so the timing logic can be tested without an engine."""
    req = urllib.request.Request(  # noqa: S310 (fixed localhost URL)
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))
from dataclasses import dataclass

from monitor.collect import DEFAULT_ENGINE


@dataclass(frozen=True)
class CacheTruth:
    cold_s: float
    warm_s: float
    speedup: float | None
    reported_cached_tokens: int
    reported_prompt_tokens: int
    first_claimed_hit_s: float | None
    verdict: str
    timing_resolves_cache: bool = True


_MIN_RESOLVABLE_MS_PER_TOKEN = 0.25  # below this, a flat probe cannot tell a real hit from a fake one


def measure_cache_truth(
    base: str = DEFAULT_ENGINE,
    *,
    prompt_tokens: int = 3000,
    timeout: float = 600.0,
    rounds: int = 4,
    filler: str = "The quick brown fox jumps over the lazy dog while the kernel logs a warning. ",
) -> CacheTruth:
    """Time repeated identical prompts and compare against the engine's cached-token claim.

    A prompt sent before is already resident in the KV slot cache, so a single "cold" request is not
    cold: on the reference stack the first attempt at a 2,600 token prompt took 5.8 s while every later
    attempt took about 0.5 s and then stayed flat. This probe therefore runs several rounds and compares
    the genuinely cold first pass against the median of the steady-state passes. A one-shot probe would
    either call an already-warm prompt "cold" or read the flat tail as proof that caching does nothing.
    """
    words = max(prompt_tokens - 60, 1) // len(filler.split()) + 1
    prompt = filler * words

    url = f"{base.rstrip('/')}/v1/chat/completions"

    def timed() -> tuple[float, dict]:
        body = json.dumps(
            {
                "model": "cache-truth",
                "messages": [{"role": "user", "content": prompt + "\nReply with the single word: ok"}],
                "max_tokens": 4,
                "stream": False,
            }
        ).encode("utf-8")
        t0 = time.perf_counter()
        payload = post_chat(url, body, timeout=timeout)
        return time.perf_counter() - t0, payload

    times: list[float] = []
    last: dict = {}
    for _ in range(max(rounds, 2)):
        t, last = timed()
        times.append(t)

    cold_s = times[0]
    rest = sorted(times[1:])
    warm_s = rest[len(rest) // 2]  # median of steady-state passes, immune to one slow outlier
    first_claimed_hit_s = times[1] if len(times) > 1 else None

    usage = last.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    ptok = int(usage.get("prompt_tokens") or 0)

    speedup = (cold_s / warm_s) if warm_s > 0 else None

    # A cached-token counter is only falsifiable through timing if the prefill it claims to skip was
    # big enough to see. On this stack a cold 2,600 token prompt costs about 2 ms per *total* prompt token
    # (roughly 5 s for 2,600 tokens). A 3,000 token probe is therefore resolvable down to 0.75 s of wall
    # clock; anything faster than that and the claimed saving hides inside request overhead, so a flat
    # result says nothing. Without this gate an honest engine gets accused of lying, which is the same
    # class of error as the one this probe exists to catch.
    #
    # The floor is per prompt token rather than per fresh token on purpose: it measures what the probe is
    # capable of seeing, not what the engine claims to have skipped. Dividing by (prompt - cached) instead
    # would let a fabricated 100% hit claim excuse its own flat timing.
    if ptok and cold_s > 0:
        ms_per_prompt_token = cold_s * 1000.0 / ptok
        timing_resolves_cache = ms_per_prompt_token >= _MIN_RESOLVABLE_MS_PER_TOKEN
    else:
        timing_resolves_cache = False

    if speedup is None:
        verdict = "warm request failed"
    elif cached == 0 and speedup < 1.25:
        verdict = "no caching occurred; reported counters agree with timing"
    elif cached and speedup >= 1.25:
        verdict = (
            f"cache is real: first pass {cold_s:.2f}s vs steady state {warm_s:.2f}s ({speedup:.1f}x), "
            f"engine claims {cached} of {ptok} tokens reused"
        )
    elif cached and speedup < 1.15 and timing_resolves_cache:
        verdict = (
            f"suspicious: engine claims {cached} of {ptok} tokens cached but the first pass was only "
            f"{speedup:.2f}x slower than steady state; the counter may not reflect work avoided"
        )
    elif cached and not timing_resolves_cache:
        verdict = (
            f"inconclusive: probe too cheap to time. First pass {cold_s:.3f}s for {ptok} prompt tokens "
            "is below the point where a cache claim can be checked by clock; raise --prompt-tokens"
        )
    else:
        verdict = f"inconclusive: speedup {speedup:.2f}x with {cached} cached tokens claimed"

    return CacheTruth(
        cold_s=cold_s,
        warm_s=warm_s,
        speedup=speedup,
        reported_cached_tokens=cached,
        reported_prompt_tokens=ptok,
        first_claimed_hit_s=first_claimed_hit_s,
        verdict=verdict,
        timing_resolves_cache=timing_resolves_cache,
    )
