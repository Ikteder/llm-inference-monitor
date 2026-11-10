# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Derived inference economics.

Everything here is computed from fields the engine already exposes. The point is that
no endpoint reports these numbers directly; they are ratios and cost models built on top
of raw counters, which is where the useful signal lives.
"""

from __future__ import annotations

from dataclasses import dataclass

from monitor.collect import Sample


@dataclass(frozen=True)
class Speculation:
    """What speculative decoding (drafting) is actually worth right now."""

    drafting: str
    spec_depth: int
    drafts_offered: int
    drafts_accepted: int
    acceptance_rate: float | None
    mean_bonus_tokens: float | None
    tokens_per_draft: float | None
    accepted_output_share: float | None
    verdict: str


@dataclass(frozen=True)
class ContextEconomics:
    """Where prompt tokens go, and what re-prefilling costs."""

    prompt_tokens: int
    reused_tokens: int
    fresh_prompt_tokens: int
    cache_hit_rate: float | None
    output_tokens: int
    io_ratio: float | None
    requests: int
    mean_prompt_per_request: float | None
    prefill_ms: float
    decode_ms: float
    prefill_share_of_busy_time: float | None
    mean_prefill_tok_s: float | None
    mean_decode_tok_s: float | None
    cost_per_1k_output_tokens: dict[str, float] | None


@dataclass(frozen=True)
class VramBudget:
    """VRAM headroom and what a longer context would cost."""

    used_bytes: int | None
    total_bytes: int | None
    free_bytes: int | None
    used_pct: float | None
    per_token_kv_bytes: float | None
    max_context_tokens: int
    spare_context_tokens: int | None
    headroom_verdict: str


def _ratio(numer: float, denom: float) -> float | None:
    if not denom:
        return None
    return numer / denom


def speculation(s: Sample) -> Speculation:
    """Acceptance rate plus the token-per-draft multiplier.

    ``mean_bonus_tokens`` is accepted drafts per offered draft: each accepted draft lets
    the engine emit that many extra tokens in one forward pass, so it is the direct speed
    multiplier of the drafting setup. ``accepted_output_share`` answers the more honest
    question, "how much of my output came from speculation being right".
    """
    offered = s.drafts_offered
    accepted = s.drafts_accepted
    acc = _ratio(accepted, offered)
    bonus = _ratio(accepted, offered)
    # Tokens emitted per draft round: 1 always-free verified token plus the accepted bonus.
    per_draft = (1.0 + bonus) if bonus is not None else None
    share = _ratio(accepted, s.output_tokens)

    if acc is None:
        verdict = "no draft data yet"
    elif acc >= 0.80:
        verdict = "excellent: drafting is pulling its weight"
    elif acc >= 0.60:
        verdict = "good: worth keeping enabled"
    elif acc >= 0.40:
        verdict = "marginal: try a shallower draft depth"
    else:
        verdict = "poor: drafts mostly rejected, drafting may be costing time"

    return Speculation(
        drafting=s.drafting or "none",
        spec_depth=s.spec_depth,
        drafts_offered=offered,
        drafts_accepted=accepted,
        acceptance_rate=acc,
        mean_bonus_tokens=bonus,
        tokens_per_draft=per_draft,
        accepted_output_share=share,
        verdict=verdict,
    )


def context_economics(s: Sample) -> ContextEconomics:
    """Prompt token accounting and the real per-output-token latency bill.

    ``reused`` counts prompt tokens served from KV cache; the rest must be recomputed.
    The I/O ratio is the agent-harness tax: how many input tokens you pay per token of
    useful output. ``cost_per_1k_output_tokens`` converts that into milliseconds, which
    is the number that actually shows up as wall-clock waiting.

    Latency fields are read as milliseconds because that is what the engine reports, and
    every conversion to seconds divides by 1000 explicitly. A misread unit here once made a
    260 s prefill look like a 260,000 s one, so magnitudes are pinned in the test fixtures.
    """
    prompt = s.prompt_tokens
    reused = min(s.reused_tokens, prompt)
    fresh = max(prompt - reused, 0)
    hit = _ratio(reused, prompt)
    io_ratio = _ratio(prompt, s.output_tokens)

    busy_ms = s.prompt_ms + s.decode_ms
    prefill_share = _ratio(s.prompt_ms, busy_ms)
    prefill_tok_s = _ratio(prompt, s.prompt_ms / 1000.0) if s.prompt_ms else None
    decode_tok_s = _ratio(s.output_tokens, s.decode_ms / 1000.0) if s.decode_ms else None

    cost: dict[str, float] | None = None
    if s.output_tokens and busy_ms:
        per_1k = busy_ms / s.output_tokens * 1000.0
        cost = {
            "total_ms_per_1k_output": per_1k,
            "prefill_ms_per_1k_output": (s.prompt_ms / s.output_tokens * 1000.0),
            "decode_ms_per_1k_output": (s.decode_ms / s.output_tokens * 1000.0),
            "fresh_prompt_tokens_per_1k_output": (fresh / s.output_tokens * 1000.0),
        }

    return ContextEconomics(
        prompt_tokens=prompt,
        reused_tokens=reused,
        fresh_prompt_tokens=fresh,
        cache_hit_rate=hit,
        output_tokens=s.output_tokens,
        io_ratio=io_ratio,
        requests=s.requests,
        mean_prompt_per_request=_ratio(prompt, s.requests),
        prefill_ms=s.prompt_ms,
        decode_ms=s.decode_ms,
        prefill_share_of_busy_time=prefill_share,
        mean_prefill_tok_s=prefill_tok_s,
        mean_decode_tok_s=decode_tok_s,
        cost_per_1k_output_tokens=cost,
    )


def vram_budget(s: Sample) -> VramBudget:
    """Free VRAM translated into spare context tokens.

    ``per_token_kv_bytes`` is derived by differencing two per-request records: the engine
    reports how much VRAM was in use and how many tokens were resident, so a pair of
    requests with different context sizes yields a measured bytes-per-token cost. That is
    better than guessing from the quantisation string, because it includes whatever the
    build actually allocates (KV plus scratch plus expert cache).
    """
    used, total = s.gpu_mem_used, s.gpu_mem_total
    free = (total - used) if (used is not None and total) else None
    used_pct = _ratio(used, total) if used is not None else None

    per_tok = _measure_kv_bytes_per_token(s)
    spare = int(free // per_tok) if (free and per_tok) else None

    if free is None:
        verdict = "no VRAM data"
    elif free < 300 * 1024 * 1024:
        verdict = "tight: near the OOM cliff, a model swap may fail"
    elif free < 1200 * 1024 * 1024:
        verdict = "low headroom: watch for load failures on bigger contexts"
    else:
        verdict = "healthy headroom"

    return VramBudget(
        used_bytes=used,
        total_bytes=total,
        free_bytes=free,
        used_pct=used_pct,
        per_token_kv_bytes=per_tok,
        max_context_tokens=s.max_context,
        spare_context_tokens=spare,
        headroom_verdict=verdict,
    )


def _measure_kv_bytes_per_token(s: Sample) -> float | None:
    """Per-request VRAM annotation, for builds that provide it.

    Current engines do not attach memory figures to request records, so this normally
    returns None and callers use the calibration file instead. It is kept because the field
    names are cheap to probe and a future build may add them; returning None is the honest
    answer when the data does not exist.
    """
    pairs: list[tuple[int, int]] = []
    for rec in s.requests_detail[:12]:
        tokens = rec.get("prompt_tokens")
        vram = rec.get("vram_bytes") or rec.get("gpu_mem_used")
        if isinstance(tokens, (int, float)) and isinstance(vram, (int, float)) and tokens > 0:
            pairs.append((int(tokens), int(vram)))

    if len(pairs) < 2:
        return None

    best: float | None = None
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dt = abs(pairs[i][0] - pairs[j][0])
            if dt < 512:  # too little token spread to divide reliably
                continue
            db = abs(pairs[i][1] - pairs[j][1])
            cand = db / dt
            if cand <= 0:
                continue
            if best is None or cand < best:
                best = cand
    return best
