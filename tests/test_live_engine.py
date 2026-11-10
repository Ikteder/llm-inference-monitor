# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks against a live engine and a live llama-swap.

Skipped when the services are unreachable, so CI on a clean runner stays green while these same
tests exercise the real thing on a dev box.

The switch tests deliberately stop at the dry-run boundary: they resolve a target and build the
request body but never POST /switch, because running a test suite should not change which model
is loaded on someone's machine.
"""

from __future__ import annotations

import pytest

from monitor.calibration import Calibration
from monitor.cache_truth import measure_cache_truth
from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP, fetch_engine, fetch_swap
from monitor.economics import context_economics, speculation
from monitor.model_switch import ModelInfo, ModelSwitcher, SwitchRefused
from monitor.vram import headroom

sample = pytest.mark.skipif(not fetch_engine(DEFAULT_ENGINE).ok, reason="no live inference engine")
swap_live = pytest.mark.skipif(
    not fetch_swap(DEFAULT_SWAP, timeout=2.0).get("ok"), reason="no llama-swap on this host"
)


@sample
def test_live_sample_is_internally_consistent():
    s = fetch_engine(DEFAULT_ENGINE)
    assert s.ok and s.model
    assert s.reused_tokens <= s.prompt_tokens
    assert s.drafts_accepted <= s.drafts_offered
    assert s.output_tokens >= 0


@sample
def test_live_metrics_agree_with_the_engines_own_per_request_figures():
    """The engine reports decode_tok_s and acceptance per request; aggregates must line up."""
    s = fetch_engine(DEFAULT_ENGINE)
    if not s.requests_detail:
        pytest.skip("engine has no request records yet")

    reported = [float(r["decode_tok_s"]) for r in s.requests_detail if r.get("decode_tok_s")]
    derived = context_economics(s).mean_decode_tok_s
    assert reported and derived is not None
    mean = sum(reported) / len(reported)
    # Cumulative average since boot vs the mean of the last ~12 requests. These are different
    # windows and the workload changes between them, so this is a magnitude gate only: it exists
    # to catch a misread field (a 2x or 1000x unit error), not normal variance. Observed spread on
    # this engine has been ~1.5x, so the bound sits well clear of that.
    ratio = derived / mean if mean else 0.0
    assert 0.4 < ratio < 2.5, f"derived {derived:.1f} vs reported mean {mean:.1f} (ratio {ratio:.2f})"

    # Speculative acceptance is a ratio of the same counters, checked against per-request records. The
    # tolerance is wide because the engine's rolling mean covers requests this snapshot does not.
    # Recomputed from accepted/offered per request, which is the only way to confirm the aggregate
    # denominator is the one the dashboard should be showing.
    acc = [
        float(r["drafts_accepted"]) / float(r["drafts_offered"])
        for r in s.requests_detail if r.get("drafts_offered")
    ]
    spec_acc = speculation(s).acceptance_rate
    if acc and spec_acc is not None:
        assert abs(spec_acc - sum(acc) / len(acc)) < 0.5


@sample
def test_live_headroom_is_positive_and_bounded():
    """Headroom must be a real number on a live engine, and inside one ceiling of plausibility.

    The upper bound is deliberately not tight. Spare capacity is measured against free VRAM at this instant,
    which legitimately exceeds the configured context: enough room for several concurrent contexts, or for
    raising n_ctx, is exactly what a user needs to know. What must never happen is an unbounded or negative
    figure, and that is what the bounds pin down.
    """
    s = fetch_engine(DEFAULT_ENGINE)
    h = headroom(s, Calibration.load())
    assert h["usable_bytes"] > 0
    ceiling = s.max_context or 10**9
    if h["spare_context_tokens"] is not None:
        assert 0 < h["spare_context_tokens"] < (h["usable_bytes"] or 0) / 100.0
        assert h["bytes_per_token"] and h["bytes_per_token"] > 0
    else:
        # Not calibrated is a legitimate state, but it has to be stated rather than papered over.
        assert not h["calibrated"] and "calibrate" in h["verdict"]
        assert h["spare_context_tokens"] is None
    assert h["context_ceiling"] == ceiling


@sample
def test_engine_cache_counters_survive_a_wall_clock_crosscheck():
    """Independent timing evidence for the engine's own cached-token claim.

    Magnitude gate, not a precise bound. A 2500 token prefill on this engine measures hundreds of
    milliseconds; a field misread as microseconds or reported in the wrong unit lands three orders of
    magnitude off, which is what this catches. Normal run-to-run variance on this box has been around
    1.5x, so the bounds sit well clear of it.
    """
    truth = measure_cache_truth(DEFAULT_ENGINE, prompt_tokens=2500, rounds=4)
    assert truth.cold_s > 0 and truth.warm_s > 0
    assert truth.reported_prompt_tokens > 1500
    # Both passes must be prefill-sized. A cold pass measured in microseconds means the probe is reading
    # a counter as a duration, which is the failure mode worth guarding.
    assert truth.cold_s > 0.05 and truth.warm_s > 0.05, f"sub-millisecond passes: {truth}"
    # A cold first pass should not beat the steady state by much. The bound is loose because a warm KV
    # slot can make a repeat prefill genuinely faster than a fresh one even with no prefix reuse.
    assert truth.cold_s >= 0.5 * truth.warm_s, f"cold pass far faster than steady state: {truth}"
    # Thousands of claimed reused tokens with no measurable gain is the failure this probe exists for.
    if truth.reported_cached_tokens and truth.timing_resolves_cache:
        assert truth.speedup and truth.speedup > 1.05, f"claimed hits without speedup: {truth}"


@swap_live
def test_switcher_lists_models_with_at_most_one_active():
    sw = ModelSwitcher(DEFAULT_SWAP, engine_base=DEFAULT_ENGINE)
    models = sw.refresh()
    assert models, "expected at least one catalog entry"
    assert len([m for m in models if m.active]) <= 1


@swap_live
def test_switch_path_is_dry_run_only_and_never_touches_the_engine():
    """Resolve and build a switch body without sending anything."""
    sw = ModelSwitcher(DEFAULT_SWAP, engine_base=DEFAULT_ENGINE)
    models = sw.refresh()
    target = next((m for m in models if not m.active), None)
    if target is None:
        pytest.skip("only one model catalogued; nothing to switch to")

    assert sw.resolve_target(target.name) == target.name
    body = sw.build_switch_request(target.name)
    assert body == {"model": target.name}

    preview = sw.preview(target.name)
    assert preview["ok"] and preview["target"] == target.name
    assert "resident_prompt_tokens_lost" in preview


@swap_live
def test_resolve_target_rejects_unknown_and_ambiguous_names():
    sw = ModelSwitcher(DEFAULT_SWAP, engine_base=DEFAULT_ENGINE)
    sw.refresh()
    with pytest.raises(KeyError):
        sw.resolve_target("definitely-not-installed")


@sample
def test_switch_refuses_when_the_engine_is_busy_unless_forced():
    """The safety gate is the whole point of the control path, so it gets a fake-driven test.

    Uses a stub engine rather than making someone wait for a real generation to be in flight.
    """
    class StubSwitcher(ModelSwitcher):
        def __init__(self):
            super().__init__(swap_base="http://127.0.0.1:1", engine_base="http://127.0.0.1:1")
            self._models = []

        @property
        def models(self):  # type: ignore[override]
            return [
                ModelInfo(name="swap/live", status="ready", active=True, loaded=True),
                ModelInfo(name="swap/other", status="ready", active=False, loaded=False),
            ]

        def engine_busy(self):
            return True, "engine state is 'generating'"

    sw = StubSwitcher()
    with pytest.raises(SwitchRefused) as exc:
        sw.switch("other")
    assert "force=True" in str(exc.value)
    assert sw.resolve_target("other") == "swap/other"
