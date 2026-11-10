# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cache-truth probe.

The offline test drives the probe with a fake transport and pinned timings so the cold-versus-warm
logic is checked without hardware. The live test skips when no engine answers, keeping CI green while
still exercising the real thing on a dev box.
"""

from __future__ import annotations

import time

import pytest

from monitor.cache_truth import measure_cache_truth
from monitor.collect import DEFAULT_ENGINE, fetch_engine


def test_first_pass_is_clearly_slower_than_steady_state(monkeypatch):
    from monitor import cache_truth as ct

    calls = {"n": 0}

    def fake_post(url, body, timeout=120.0):
        calls["n"] += 1
        # First pass pays for prefill; later passes are served from the slot cache and flatten out.
        time.sleep(0.06 if calls["n"] == 1 else 0.005)
        return {
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2990},
            }
        }

    monkeypatch.setattr(ct, "post_chat", fake_post)
    t = ct.measure_cache_truth(prompt_tokens=3000, rounds=4)

    assert calls["n"] == 4
    assert t.cold_s > t.warm_s * 2.0, f"first pass should dwarf steady state: {t}"
    assert t.reported_cached_tokens == 2990 and t.reported_prompt_tokens == 3000
    # The second pass is the first one that could claim a hit; it is reported separately so a reader
    # can see the transition rather than a single averaged number.
    assert t.first_claimed_hit_s is not None and t.first_claimed_hit_s < t.cold_s
    assert "cache is real" in t.verdict


def test_a_slow_flat_probe_with_a_big_cached_claim_is_flagged(monkeypatch):
    """Flat timing plus a cached claim that should have been visible equals suspicious.

    The sleep is long enough that 3,000 tokens of prefill would clearly show up if it were really being
    redone on every pass. Flatness at that scale means the counter is not describing work avoided.
    """
    from monitor import cache_truth as ct

    def fake_post(url, body, timeout=120.0):
        time.sleep(1.0)
        return {
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2990},
            }
        }

    monkeypatch.setattr(ct, "post_chat", fake_post)
    t = ct.measure_cache_truth(prompt_tokens=3000, rounds=4)
    assert t.speedup < 1.5, f"probe should be near flat, measured {t.speedup:.2f}x"
    assert t.timing_resolves_cache, "a 1 s pass over 3,000 prompt tokens is ~1.7 ms/token, well inside range"
    assert "suspicious" in t.verdict, t.verdict


def test_a_probe_too_cheap_to_time_is_inconclusive_not_suspicious(monkeypatch):
    """A 10 ms pass is HTTP overhead, not prefill.

    At that scale a real engine with a warm cache really does return nearly flat times, so calling it
    suspicious would accuse an honest engine. The probe says what it cannot measure instead.
    """
    from monitor import cache_truth as ct

    def fake_post(url, body, timeout=120.0):
        time.sleep(0.01)
        return {
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2990},
            }
        }

    monkeypatch.setattr(ct, "post_chat", fake_post)
    t = ct.measure_cache_truth(prompt_tokens=3000, rounds=4)
    assert t.speedup < 1.5, f"probe should be near flat, measured {t.speedup:.2f}x"
    assert not t.timing_resolves_cache
    assert "inconclusive" in t.verdict and "--prompt-tokens" in t.verdict, t.verdict


def test_no_caching_and_matching_counters_agree(monkeypatch):
    from monitor import cache_truth as ct

    def fake_post(url, body, timeout=120.0):
        time.sleep(0.02)
        return {
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }

    monkeypatch.setattr(ct, "post_chat", fake_post)
    t = ct.measure_cache_truth(prompt_tokens=300, rounds=3)
    assert t.reported_cached_tokens == 0
    assert "no caching occurred" in t.verdict


@pytest.mark.skipif(not fetch_engine(DEFAULT_ENGINE, timeout=2.0).ok, reason="no live engine")
def test_reported_cache_hits_survive_a_timing_crosscheck():
    truth = measure_cache_truth(DEFAULT_ENGINE, prompt_tokens=2500, rounds=4)
    assert truth.cold_s > 0 and truth.warm_s > 0
    # A genuinely cold first pass must cost at least about as much as the steady state. It can legitimately
    # be slightly cheaper: a repeat prefill can land on warmer allocator state even with no prefix reuse.
    # A cold pass much faster than steady state is what needs flagging, so the gate sits well below parity.
    assert truth.cold_s >= 0.5 * truth.warm_s, f"cold pass far faster than steady state: {truth}"
    assert truth.reported_prompt_tokens > 1500
    assert truth.verdict
