# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the model-switch control path.

No network, no llama-swap required. Covers catalog parsing across both status shapes, target
resolution including ambiguity refusal, and the busy-engine safety gate. No test performs a real
switch.
"""

from __future__ import annotations

import pytest

from monitor.model_switch import ModelInfo, ModelSwitcher, SwitchRefused, parse_models

NEW_SHAPE = {
    "data": [
        {"id": "swap/big-model", "active": True, "status": {"value": "ready"}},
        {"id": "swap/fast-model", "active": False, "status": {"value": "ready"}},
        {"id": "swap/vision-model", "active": False, "status": {"value": "unloaded"}},
    ]
}

OLD_SHAPE = {
    "data": [
        {"id": "swap/big-model", "status": "ready"},
        {"id": "swap/fast-model", "status": "unloaded"},
    ]
}


def test_new_status_shape_marks_the_active_model():
    rows = parse_models(NEW_SHAPE, loaded_names={"swap/big-model"})
    assert [r.name for r in rows] == ["swap/big-model", "swap/fast-model", "swap/vision-model"]
    assert rows[0].active and rows[0].status == "ready"
    assert not rows[1].active
    assert rows[2].status == "unloaded" and not rows[2].loaded


def test_old_string_status_shape_is_accepted():
    rows = parse_models(OLD_SHAPE, loaded_names=set())
    assert rows[0].status == "ready" and rows[1].status == "unloaded"
    # parse_models never invents liveness: an old build gives no active flag at all, and
    # resolve_active() is what decides which entry is live.
    assert not any(r.active for r in rows)


def test_resolve_active_matches_engine_names_across_aliasing():
    from monitor.model_switch import resolve_active

    catalog = ["swap/qwen3.8-flash-next-iq3_s", "swap/qwen3.6-35b-a3b"]
    assert resolve_active(catalog, "swap/qwen3.6-35b-a3b", set()) == {"swap/qwen3.6-35b-a3b"}
    # The engine reports a GGUF file name; shared distinctive tokens pick the right alias.
    assert resolve_active(catalog, "Qwen3.8-Flash-next-DFlash2-GGUF", set()) == {
        "swap/qwen3.8-flash-next-iq3_s"
    }
    # Nothing matches: fall back to llama-swap's own flag rather than guess.
    assert resolve_active(catalog, "totally-unrelated", {"swap/qwen3.6-35b-a3b"}) == {
        "swap/qwen3.6-35b-a3b"
    }
    # An ambiguous token overlap also falls back instead of picking arbitrarily.
    both = ["alpha-shared-one", "alpha-shared-two"]
    assert resolve_active(both, "alpha-shared-three", {"alpha-shared-one"}) == {"alpha-shared-one"}


def test_loaded_names_override_missing_active_flag():
    rows = parse_models(OLD_SHAPE, loaded_names={"swap/fast-model"})
    fast = next(r for r in rows if r.name == "swap/fast-model")
    assert fast.active and fast.loaded


def test_resolution_accepts_unique_substrings_and_refuses_ambiguity():
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, set())

    assert sw.resolve_target("swap/fast-model") == "swap/fast-model"
    assert sw.resolve_target("fast") == "swap/fast-model"
    assert sw.resolve_target("FAST") == "swap/fast-model"  # case-insensitive


def test_ambiguous_substring_raises_with_the_candidates_listed():
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, set())
    with pytest.raises(KeyError) as exc:
        sw.resolve_target("model")
    assert "ambiguous" in str(exc.value)
    assert "swap/big-model" in str(exc.value)


def test_unknown_name_raises_with_the_full_catalog():
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, set())
    with pytest.raises(KeyError) as exc:
        sw.resolve_target("nope")
    assert "no model matches" in str(exc.value)


def test_build_switch_request_is_pure_and_names_the_resolved_id():
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, set())
    assert sw.build_switch_request("fast") == {"model": "swap/fast-model"}


def test_noop_switch_when_target_is_already_active():
    """Switching to the live model short-circuits instead of reloading it."""
    class Idle(ModelSwitcher):
        def engine_busy(self):
            return False, "idle"

    sw = Idle(swap_base="http://127.0.0.1:1", engine_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, {"swap/big-model"})
    out = sw.switch("big")
    assert out["noop"] is True and out["model"] == "swap/big-model"


def test_switch_refuses_while_generating_and_names_the_escape_hatch():
    class Busy(ModelSwitcher):
        def engine_busy(self):
            return True, "engine state is 'generating'"

    sw = Busy(swap_base="http://127.0.0.1:1", engine_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, {"swap/big-model"})
    with pytest.raises(SwitchRefused) as exc:
        sw.switch("fast")
    assert "force=True" in str(exc.value)


def test_missing_engine_url_counts_as_busy_not_idle():
    """Failing to prove the engine is idle must not be read as permission to swap."""
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1")  # no engine_base
    busy, why = sw.engine_busy()
    assert busy and "cannot prove" in why


def test_preview_reports_cache_loss_without_sending_anything():
    sw = ModelSwitcher(swap_base="http://127.0.0.1:1", engine_base="http://127.0.0.1:1")
    sw._models = parse_models(NEW_SHAPE, {"swap/big-model"})
    out = sw.preview("fast")
    assert out["ok"] and out["would_switch"] is True
    assert out["engine_busy"] is True  # unreachable engine counts as busy
    assert "KV cache" in out["note"]
