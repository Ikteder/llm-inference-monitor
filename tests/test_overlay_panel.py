# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Overlay panel tests. No display, no network, deterministic."""

from __future__ import annotations

import pytest

from monitor import panel
from monitor.collect import Sample
from monitor.overlay import CORNERS


def sample(**over) -> Sample:
    base = dict(
        ok=True,
        fetched_at=0.0,
        model="qwen3.8-flash-next-iq3_s",
        engine_version="0.1.59",
        max_context=65536,
        drafting="dflash2",
        spec_depth=8,
        vram_free_mib=1146,
        state="generating",
        phase="thinking",
        live_generated=212,
        live_tok_s=21.4,
        live_elapsed_s=9.7,
        queued=0,
        requests=52,
        prompt_tokens=1_355_537,
        reused_tokens=1_125_756,
        output_tokens=21_876,
        prompt_ms=318_943.8,
        decode_ms=1_061_230.2,
        drafts_offered=20_676,
        drafts_accepted=17_611,
        gpu_name="RTX 4070 SUPER",
        gpu_util=98.0,
        gpu_mem_used=int(0.88 * 12878610432),
        gpu_mem_total=12878610432,
        gpu_temp=83.0,
        gpu_power=205.0,
        gpu_power_limit=220.0,
        cpu=47.0,
        ram_used=int(29.5 * 1024**3),
        ram_total=int(31.9 * 1024**3),
        hist_tok_s=[18 + (i % 5) for i in range(40)],
        hist_gpu_util=[90 + (i % 8) for i in range(40)],
        hist_gpu_mem_used=[int(0.80 * 12878610432) + i * 900_000 for i in range(40)],
        hist_cpu=[40 + (i % 10) for i in range(40)],
        requests_detail=[],
    )
    base.update(over)
    return Sample(**base)


ARGS = {
    "swap": {"gpu": {}},
    "vram": {"spare_context_tokens": 19_000},
    "spec": {"acceptance_rate": 0.85},
    "econ": {"cache_hit_rate": 0.82, "io_ratio": 64.0, "mean_prompt_per_request": 39_000},
    "proj": {"available": True, "projectable": True, "slope_mib_per_min": 700.0},
}


def render(s: Sample, **over) -> panel.Recorder:
    kw = {**ARGS, **over}
    rec = panel.Recorder()
    panel.draw(rec, s, kw["swap"], kw["vram"], kw["spec"], kw["econ"], kw["proj"])
    return rec


def texts(rec: panel.Recorder) -> list[str]:
    return [p.text for p in rec.prims if p.kind == "text"]


def colored(rec: panel.Recorder, color: str) -> list[panel.Prim]:
    return [p for p in rec.prims if p.color == color]


# --- content ---------------------------------------------------------------

def test_headline_shows_speed_and_phase():
    t = texts(render(sample()))
    assert any("21.4 tok/s" in x for x in t)
    assert any("thinking" in x for x in t)


def test_idle_state_uses_history_not_live_rate():
    s = sample(state="idle", live_tok_s=None, phase=None)
    t = texts(render(s))
    assert any(f"{s.hist_tok_s[-1]:.1f} tok/s" in x for x in t), t


def test_offline_engine_renders_an_error_not_a_crash():
    rec = render(sample(ok=False, error="connection refused"))
    assert any("engine offline" in x for x in texts(rec))
    assert any("connection refused" in x for x in texts(rec))
    assert colored(rec, panel.BAD), "offline must be shown in the alarm color"


def test_vram_bar_and_free_mib_are_shown():
    t = texts(render(sample()))
    assert any("VRAM" in x for x in t)
    assert any("10808/12282 MiB" in x for x in t)
    assert any("engine free 1146 MiB" in x for x in t)


def test_context_row_shares_used_and_ceiling():
    t = texts(render(sample()))
    assert any("39,000/65,536" in x for x in t)
    assert any("+19,000 spare" in x for x in t)


def test_uncalibrated_says_so_instead_of_guessing():
    t = texts(render(sample(), vram={"spare_context_tokens": None}))
    assert any("calibrate for spare" in x for x in t)
    assert not any("spare tokens ~" in x for x in t)


def test_economics_rows_present():
    t = texts(render(sample()))
    assert any("KV RESIDENT" in x for x in t)
    assert any("82%" in x for x in t)
    assert any("I/O 64" in x for x in t)
    assert any("DRAFTS" in x for x in t)


def test_thermal_strip_flags_hot_gpu_and_near_power_limit():
    t = texts(render(sample()))
    strip = [x for x in t if "83C" in x]
    assert strip, t
    assert "!" in strip[0], "83C must be flagged"
    assert "205/220W!" in strip[0], "93% of the power limit must be flagged"
    assert colored(render(sample()), panel.BAD), "hot state must paint the strip red"


def test_cool_gpu_strip_is_not_flagged():
    s = sample(gpu_temp=61.0, gpu_power=120.0, gpu_power_limit=220.0)
    rec = render(s)
    strip = [x for x in texts(rec) if "61C" in x]
    assert strip and "!" not in strip[0]
    assert not colored(rec, panel.BAD)


def test_swap_fallback_used_when_engine_lacks_gpu_block():
    s = sample(gpu_util=None, gpu_temp=None, gpu_power=None, gpu_power_limit=None)
    swap = {"gpu": {"util": 55.0, "temp_celsius": 62.0, "power_w": 90.0, "power_limit_w": 200.0}}
    t = texts(render(s, swap=swap))
    assert any("GPU 55%" in x and "62C" in x and "90/200W" in x for x in t), t


def test_model_name_is_always_in_the_footer():
    assert any("qwen3.8-flash-next-iq3_s" in x for x in texts(render(sample())))


def test_trend_only_when_projection_is_projectable():
    assert not any("MiB/min" in x for x in texts(render(sample(), proj={"available": True, "projectable": False})))
    assert any("+700 MiB/min" in x for x in texts(render(sample())))


# --- bars and colors -------------------------------------------------------

def test_bars_draw_track_and_fill_with_correct_width():
    rec = render(sample())
    rects = [p for p in rec.prims if p.kind == "rect"]
    tracks = [p for p in rects if p.color == panel.TRACK]
    fills = [p for p in rects if p.color != panel.TRACK]
    assert len(tracks) == 2, "VRAM and CONTEXT each draw a track"
    widths = {p.x1 - p.x for p in tracks}
    assert widths == {panel.PANEL_W - 2 * panel.PAD}
    vram_fill = next(p for p in fills if abs((p.x1 - p.x) / (panel.PANEL_W - 2 * panel.PAD) - 0.88) < 0.01)
    assert vram_fill.color == panel.WARN, "88% VRAM must be amber"


def test_color_ramp_thresholds():
    assert panel.color_for(0.5) == panel.OK
    assert panel.color_for(0.79) == panel.WARN
    assert panel.color_for(0.93) == panel.BAD
    assert panel.color_for(None) == panel.DIM


def test_missing_values_never_render_nan_or_crash():
    s = sample(gpu_mem_used=None, gpu_mem_total=0, vram_free_mib=None, max_context=0,
               gpu_util=None, gpu_temp=None, gpu_power=None, gpu_power_limit=None, cpu=None,
               hist_tok_s=[], live_tok_s=None, state="idle", phase=None, model=None)
    rec = render(
        s,
        vram={"spare_context_tokens": None},
        spec={"acceptance_rate": None},
        econ={"cache_hit_rate": None, "io_ratio": None, "mean_prompt_per_request": 0},
        proj={"available": False, "projectable": False, "error": "no history"},
    )
    joined = " ".join(texts(rec))
    assert "nan" not in joined.lower() and "none" not in joined.lower(), joined


def test_every_frame_is_fully_redrawn():
    """The canvas is cleared each tick; draw must not rely on leftovers from the last frame."""
    first = render(sample())
    second = render(sample(state="idle", live_tok_s=None))
    assert len(first.prims) > 0 and len(second.prims) > 0
    assert [p.text for p in first.prims if p.kind == "text"] != [p.text for p in second.prims if p.kind == "text"]


def test_nothing_draws_outside_the_panel():
    rec = render(sample())
    for p in rec.prims:
        assert 0 <= p.x <= panel.PANEL_W, p
        assert 0 <= p.y <= panel.PANEL_H, p
        if p.kind in ("rect", "oval"):
            assert p.x1 <= panel.PANEL_W and p.y1 <= panel.PANEL_H, p


def test_recorder_clear_semantics():
    rec = render(sample())
    assert rec.prims
    rec.delete("all")
    assert rec.prims == []


# --- window configuration (no display needed) ------------------------------

@pytest.mark.parametrize("corner", sorted(CORNERS))
def test_corner_anchors_are_valid(corner):
    ax, ay = CORNERS[corner]
    assert ax in (0, 1) and ay in (0, 1)


def test_panel_has_a_fixed_footprint():
    """A corner widget that changes size between frames would jitter; the panel must not."""
    assert (panel.PANEL_W, panel.PANEL_H) == (300, 176)
