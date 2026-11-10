# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Local LLM inference monitor.

Zero third-party dependencies at runtime. Talks to two HTTP endpoints that already
exist on this machine and computes the numbers neither one reports on its own.

Stdlib only, Python 3.10+.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from monitor.prom import (
    as_float,
    as_int,
    history as clean_history,
    label_value,
    parse_prometheus,
    pick,
    summarize_cores,
)

__all__ = [
    "DEFAULT_ENGINE",
    "DEFAULT_SWAP",
    "Sample",
    "fetch_engine",
    "fetch_swap",
    "http_json",
    "http_text",
    "parse_engine",
    "parse_prometheus",
    "swap_gpu_metrics",
    "swap_memory_metrics",
    "swap_models",
]


DEFAULT_ENGINE = "http://127.0.0.1:19086"
DEFAULT_SWAP = "http://127.0.0.1:18080"

_TIMEOUT = 3.0


@dataclass
class Sample:
    """One poll of the inference engine, already normalized."""

    ok: bool
    error: str | None = None
    fetched_at: float = 0.0

    # engine identity / config
    model: str = ""
    engine_version: str = ""
    max_context: int = 0
    drafting: str = ""
    spec_depth: int = 0
    kv_quant: str = ""
    prefill_cap: int = 0
    vram_free_mib: int | None = None
    arena_mib: int | None = None
    expert_cache_mib: int | None = None

    # live generation state
    state: str = "idle"
    phase: str | None = None
    live_generated: int | None = None
    live_tok_s: float | None = None
    live_elapsed_s: float | None = None
    queued: int = 0

    # cumulative totals since engine boot
    requests: int = 0
    prompt_tokens: int = 0
    reused_tokens: int = 0
    output_tokens: int = 0
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    drafts_offered: int = 0
    drafts_accepted: int = 0

    # hardware
    gpu_name: str = ""
    gpu_util: float | None = None
    gpu_mem_used: int | None = None
    gpu_mem_total: int | None = None
    gpu_temp: float | None = None
    gpu_power: float | None = None
    gpu_power_limit: float | None = None
    cpu: float | None = None
    ram_used: int | None = None
    ram_total: int | None = None

    # rolling history arrays as reported by the engine (up to 60 samples)
    hist_tok_s: list[float] = field(default_factory=list)
    hist_gpu_util: list[float] = field(default_factory=list)
    hist_gpu_mem_used: list[int] = field(default_factory=list)
    hist_cpu: list[float] = field(default_factory=list)

    # per-request records kept by the engine (up to ~21, newest first)
    requests_detail: list[dict[str, Any]] = field(default_factory=list)


def _num(value: Any, default: float | None = None) -> float | None:
    """Local wrapper so the Sample fields keep their explicit defaults."""
    got = as_float(value)
    return default if got is None else got


def _int(value: Any, default: int = 0) -> int:
    return as_int(value, default)


def http_json(url: str, timeout: float = _TIMEOUT) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed localhost URLs)
        return json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout: float = _TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed localhost URLs)
        return resp.read().decode("utf-8", errors="replace")


def _clean_hist(values: Any, cast_int: bool = False) -> list[Any]:
    return clean_history(values, as_integer=cast_int)


def parse_engine(payload: dict[str, Any], fetched_at: float) -> Sample:
    """Turn the engine's /metrics JSON into a normalized Sample.

    Missing keys never raise: an older or differently built engine still yields a
    usable sample with whatever fields do exist.
    """
    eng = payload.get("engine") or {}
    live = payload.get("live") or {}
    totals = payload.get("totals") or {}
    hw = payload.get("hardware") or {}
    hist = payload.get("history") or {}

    return Sample(
        ok=True,
        fetched_at=fetched_at,
        model=str(eng.get("model") or ""),
        engine_version=str(eng.get("version") or eng.get("engine") or ""),
        max_context=_int(eng.get("max_context") or eng.get("context")),
        drafting=str(eng.get("drafting") or ""),
        spec_depth=_int(eng.get("spec")),
        kv_quant=str(eng.get("kv") or ""),
        prefill_cap=_int(eng.get("prefill_cap")),
        vram_free_mib=(None if eng.get("vram_free_mib") is None else _int(eng.get("vram_free_mib"))),
        arena_mib=(None if eng.get("arena_mib") is None else _int(eng.get("arena_mib"))),
        expert_cache_mib=(None if eng.get("expert_cache_mib") is None else _int(eng.get("expert_cache_mib"))),
        state=str(live.get("state") or "idle"),
        phase=live.get("phase"),
        live_generated=(None if live.get("generated") is None else _int(live.get("generated"))),
        live_tok_s=_num(live.get("tok_s")),
        live_elapsed_s=_num(live.get("elapsed_s")),
        queued=_int(live.get("queued")),
        requests=_int(totals.get("requests")),
        prompt_tokens=_int(totals.get("prompt_tokens")),
        reused_tokens=_int(totals.get("reused")),
        output_tokens=_int(totals.get("output_tokens")),
        prompt_ms=_num(totals.get("prompt_ms"), 0.0) or 0.0,
        decode_ms=_num(totals.get("decode_ms"), 0.0) or 0.0,
        drafts_offered=_int(totals.get("drafts_offered")),
        drafts_accepted=_int(totals.get("drafts_accepted")),
        gpu_name=str(hw.get("gpu_name") or ""),
        gpu_util=_num(hw.get("gpu_util")),
        gpu_mem_used=(None if hw.get("gpu_mem_used") is None else _int(hw.get("gpu_mem_used"))),
        gpu_mem_total=(None if hw.get("gpu_mem_total") is None else _int(hw.get("gpu_mem_total"))),
        gpu_temp=_num(hw.get("gpu_temp")),
        gpu_power=_num(hw.get("gpu_power")),
        gpu_power_limit=_num(hw.get("gpu_power_limit")),
        cpu=_num(hw.get("cpu")),
        ram_used=(None if hw.get("ram_used") is None else _int(hw.get("ram_used"))),
        ram_total=(None if hw.get("ram_total") is None else _int(hw.get("ram_total"))),
        hist_tok_s=_clean_hist(hist.get("tok_s")),
        hist_gpu_util=_clean_hist(hist.get("gpu_util")),
        hist_gpu_mem_used=_clean_hist(hist.get("gpu_mem_used"), cast_int=True),
        hist_cpu=_clean_hist(hist.get("cpu")),
        requests_detail=[r for r in (payload.get("requests") or []) if isinstance(r, dict)],
    )


def fetch_engine(base: str = DEFAULT_ENGINE, timeout: float = _TIMEOUT, now: float | None = None) -> Sample:
    """GET the engine metrics endpoint. Never raises; failures land in ``error``."""
    import time as _time

    stamp = _time.time() if now is None else now
    try:
        return parse_engine(http_json(f"{base.rstrip('/')}/metrics", timeout=timeout), fetched_at=stamp)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Sample(ok=False, error=f"{type(exc).__name__}: {exc}", fetched_at=stamp)


# --- llama-swap side: Prometheus text format, parsed in monitor.prom ---


def swap_gpu_metrics(series: dict[str, Any]) -> dict[str, float]:
    """Pull the GPU gauges out of parsed llama-swap series.

    llama-swap publishes some GPU gauges unlabelled and the same names again with device labels. The
    labelled copies win when present, because on a multi-adapter box the unlabelled ones can belong to a
    different device than the one running inference. ``pick`` resolves that choice in one place instead of
    at every call site.
    """
    wanted = {
        "llamaswap_gpu_util_percent": "util",
        "llamaswap_gpu_memory_util_percent": "mem_pct",
        "llamaswap_gpu_power_draw_watts": "power_w",
        "llamaswap_gpu_fan_speed_percent": "fan_pct",
        "llamaswap_gpu_temperature_celsius": "temp_celsius",
        "llamaswap_gpu_memory_used_bytes": "mem_used_bytes",
        "llamaswap_gpu_memory_total_bytes": "mem_total_bytes",
        "llamaswap_gpu_power_limit_watts": "power_limit_w",
    }
    out: dict[str, float] = {}
    for metric, field_name in wanted.items():
        val = pick(series, metric, labelled=True)
        if val is None:
            val = pick(series, metric, labelled=False)
        if val is not None:
            out[field_name] = val
    gpu_name = label_value(series, "llamaswap_gpu_util_percent", "name")
    if gpu_name:
        out["name"] = gpu_name
    return out


def swap_memory_metrics(series: dict[str, Any]) -> dict[str, float]:
    """System RAM and CPU figures from llama-swap, kept separate from the engine's own readings.

    Both sources report RAM and neither value is averaged away anywhere in this project. A disagreement
    between them is the signal that something outside the inference process is holding memory, which is
    exactly what crosscheck.py reports.
    """
    out: dict[str, float] = {}
    for metric, field_name in (
        ("llamaswap_memory_total_bytes", "ram_total_bytes"),
        ("llamaswap_memory_used_bytes", "ram_used_bytes"),
        ("llamaswap_memory_free_bytes", "ram_free_bytes"),
    ):
        val = pick(series, metric)
        if val is not None:
            out[field_name] = val
    cores_mean, cores = summarize_cores(series, "llamaswap_cpu_util_percent")
    if cores_mean is not None:
        out["cpu_mean_pct"] = cores_mean
        out["cpu_core_count"] = float(cores)
    return out


def swap_models(payload: Any) -> list[tuple[str, str]]:
    """Return (model_id, status) pairs from llama-swap /models."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    rows: list[tuple[str, str]] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        status = ((item.get("status") or {}).get("value")) or "unloaded"
        rows.append((str(item.get("id") or "?"), str(status)))
    return rows


def fetch_swap(base: str = DEFAULT_SWAP, timeout: float = _TIMEOUT) -> dict[str, Any]:
    """Best-effort llama-swap snapshot. Returns a result dict with ``ok=False`` on failure.

    The metrics body and the model catalog are fetched independently. A server that answers /metrics but
    not /models still yields a usable hardware snapshot, which is what keeps the panel alive when the
    catalog endpoint is the part that broke.
    """
    result: dict[str, Any] = {"ok": False, "error": None, "gpu": {}, "memory": {}, "series": {}, "models": []}
    try:
        result["series"] = parse_prometheus(http_text(f"{base.rstrip('/')}/metrics", timeout=timeout))
        result["gpu"] = swap_gpu_metrics(result["series"])
        result["memory"] = swap_memory_metrics(result["series"])
        result["ok"] = True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["models"] = swap_models(http_json(f"{base.rstrip('/')}/models", timeout=timeout))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        pass  # model list is optional garnish, never fatal
    return result
