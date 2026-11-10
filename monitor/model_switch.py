# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Model switching through llama-swap, with an explicit commit step.

llama-swap already owns model lifecycle on this box: it holds the list of loaded and loadable
models and performs the actual swap. What it does not give you is a safe way to ask "what would
this switch cost me right now", which is the question that matters when a generation is in flight
and an unattended job is mid-run.

The split here is deliberate:

* ``list_models`` / ``status`` are read-only and always safe.
* ``resolve_target`` and ``build_switch_request`` prepare everything but send nothing.
* ``switch`` is the only method that mutates state, and it refuses unless the engine is idle
  and the caller passes ``force=True`` past an in-flight generation.

That boundary exists because a model swap mid-generation silently truncates the assistant turn
and the harness usually retries against the new weights without noticing. Tests cover the refusal
paths; no test ever calls ``switch`` for real, because that would change what is loaded on the
user's machine as a side effect of running a test suite.

Endpoints used (llama-swap 4.x):

    GET  /models                 catalog with per-model status ('loaded' means it is live)
    POST /switch {"model": name} perform the swap

The engine's own /metrics model name is cross-checked against the catalog, because llama-swap has no
endpoint that says "this is the active model" in current builds.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from monitor.collect import DEFAULT_SWAP, http_json

_TIMEOUT = 5.0


class SwitchRefused(RuntimeError):
    """Raised when a switch would be unsafe. Nothing was sent to the server."""


@dataclass
class ModelInfo:
    name: str
    status: str
    active: bool
    loaded: bool
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ready_to_switch(self) -> bool:
        """A model llama-swap says is available and is not already the live one."""
        return self.status in ("ready", "loaded", "loaded-in-ram") and not self.active


def loaded_names(base: str, timeout: float = _TIMEOUT) -> set[str]:
    """Models llama-swap itself flags as live.

    Current builds mark this on the catalog entry (``status.value == "loaded"``) and return 404 for
    ``/loaded``; older builds answered ``/loaded``. Both are tried so a server upgrade does not leave
    the monitor blind to which model is running.
    """
    base = base.rstrip("/")
    out: set[str] = set()
    try:
        payload = http_json(f"{base}/models", timeout=timeout)
        data = payload.get("data") if isinstance(payload, dict) else payload
        for item in data or []:
            if not isinstance(item, dict):
                continue
            raw = item.get("status")
            status = str((raw or {}).get("value") if isinstance(raw, dict) else raw or "").lower()
            if status == "loaded" or item.get("active"):
                out.add(str(item.get("id") or item.get("name") or ""))
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        payload = http_json(f"{base}/loaded", timeout=timeout)
        data = payload.get("data") if isinstance(payload, dict) else payload
        for item in data or []:
            if isinstance(item, dict):
                out.add(str(item.get("id") or item.get("name") or ""))
            elif isinstance(item, str):
                out.add(item)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return {n for n in out if n}


def engine_model_name(engine_base: str, timeout: float = _TIMEOUT) -> str:
    """The model id the inference engine reports it is running, or '' if unreachable."""
    if not engine_base:
        return ""
    try:
        m = http_json(f"{engine_base.rstrip('/')}/metrics", timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return ""
    eng = m.get("engine") or {}
    return str(eng.get("model") or eng.get("model_name") or "")


def parse_models(payload: Any, loaded_names: set[str]) -> list[ModelInfo]:
    """Normalize llama-swap's catalog into ModelInfo rows.

    llama-swap reports status as a nested object (``{"status": {"value": "ready"}}``) in newer
    builds and as a bare string in older ones; both shapes are accepted so a monitor against an
    upgraded server does not silently report everything as unloaded.
    """
    data = payload.get("data") if isinstance(payload, dict) else payload
    rows: list[ModelInfo] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or item.get("name") or "?")
        raw_status = item.get("status")
        if isinstance(raw_status, dict):
            status = str(raw_status.get("value") or "unloaded")
        else:
            status = str(raw_status or "unloaded")
        status = status.lower()
        rows.append(
            ModelInfo(
                name=name,
                status=status,
                active=bool(item.get("active")) or name in loaded_names,
                loaded=status != "unloaded" or name in loaded_names,
                detail=item,
            )
        )
    return rows


def resolve_active(catalog_names: list[str], engine_name: str, swap_live: set[str]) -> set[str]:
    """Decide which catalog entry is the live one.

    Prefers an exact id match against what the engine says it is running, then scores candidates by
    distinctive name tokens (quant labels like ``q4_k_m`` are ignored because they collide across
    models). The best candidate wins only if it leads outright; ties and zero matches fall back to
    llama-swap's own 'loaded' flag rather than guessing, since the engine reports GGUF file names while
    llama-swap catalogues config aliases and the two strings are often unrelated.
    """
    if not engine_name:
        return set(swap_live)
    low = engine_name.lower()
    exact = [n for n in catalog_names if n.lower() == low]
    if exact:
        return {exact[0]}

    def tokens(name: str) -> set[str]:
        parts = {t for t in re.split(r"[^a-z0-9.]+", name.lower()) if len(t) > 2}
        return {t for t in parts if t not in _GENERIC_TOKENS}

    eng = tokens(engine_name)
    scores = [(len(eng & tokens(n)), n) for n in catalog_names]
    best = max((s for s, _ in scores), default=0)
    winners = [n for s, n in scores if s == best and s > 0]
    if len(winners) == 1:
        return {winners[0]}
    return set(swap_live)


_GENERIC_TOKENS = {
    # Quantisation and packaging labels: shared by nearly every entry in a catalog, so they carry
    # no signal about which model is actually running.
    "gguf", "q4", "q4_k_m", "q5", "q5_k_m", "q8", "fp16", "bf16", "instruct", "chat",
    "model", "models", "ud", "gsq", "abliterated", "heretic",
}


class ModelSwitcher:
    def __init__(self, swap_base: str = DEFAULT_SWAP, engine_base: str | None = None) -> None:
        self.swap_base = swap_base.rstrip("/")
        self.engine_base = (engine_base or "").rstrip("/")
        self._models: list[ModelInfo] = []

    # --- read-only ---------------------------------------------------------

    def refresh(self) -> list[ModelInfo]:
        swap_live = loaded_names(self.swap_base)
        try:
            payload = http_json(f"{self.swap_base}/models", timeout=_TIMEOUT)
        except (urllib.error.URLError, OSError, ValueError):
            payload = {"data": []}
        rows = parse_models(payload, set())
        names = [r.name for r in rows]
        active = resolve_active(names, engine_model_name(self.engine_base), swap_live)
        for row in rows:
            if row.name in active:
                row.active = True
                row.loaded = True
                if row.status == "unloaded":
                    row.status = "loaded"
        self._models = rows
        return rows

    @property
    def models(self) -> list[ModelInfo]:
        return self._models or self.refresh()

    def engine_busy(self) -> tuple[bool, str]:
        """Ask the engine whether a generation is in flight. Unknown counts as busy."""
        if not self.engine_base:
            return True, "no engine URL configured; cannot prove the engine is idle"
        try:
            m = http_json(f"{self.engine_base}/metrics", timeout=3.0)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return True, f"engine unreachable ({type(exc).__name__})"
        live = m.get("live") or {}
        state = str(live.get("state") or "idle")
        if state in ("generating", "prefilling", "prompt_eval") or int(live.get("queued") or 0) > 0:
            return True, f"engine state is {state!r}"
        return False, state

    # --- preparation (sends nothing) ---------------------------------------

    def resolve_target(self, name: str) -> str:
        """Map a user-supplied name to a real catalog id, or raise.

        Accepts an exact id, or a unique case-insensitive substring, so "fast" can pick
        "swap:fast-model". Ambiguity raises rather than guessing, because switching to the
        wrong weights is worse than not switching.
        """
        models = self.models
        exact = [m for m in models if m.name == name]
        if exact:
            return exact[0].name
        probe = name.lower()
        hits = [m for m in models if probe in m.name.lower()]
        if len(hits) == 1:
            return hits[0].name
        if not hits:
            raise KeyError(f"no model matches {name!r}; available: {', '.join(m.name for m in models)}")
        raise KeyError(f"{name!r} is ambiguous: {', '.join(m.name for m in hits)}")

    def build_switch_request(self, name: str) -> dict[str, Any]:
        """The exact JSON body that would be POSTed. Pure, so it can be asserted in tests."""
        return {"model": self.resolve_target(name)}

    # --- the only mutating call --------------------------------------------

    def switch(self, name: str, force: bool = False) -> dict[str, Any]:
        """Perform a model swap. Refuses if the engine is busy unless force is set.

        Returns llama-swap's response on success. Raises SwitchRefused before any request is
        sent when the switch would interrupt work in progress.
        """
        target = self.resolve_target(name)
        current = {m.name for m in self.models if m.active}
        if current == {target}:
            return {"noop": True, "model": target, "reason": "already active"}

        busy, why = self.engine_busy()
        if busy and not force:
            raise SwitchRefused(
                f"refusing to swap models while {why}. Wait for the run to finish, or pass "
                f"force=True if you accept that the current generation is lost."
            )

        body = json.dumps({"model": target}).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 - fixed localhost endpoint
            f"{self.swap_base}/switch",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw) if raw.strip() else {"ok": True, "model": target}
        except ValueError:
            return {"ok": True, "model": target, "response": raw}

    # --- what a switch would cost, without doing it -------------------------

    def preview(self, name: str) -> dict[str, Any]:
        """Everything you would want to know before committing to a swap.

        Reports the target's status, whether the engine is idle enough to allow it, and how much
        context the current model has resident, because that is what gets thrown away: a swap
        drops the KV cache, so the next turn re-prefills from zero.
        """
        try:
            target = self.resolve_target(name)
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}

        row = next((m for m in self.models if m.name == target), None)
        busy, why = self.engine_busy()

        resident = 0
        try:
            m = http_json(f"{self.engine_base or self.swap_base}/metrics", timeout=3.0)
            totals = m.get("totals") or {}
            resident = int(totals.get("prompt_tokens") or 0)
        except (urllib.error.URLError, OSError, ValueError):
            pass

        return {
            "ok": True,
            "target": target,
            "target_status": (row.status if row else "unknown"),
            "would_switch": bool(row and not row.active),
            "engine_busy": busy,
            "busy_reason": why,
            "resident_prompt_tokens_lost": resident,
            "note": (
                "a swap drops the KV cache; the next turn re-prefills its whole prompt. "
                f"Current session has {resident:,} prompt tokens resident."
            ),
        }
