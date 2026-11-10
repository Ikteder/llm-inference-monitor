# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Calibration fitting and the stored calibration record.

Kept apart from the live probe (monitor.calibrate) so the arithmetic can be tested on synthetic points.
The headroom estimate in monitor.vram depends on a bytes-per-resident-token coefficient; this module owns
where that number comes from, how it is validated, and how it survives between runs.

Two facts drive the design:

* The coefficient is machine-specific. It depends on GPU, quantization, KV geometry and whatever arena or
  expert cache the build reserves, so no published constant is correct for someone else's box. Shipping a
  number and calling it measured would be the same mistake this project exists to catch.
* A wrong coefficient is worse than no coefficient. A plausible but unmeasured bytes/token turns headroom
  into a confident recommendation that fails at load time, so an uncalibrated record refuses to produce a
  spare-token figure rather than substituting a default.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_MIN_PAIR_GAP = 512  # token gap below which a slope is allocator jitter rather than signal
_MAX_PLAUSIBLE_BYTES_PER_TOKEN = 65_536.0  # ~4x the widest dense q8 KV row seen on consumer hardware


def slopes(rows: list[tuple[int, float]], min_token_gap: int = _MIN_PAIR_GAP) -> list[tuple[str, float]]:
    """Bytes per token from every probe pair whose token counts differ enough to be meaningful.

    Pairs closer than ``min_token_gap`` produce slopes dominated by allocator jitter, so they are dropped
    rather than averaged in. Each result carries its pair description, because when two pairs disagree the
    reader needs to know which probes produced which number.
    """
    out: list[tuple[str, float]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d_tokens = rows[j][0] - rows[i][0]
            if abs(d_tokens) < min_token_gap:
                continue
            out.append((f"{rows[i][0]} -> {rows[j][0]} tokens", (rows[j][1] - rows[i][1]) / d_tokens))
    return out


def fit(rows: list[tuple[int, float]]) -> dict[str, Any]:
    """Coefficient summary for (resident tokens, VRAM bytes) points.

    Median rather than mean: one allocator spike during a run would drag a mean across a four point
    regression and there is no reason to trust such an outlier. Negative slopes mean memory fell as context
    grew, which happens when a probe evicted something; those pairs are excluded rather than averaged,
    because a negative bytes/token silently turns headroom into nonsense.

    ``spread`` is returned alongside the coefficient on purpose. A wide spread means the probes disagree and
    the median is hiding it, so a caller should re-run with bigger steps instead of trusting the number.
    """
    pairs = slopes(rows)
    if not pairs:
        return {"bytes_per_token": None, "spread": None, "pairs": 0, "rejected": 0}
    vals = sorted(v for _, v in pairs)
    positive = [v for v in vals if v > 0]
    mid = len(positive) // 2
    median = positive[mid] if len(positive) % 2 else (positive[mid - 1] + positive[mid]) / 2.0
    return {
        "bytes_per_token": median if positive else None,
        "spread": vals[-1] - vals[0],
        "pairs": len(vals),
        "rejected": len(vals) - len(positive),
    }


def plausible(bytes_per_token: Any) -> bool:
    """Is a stored coefficient inside the range a real KV cache could occupy?

    Guards against unit errors, which are the failure mode with the most damage available here: a value in
    bytes rather than MiB-per-token, or a slope taken against a character count instead of tokens, both land
    one to three orders of magnitude off and still look like numbers. The lower bound is not zero because a
    sub-100-bytes/token figure means the probes barely grew anything and the slope is noise.
    """
    got = float(bytes_per_token) if isinstance(bytes_per_token, (int, float)) and not isinstance(bytes_per_token, bool) else None
    if got is None or got != got:
        return False
    return 100.0 <= got <= _MAX_PLAUSIBLE_BYTES_PER_TOKEN


@dataclass
class Calibration:
    """Calibration measured on this machine, or an explicit record that nothing has been measured.

    ``bytes_per_token`` is None until a probe run fills it in. That is the important distinction in this
    file: a missing coefficient disables the spare-token figure, while a present one was measured here and
    can be quoted.
    """

    bytes_per_token: float | None = None
    reserved_floor_bytes: int | None = None
    calibrated_at: str = ""
    engine_version: str = ""
    gpu_name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        """True only when a coefficient exists and survives the plausibility gate."""
        return plausible(self.bytes_per_token)

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        """Read the stored record. Never raises; problems land in ``notes`` and disable calibration."""
        p = path or default_path()
        if not p.exists():
            return cls(notes=["not calibrated: run `python -m monitor.calibrate --save` against a live engine"])
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return cls(notes=[f"calibration file unreadable ({type(exc).__name__}); treating as uncalibrated"])
        if not isinstance(data, dict):
            return cls(notes=["calibration file is not a JSON object; treating as uncalibrated"])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "notes"}
        extra = sorted(set(data) - set(known))
        notes = [str(n) for n in (data.get("notes") or [])]
        if extra:
            notes.append(f"ignored unknown keys: {', '.join(extra)}")
        bpt = known.get("bytes_per_token")
        if bpt is not None and not plausible(bpt):
            notes.append(
                f"stored bytes_per_token={bpt} is outside the plausible range; ignoring it. "
                "Re-run the probe, and check that gpu_mem_used is a per-device figure on this build."
            )
            known["bytes_per_token"] = None
        floor = known.get("reserved_floor_bytes")
        known["reserved_floor_bytes"] = int(floor) if isinstance(floor, (int, float)) and floor >= 0 else None
        return cls(**known, notes=notes)

    def save(self, path: Path | None = None) -> Path:
        """Write atomically. A crash mid-write must not leave a half-written coefficient behind."""
        p = path or default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload.pop("notes", None)  # load-time commentary, not calibration data
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        return p

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_path() -> Path:
    """Per-user calibration store, overridable with LLM_MONITOR_CALIBRATION.

    Deliberately outside the source tree: the file records one machine's GPU and engine build, so a checkout
    shared between machines must not carry a coefficient that belongs to neither.
    """
    env = os.environ.get("LLM_MONITOR_CALIBRATION")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "llm-inference-monitor" / "calibration.json"
