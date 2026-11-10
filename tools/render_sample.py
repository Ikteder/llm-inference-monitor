# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""A canned sample shaped like the live engine's, shared by the render tools and tests.

Values are chosen to exercise every branch of the drawing code at once: busy state, a VRAM bar in
the warning band, calibrated spare tokens, a projectable trend, a hot GPU reading (temperature over
80C and power within 10% of the limit), and a non-empty economics block.

The numbers are internally consistent: reused is under prompt, drafts accepted is under drafts
offered, decode_ms and output_tokens imply ~21 tok/s, and the VRAM figures sum to the card total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Works both as a package import (installed tree) and as a direct script run from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.collect import Sample

GPU_TOTAL = 12_878_610_432  # 12 GiB, as reported by a 12 GB card


def canned_sample(fetched_at: float | None = None) -> Sample:
    return Sample(
        ok=True,
        fetched_at=fetched_at if fetched_at is not None else time.time(),
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
        gpu_name="NVIDIA GeForce RTX 4070 SUPER",
        gpu_util=98.0,
        gpu_mem_used=int(0.88 * GPU_TOTAL),
        gpu_mem_total=GPU_TOTAL,
        gpu_temp=83.0,
        gpu_power=205.0,
        gpu_power_limit=220.0,
        cpu=47.0,
        ram_used=int(29.5 * 1024**3),
        ram_total=int(31.9 * 1024**3),
        hist_tok_s=[18 + (i % 5) for i in range(40)],
        hist_gpu_util=[90 + (i % 8) for i in range(40)],
        hist_gpu_mem_used=[int(0.80 * GPU_TOTAL) + i * 900_000 for i in range(40)],
        hist_cpu=[40 + (i % 10) for i in range(40)],
        requests_detail=[
            {
                "id": "a1",
                "prompt_tokens": 38540,
                "reused": 38276,
                "prompt_read": 264,
                "output_tokens": 273,
                "prompt_ms": 2410.0,
                "decode_ms": 11762.4,
                "decode_tok_s": 23.2,
                "hit_rate": 0.491,
                "drafts_offered": 227,
                "drafts_accepted": 211,
            }
        ],
    )
