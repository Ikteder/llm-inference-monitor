# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Run the cache-truth probe from the command line.

    python -m monitor.cache_truth_cli [--engine URL] [--rounds 4]

Exits 0 when the measured steady-state cost confirms the engine's own cache counters, and 2 when
the two disagree, so it can gate a benchmark script.
"""

from __future__ import annotations

import argparse
import sys

from monitor.cache_truth import measure_cache_truth
from monitor.collect import DEFAULT_ENGINE

DEFAULT_PROMPT_TOKENS = 3000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS,
                    help="synthetic prompt size; must be large enough to prefill measurably")
    args = ap.parse_args()

    r = measure_cache_truth(base=args.engine, rounds=args.rounds, prompt_tokens=args.prompt_tokens)
    print(f"cold      {r.cold_s:7.2f} s")
    print(f"steady    {r.warm_s:7.2f} s")
    if r.speedup is not None:
        print(f"speedup   {r.speedup:7.2f}x")
    if r.first_claimed_hit_s is not None:
        print(f"first claimed hit  {r.first_claimed_hit_s:6.2f} s")
    print(f"engine reported {r.reported_cached_tokens:,} cached of {r.reported_prompt_tokens:,} prompt tokens")
    print(f"\nverdict: {r.verdict}")
    return 0 if r.verdict.startswith("confirmed") else 2


if __name__ == "__main__":
    sys.exit(main())
