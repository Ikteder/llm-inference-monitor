# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Cross-checks derived metrics against the engine's own per-request figures.

The engine already reports decode_tok_s and hit_rate for each recent request. If the
cumulative aggregates disagree with those, one of the two is being read wrong. Run this
against a live engine; it prints the spread rather than asserting a single number, because
legitimate variation between requests is real and a pass/fail gate would be theater.

Usage:  python -m monitor.crosscheck [--engine URL] [--tolerance 0.25]
"""

from __future__ import annotations

import argparse

from monitor.collect import DEFAULT_ENGINE, fetch_engine
from monitor.economics import context_economics, speculation


def _spread(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    return (min(values), sum(values) / len(values), max(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--tolerance", type=float, default=0.25, help="relative agreement gate")
    args = ap.parse_args()

    s = fetch_engine(args.engine)
    if not s.ok:
        print(f"engine unreachable: {s.error}")
        return 1
    if not s.requests_detail:
        print("no request records yet; run a few inferences first")
        return 1

    econ = context_economics(s)
    spec = speculation(s)

    decode_reported = [float(r["decode_tok_s"]) for r in s.requests_detail if r.get("decode_tok_s")]
    hit_reported = [float(r["hit_rate"]) for r in s.requests_detail if isinstance(r.get("hit_rate"), (int, float))]
    acc_reported = [
        float(r["drafts_accepted"]) / float(r["drafts_offered"])
        for r in s.requests_detail
        if r.get("drafts_offered")
    ]

    problems = 0

    def compare(label: str, derived: float | None, reported: list[float]) -> None:
        nonlocal problems
        if derived is None or not reported:
            print(f"  {label:<26} no data to compare")
            return
        lo, mean, hi = _spread(reported)
        rel = abs(derived - mean) / mean if mean else 0.0
        flag = "OK" if rel <= args.tolerance else "CHECK"
        if flag != "OK":
            problems += 1
        print(
            f"  {label:<26} derived {derived:9.3f} | reported mean {mean:9.3f}"
            f" (range {lo:.3f}..{hi:.3f}) rel {rel * 100:5.1f}%  [{flag}]"
        )

    print(f"engine: {args.engine}   requests sampled: {len(s.requests_detail)}")
    compare("decode tok/s", econ.mean_decode_tok_s, decode_reported)
    compare("prompt cache hit rate", econ.cache_hit_rate, hit_reported)
    compare("draft acceptance rate", spec.acceptance_rate, acc_reported)

    # Internal consistency: aggregates must be at least as large as the newest single request.
    newest = s.requests_detail[0]
    checks = [
        ("prompt_tokens >= last request", s.prompt_tokens >= int(newest.get("prompt_tokens") or 0)),
        ("output_tokens >= last request", s.output_tokens >= int(newest.get("output_tokens") or 0)),
        ("reused <= prompt_tokens", econ.reused_tokens <= econ.prompt_tokens),
        ("fresh = prompt - reused >= 0", econ.fresh_prompt_tokens >= 0),
        ("drafts_accepted <= drafts_offered", s.drafts_accepted <= s.drafts_offered),
    ]
    print("\n  internal consistency:")
    for label, ok in checks:
        print(f"    {label:<34} {'OK' if ok else 'FAIL'}")
        if not ok:
            problems += 1

    print(f"\n{'all within tolerance' if not problems else str(problems) + ' disagreement(s)'}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
