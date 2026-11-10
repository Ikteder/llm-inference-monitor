# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Command line for the model switcher. Read-only unless you pass --commit."""

from __future__ import annotations

import argparse

from monitor.collect import DEFAULT_ENGINE, DEFAULT_SWAP
from monitor.model_switch import ModelSwitcher, SwitchRefused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", help="model name or unique substring")
    ap.add_argument("--list", action="store_true", help="show the catalog and stop")
    ap.add_argument("--commit", action="store_true", help="perform the switch instead of previewing")
    ap.add_argument("--force", action="store_true", help="allow switching during a generation")
    ap.add_argument("--swap", default=DEFAULT_SWAP)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    args = ap.parse_args()

    sw = ModelSwitcher(args.swap, engine_base=args.engine)
    try:
        models = sw.refresh()
    except Exception as exc:  # noqa: BLE001 - a CLI should report reachability plainly
        print(f"llama-swap unreachable at {args.swap}: {type(exc).__name__}: {exc}")
        return 2

    busy, why = sw.engine_busy()
    for m in models:
        mark = "* live *" if m.active else ("ready" if m.status != "unloaded" else "not loaded")
        print(f"  {mark:10} {m.name}")
    print(f"engine: {'busy' if busy else 'idle'} ({why})")

    if args.list or not args.target:
        return 0

    try:
        if not args.commit:
            prev = sw.preview(args.target)
            if not prev["ok"]:
                print(prev["error"])
                return 2
            print(f"\npreview: {prev['target']} (status {prev['target_status']})")
            print(f"  would switch: {prev['would_switch']}")
            print(f"  engine busy:  {prev['engine_busy']} ({prev['busy_reason']})")
            print(f"  {prev['note']}")
            print("\nnothing sent. add --commit to perform the switch.")
            return 0
        print(f"\nswitch response: {sw.switch(args.target, force=args.force)}")
        return 0
    except (SwitchRefused, KeyError) as exc:
        print(f"\nnot switched: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
