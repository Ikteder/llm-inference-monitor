# Copyright (c) 2026 Ikteder
# SPDX-License-Identifier: Apache-2.0
"""Parser and value normalization for the two metric sources.

Two formats are in play and they do not agree with each other:

* the engine serves JSON from ``/metrics``, whose field names and units vary by build;
* llama-swap serves Prometheus text exposition, where every sample carries a label set.

Both parsers live here so they can be tested against captured text with no network involved. The HTTP
calls stay in collect.py; everything below is pure and deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_UNSET = object()

# name{labels} value  |  name[value=...] value (llama-swap's own bracket spelling)
_SAMPLE_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?P<labels>\{[^{}]*\}|\[[^\[\]]*\])?"
    r"\s+(?P<value>[-+]?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NaN))$"
)
_LABEL_PAIR = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<val>"[^"]*"|\'[^\']*\'|[^\s,\]]+)')


def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten a Prometheus text body into ``name[k=v,k=v]`` -> value.

    Labels are kept in the key and sorted, because the exposition format does not promise label order
    and a lookup that depends on it breaks between builds. Values that do not parse as numbers are
    skipped rather than coerced to zero: a silently zeroed gauge makes a dashboard look healthy while it
    is reporting nothing. Comment lines (``# TYPE``, ``# HELP``), bare names with no value, and NaN
    samples are ignored.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_LINE.match(line)
        if not m:
            continue
        val = as_float(m.group("value"))
        if val is None:
            continue
        labels = m.group("labels")
        if labels:
            pairs = {pm.group("key"): _unquote(pm.group("val")) for pm in _LABEL_PAIR.finditer(labels[1:-1])}
            key = f"{m.group('name')}[{','.join(f'{k}={v}' for k, v in sorted(pairs.items()))}]"
        else:
            key = m.group("name")
        out[key] = val
    return out


def _unquote(raw: str) -> str:
    return raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"} else raw


def metric_name(key: str) -> str:
    """Strip any label set from a sample key."""
    for opener in ("{", "["):
        if opener in key:
            return key.split(opener, 1)[0]
    return key


def labels_of(key: str) -> dict[str, str]:
    """Parse a sample key's label set back into a dict. Empty when the key has none."""
    for opener, closer in (("{", "}"), ("[", "]")):
        if key.endswith(closer) and opener in key:
            body = key[key.index(opener) + 1 : -1]
            return {pm.group("key"): _unquote(pm.group("val")) for pm in _LABEL_PAIR.finditer(body)}
    return {}


def pick(series: dict[str, Any], metric: str, *, labelled: bool | None = None) -> float | None:
    """First value for a metric name regardless of its label set.

    ``labelled`` restricts the search to series with (True) or without (False) labels; llama-swap emits
    both forms for some gauges and picking the wrong one silently reports a different device. None means
    "this build does not publish it", which callers must keep distinct from a real zero.
    """
    for key, val in series.items():
        if metric_name(key) != metric:
            continue
        has_labels = any(opener in key for opener in ("{", "["))
        if labelled is not None and has_labels != labelled:
            continue
        out = as_float(val)
        if out is not None:
            return out
    return None


def pick_all(series: dict[str, Any], metric: str) -> list[float]:
    """Every value for a metric name, e.g. one entry per CPU core or per GPU."""
    out: list[float] = []
    for key, val in series.items():
        if metric_name(key) != metric:
            continue
        got = as_float(val)
        if got is not None:
            out.append(got)
    return out


def label_value(series: dict[str, Any], metric: str, label: str) -> str:
    """First value of a named label for a metric, e.g. the reported GPU name."""
    for key in series:
        if metric_name(key) != metric:
            continue
        got = labels_of(key).get(label)
        if got:
            return got
    return ""


def as_float(value: Any) -> float | None:
    """Float or None. Rejects bool and NaN so neither can masquerade as a reading."""
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def as_int(value: Any, default: int = 0) -> int:
    got = as_float(value)
    return default if got is None else int(got)


def first_present(mapping: Any, *keys: str, default: Any = _UNSET) -> Any:
    """First present key from a payload, without treating falsy values as missing.

    ``mapping.get(k) or fallback`` hides a counter that legitimately reads 0 right after boot, which is
    how a monitor starts reporting "no data" for an actual zero. Key presence decides, not truthiness.
    """
    for k in keys:
        if isinstance(mapping, dict) and k in mapping:
            return mapping[k]
    return None if default is _UNSET else default


def to_mib(value: Any) -> float | None:
    """MiB figure from a field that may be labeled MiB or arrive as bytes.

    The engine labels some gauges in MiB and others in bytes with no suffix at all, so magnitude is the
    only discriminator. At least 10^7 cannot be a plausible MiB reading on a consumer GPU (~10 TB), and
    below 10^4 cannot be a byte count for a real allocation (~10 KB). Boundary values are pinned in tests
    because getting this factor of 1024 wrong is the most damaging unit error available here.
    """
    v = as_float(value)
    if v is None:
        return None
    if abs(v) >= 10_000_000:
        return v / (1024.0 * 1024.0)
    return v


def to_bytes(value: Any) -> int | None:
    """Byte count from a field that may be labeled either way. Inverse of :func:`to_mib`."""
    v = as_float(value)
    if v is None:
        return None
    if abs(v) < 10_000_000:
        return int(v * 1024 * 1024)
    return int(v)


def history(values: Any, *, as_integer: bool = False, digits: int = 2) -> list[Any]:
    """Clean an engine-supplied rolling array: drop holes, round, optionally cast to int.

    Engines emit nulls for samples they never collected. Keeping them shifts every later point one slot
    left in a sparkline, which makes the graph lie about when something happened.
    """
    out: list[Any] = []
    for raw in values or []:
        v = as_float(raw)
        if v is None:
            continue
        out.append(int(v) if as_integer else round(v, digits))
    return out


def summarize_cores(series: dict[str, Any], metric: str) -> tuple[float | None, int]:
    """Mean of a per-core gauge plus the core count it was taken over.

    Returning the count matters: a mean over 2 cores and a mean over 32 look identical otherwise, and
    the difference changes whether "CPU is busy" means anything.
    """
    vals = pick_all(series, metric)
    if not vals:
        return (None, 0)
    return (sum(vals) / len(vals), len(vals))
