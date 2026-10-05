"""Client-side monotonic timing and percentile summaries.

Durations are measured with :func:`time.monotonic`, which never goes backwards
and is immune to wall-clock adjustments, so a measured elapsed time cannot be
negative or distorted by clock changes. Three scopes matter for the benchmark:
each API attempt (including completed failed/retried attempts), each terminal
row-provider workflow (including a conditional second stage), and the whole
run.

An attempt that was interrupted before it finished has no end time; its
duration is ``None`` and it is *excluded* from percentile summaries rather than
being invented as a measured value.
"""

from __future__ import annotations

import math
import time


def monotonic() -> float:
    """A monotonic client clock reading in seconds."""
    return time.monotonic()


def elapsed(start: float, end: float | None) -> float | None:
    """Non-negative elapsed seconds, or ``None`` when the end was never recorded."""
    if end is None:
        return None
    value = end - start
    return value if value >= 0 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile of ``values`` (0 <= ``q`` <= 100).

    Returns ``None`` for an empty sample. ``values`` need not be pre-sorted.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (q / 100.0) * (len(clean) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[low]
    frac = rank - low
    return clean[low] * (1 - frac) + clean[high] * frac


def summarize(durations: list[float | None]) -> dict:
    """Summarise a list of durations, dropping ``None`` (unfinished) entries.

    ``count`` is the number of usable (finished) samples; the percentile
    denominator is explicit so interrupted attempts are never silently counted.
    """
    clean = [d for d in durations if d is not None]
    return {
        "count": len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": (sum(clean) / len(clean)) if clean else None,
    }
