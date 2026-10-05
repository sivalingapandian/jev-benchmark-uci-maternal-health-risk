"""Timing and percentile tests, including the interrupted-attempt rule."""

import pytest

from jev_benchmark import timing


def test_monotonic_is_non_decreasing():
    a = timing.monotonic()
    b = timing.monotonic()
    assert b >= a


def test_elapsed_none_when_end_missing():
    assert timing.elapsed(10.0, None) is None


def test_elapsed_is_non_negative():
    assert timing.elapsed(5.0, 7.5) == pytest.approx(2.5)
    # A backwards clock would never happen with monotonic, but guard anyway.
    assert timing.elapsed(7.5, 5.0) == 0.0


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert timing.percentile(values, 50) == pytest.approx(2.5)
    assert timing.percentile(values, 95) == pytest.approx(3.85)


def test_percentile_empty_is_none():
    assert timing.percentile([], 50) is None


def test_summarize_excludes_interrupted_durations():
    # Two finished attempts and one interrupted (None) attempt: the interrupted
    # one is not invented as a measured duration and not counted.
    summary = timing.summarize([0.2, None, 0.4])
    assert summary["count"] == 2
    assert summary["p50"] == pytest.approx(0.30000000000000004)
    assert summary["min"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.4)


def test_summarize_all_none_is_empty():
    summary = timing.summarize([None, None])
    assert summary["count"] == 0
    assert summary["p50"] is None
