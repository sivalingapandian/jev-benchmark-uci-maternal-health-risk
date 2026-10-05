"""Benchmark-specific exceptions.

These let the CLI distinguish an expected, user-facing failure (bad source,
bad schema) from an unexpected crash and exit with a non-zero status.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for expected, reportable benchmark failures."""


class SourceError(BenchmarkError):
    """The official source could not be fetched or read."""


class SchemaError(BenchmarkError):
    """The source columns or labels do not match the frozen specification."""
