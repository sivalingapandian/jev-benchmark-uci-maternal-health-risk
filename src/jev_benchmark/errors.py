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


class ProviderError(BenchmarkError):
    """Base class for a failed provider stage request.

    Carries optional redacted metadata so a caller can record usage and the
    actually-returned model identity even when the decision itself is unusable.
    Never put secrets or raw authorization data on these.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: object | None = None,
        actual_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.actual_model = actual_model


class ProviderRequestError(ProviderError):
    """Transport, HTTP, authentication, or timeout failure reaching a provider."""


class InvalidDecisionError(ProviderError):
    """A response did not validate: malformed JSON/choice, missing required field,
    out-of-range probability, or an incomplete (truncated) answer."""


class RefusalError(ProviderError):
    """The provider explicitly refused to answer the bounded question."""


class ModelMismatchError(ProviderError):
    """The provider exposed a model identity other than the requested one.

    A silent substitution must block the benchmark rather than be scored.
    """


class BudgetExceeded(BenchmarkError):
    """A paid dispatch was refused because the conservative exposure guard would
    be breached.

    Carries the exposure that would result and the shared cap so the runner can
    stop cleanly, record why, and leave completed work intact and billable only
    once. Deliberately *not* a :class:`ProviderError`, so the two-stage workflow
    does not swallow it as a per-stage outcome; it propagates up to stop the run.
    """

    def __init__(self, message: str, *, exposure: float, cap: float) -> None:
        super().__init__(message)
        self.exposure = exposure
        self.cap = cap


class RunIdentityError(BenchmarkError):
    """A resume target does not match the current dataset/task specification.

    Reusing results from a different dataset or frozen specification would
    silently mix incompatible experiments, so resume refuses instead.
    """


class RunDirectoryError(BenchmarkError):
    """A fresh (non-resume) run was pointed at an already-occupied directory.

    Appending a new run on top of an existing one would duplicate cases and
    overwrite the prior header, so a fresh run refuses an occupied directory and
    the user must choose a new directory or pass ``--resume``.
    """


class BudgetConfigError(BenchmarkError):
    """A user-configured budget cap is invalid (e.g. above the approved $20)."""


class HistoricalExposureError(BenchmarkError):
    """Prior paid exposure recorded on disk could not be safely bounded.

    Rather than guess, the shared budget refuses to initialise so the operator
    can reconcile the historical spend before any further paid calls.
    """
