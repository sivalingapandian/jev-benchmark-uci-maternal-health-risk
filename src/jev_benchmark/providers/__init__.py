"""Live provider adapters exposing a common typed decision interface.

Each adapter turns one frozen stage question plus the six permitted
measurements into either a validated :class:`StageDecision` or a typed
:class:`~jev_benchmark.errors.ProviderError`. The two-stage branching, label
mapping, and invalid/abstained bookkeeping live in
:mod:`jev_benchmark.workflow`; adapters only call a provider and validate the
single response in front of them.
"""

from __future__ import annotations

from .base import ProviderAdapter, StageDecision, Usage, validate_choice_response
from .jev import JevAdapter
from .openrouter import OpenRouterAdapter

__all__ = [
    "ProviderAdapter",
    "StageDecision",
    "Usage",
    "validate_choice_response",
    "JevAdapter",
    "OpenRouterAdapter",
]
