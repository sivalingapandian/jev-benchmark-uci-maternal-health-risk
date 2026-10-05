"""Small live preflight across all three approved providers.

Before any bulk run, this makes a genuine bounded request to Jev and both
OpenRouter models with the shared two-stage task, using a clearly synthetic,
non-patient state. Each response is locally validated; the exposed model
identity, usage, and any failure are recorded in redacted form. If a credential
is missing, a schema is unsupported, a model refuses, or an exposed identity
mismatches, the provider fails and bulk execution must not proceed. Offline
fixtures cannot substitute for this check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import spec, workflow
from .errors import ProviderError, ProviderRequestError
from .providers import JevAdapter, OpenRouterAdapter
from .providers.base import ProviderAdapter, Usage

# A deliberately synthetic, non-patient record used only to exercise the task.
SYNTHETIC_STATE: dict = {
    "Age": 30,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 98.0,
    "HeartRate": 76,
}


@dataclass
class ProviderPreflight:
    requested_model: str
    ok: bool
    outcome: str
    actual_model: str | None
    final_label: str | None
    stage_count: int
    stages: list[dict]
    usage_totals: dict
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "requested_model": self.requested_model,
            "ok": self.ok,
            "outcome": self.outcome,
            "actual_model": self.actual_model,
            "final_label": self.final_label,
            "stage_count": self.stage_count,
            "stages": self.stages,
            "usage_totals": self.usage_totals,
            "detail": self.detail,
        }


@dataclass
class PreflightResult:
    config_hash: str
    providers: list[ProviderPreflight] = field(default_factory=list)

    @property
    def covered_models(self) -> set[str]:
        return {p.requested_model for p in self.providers}

    @property
    def ok(self) -> bool:
        """All three approved providers were checked and each returned a label."""
        if self.covered_models != set(spec.MODEL_IDS):
            return False
        return all(p.ok for p in self.providers)

    def as_dict(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "ok": self.ok,
            "providers": [p.as_dict() for p in self.providers],
        }


def _sum_usage(usages: list[Usage]) -> dict:
    input_tokens = sum(u.input_tokens or 0 for u in usages)
    output_tokens = sum(u.output_tokens or 0 for u in usages)
    costs = [u.cost for u in usages if u.cost is not None]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": sum(costs) if costs else None,
    }


def _usage_known(usages: list[Usage]) -> bool:
    """Every completed stage must report token usage.

    Missing usage is treated as unknown, never free and never a pass: a provider
    that returned a label but no accountable usage cannot be a complete preflight.
    """
    return bool(usages) and all(u.input_tokens is not None for u in usages)


def preflight_provider(adapter: ProviderAdapter, state: dict | None = None) -> ProviderPreflight:
    """Run one provider through the full two-stage task and summarise it."""
    state = SYNTHETIC_STATE if state is None else state
    outcome = workflow.evaluate_state(adapter, state)
    usage_known = _usage_known(outcome.usages)
    detail = outcome.detail
    if outcome.outcome == workflow.PREDICTED and not usage_known:
        detail = "provider returned a label but reported no usage (unknown, not a pass)"
    return ProviderPreflight(
        requested_model=adapter.model_id,
        ok=outcome.outcome == workflow.PREDICTED and usage_known,
        outcome=outcome.outcome,
        actual_model=outcome.actual_model,
        final_label=outcome.final_label,
        stage_count=outcome.stage_count,
        stages=[s.as_dict() for s in outcome.stages],
        usage_totals=_sum_usage(outcome.usages),
        detail=detail,
    )


def _failed_provider(model_id: str, detail: str) -> ProviderPreflight:
    return ProviderPreflight(
        requested_model=model_id,
        ok=False,
        outcome=workflow.ERROR,
        actual_model=None,
        final_label=None,
        stage_count=0,
        stages=[],
        usage_totals=_sum_usage([]),
        detail=detail,
    )


def run_preflight(adapters: list[ProviderAdapter], state: dict | None = None) -> PreflightResult:
    """Preflight an explicit list of adapters (used directly in tests)."""
    result = PreflightResult(config_hash=spec.config_hash())
    for adapter in adapters:
        try:
            result.providers.append(preflight_provider(adapter, state))
        except ProviderError as exc:  # defensive; evaluate_state already catches
            result.providers.append(_failed_provider(adapter.model_id, str(exc)))
    return result


def run_preflight_from_env(env: dict, state: dict | None = None) -> PreflightResult:
    """Build the three approved adapters from environment keys and preflight them.

    A missing credential is recorded as a failed provider rather than raising, so
    the result still reports which providers could and could not be reached.
    """
    result = PreflightResult(config_hash=spec.config_hash())
    typesafe_key = env.get("TYPESAFE_API_KEY")
    openrouter_key = env.get("OPENROUTER_API_KEY")

    specs: list[tuple[str, object]] = [
        ("jev-1.13.0", lambda: JevAdapter(typesafe_key)),
        (
            "openai/gpt-4o-mini",
            lambda: OpenRouterAdapter("openai/gpt-4o-mini", openrouter_key),
        ),
        (
            "google/gemini-2.5-flash",
            lambda: OpenRouterAdapter("google/gemini-2.5-flash", openrouter_key),
        ),
    ]

    for model_id, factory in specs:
        adapter = None
        try:
            adapter = factory()
        except ProviderRequestError as exc:
            result.providers.append(_failed_provider(model_id, str(exc)))
            continue
        try:
            result.providers.append(preflight_provider(adapter, state))
        finally:
            if adapter is not None:
                adapter.close()
    return result
