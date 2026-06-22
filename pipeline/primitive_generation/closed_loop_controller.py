"""
Explicit closed-loop controller for PASG primitive generation.

The single-pass builder constructs primitives from the current geometry and
semantic alignment response. This controller adds the PASG outer loop: inspect
missing or low-confidence primitives, adapt the sampling budget or request a
refreshed semantic response through a callback, and rebuild until convergence
or the iteration budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal



ObjectClass = Literal["normal", "rjo", "fco"]
RefineCallback = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class RefinementEvent:
    """One closed-loop refinement decision."""

    iteration: int
    reasons: list[str]
    sample_budget: int
    refined: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "reasons": self.reasons,
            "sample_budget": self.sample_budget,
            "refined": self.refined,
        }


@dataclass
class ClosedLoopResult:
    """PASG closed-loop output with a trace of refinement decisions."""

    result: dict[str, Any]
    converged: bool
    refinement_history: list[RefinementEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.result)
        payload["closed_loop"] = {
            "converged": self.converged,
            "iterations": len(self.refinement_history),
            "history": [event.to_dict() for event in self.refinement_history],
        }
        return payload


def _primitive_confidence(primitive: dict[str, Any]) -> float:
    return float(primitive.get("confidence", 1.0))


def _as_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def inspect_primitives(
    result: dict[str, Any],
    *,
    required_primitives: Iterable[str] = ("grasp", "place", "utility"),
    confidence_threshold: float = 0.7,
) -> list[str]:
    """Return reasons that require another closed-loop refinement round."""

    reasons: list[str] = []
    operation = result.get("operation_primitives", {})
    axes = operation.get("axes", result.get("axis_primitives", result.get("primitives", {})))
    points = operation.get("points", result.get("point_primitives", {}))

    for name in required_primitives:
        axis_entries = _as_entries(axes.get(name))
        point_entries = _as_entries(points.get(name))
        if not axis_entries and not point_entries:
            reasons.append(f"missing primitive: {name}")
            continue

        if axis_entries:
            best_axis_confidence = max(_primitive_confidence(item) for item in axis_entries)
            if best_axis_confidence < confidence_threshold:
                reasons.append(
                    f"low confidence: {name} axis={best_axis_confidence:.3f} < {confidence_threshold:.3f}"
                )
        if point_entries:
            best_point_confidence = max(_primitive_confidence(item) for item in point_entries)
            if best_point_confidence < confidence_threshold:
                reasons.append(
                    f"low confidence: {name} point={best_point_confidence:.3f} < {confidence_threshold:.3f}"
                )

    for branch in ("rjo", "fco"):
        if branch in result and result[branch].get("confidence", 1.0) < confidence_threshold:
            confidence = float(result[branch].get("confidence", 1.0))
            reasons.append(
                f"low confidence: {branch}={confidence:.3f} < {confidence_threshold:.3f}"
            )

    return reasons


def run_closed_loop_refinement(
    *,
    build_kwargs: dict[str, Any],
    refine_fn: RefineCallback | None = None,
    object_class: ObjectClass = "normal",
    required_primitives: Iterable[str] = ("grasp", "place", "utility"),
    confidence_threshold: float = 0.7,
    max_iterations: int = 2,
    sample_budget_step: int = 10,
) -> ClosedLoopResult:
    """Run PASG primitive generation with confidence-triggered refinement."""

    from .primitive_builder import build_primitives

    current_kwargs = dict(build_kwargs)
    current_kwargs["object_class"] = object_class
    history: list[RefinementEvent] = []

    result = build_primitives(**current_kwargs)
    reasons = inspect_primitives(
        result,
        required_primitives=required_primitives,
        confidence_threshold=confidence_threshold,
    )

    iteration = 0
    while reasons and iteration < max_iterations:
        iteration += 1
        current_kwargs["sample_budget"] = int(
            current_kwargs.get("sample_budget", 25)
        ) + sample_budget_step

        context = {
            "iteration": iteration,
            "reasons": reasons,
            "object_class": object_class,
            "result": result,
            "build_kwargs": dict(current_kwargs),
        }

        refined = False
        if refine_fn is not None:
            updates = refine_fn(context) or {}
            current_kwargs.update(updates)
            refined = bool(updates)

        result = build_primitives(**current_kwargs)
        history.append(
            RefinementEvent(
                iteration=iteration,
                reasons=reasons,
                sample_budget=int(current_kwargs.get("sample_budget", 25)),
                refined=refined,
            )
        )
        reasons = inspect_primitives(
            result,
            required_primitives=required_primitives,
            confidence_threshold=confidence_threshold,
        )

    return ClosedLoopResult(result=result, converged=not reasons, refinement_history=history)
