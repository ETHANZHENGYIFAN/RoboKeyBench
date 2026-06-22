"""
Primitive Builder - top-level PASG primitive generation entry point.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..data_processing.axis_calibration import calibrate_from_masks
from ..data_processing.connected_component import refine_mask
from ..data_processing.dbscan_clustering import cluster_pipeline
from ..data_processing.strategic_sampling import strategic_sample
from .fco import FCOResult, process_fco
from .functional_axis import build_point_primitives_from_gpt, build_primitives_from_gpt
from .rjo import RJOResult, process_rjo


ObjectClass = Literal["normal", "rjo", "fco"]


def _keypoint_lookup(candidates: np.ndarray) -> dict[int, np.ndarray]:
    """Expose one-based IDs matching the point labels used in annotations."""

    return {i + 1: pt for i, pt in enumerate(candidates)}


def build_primitives(
    mask_top: np.ndarray,
    mask_bot: np.ndarray,
    depth_top: np.ndarray,
    depth_bot: np.ndarray,
    K_top: np.ndarray,
    K_bot: np.ndarray,
    extrinsic_top: np.ndarray,
    extrinsic_bot: np.ndarray,
    raw_keypoints: np.ndarray,
    gpt_response: dict,
    object_class: ObjectClass = "normal",
    rjo_kwargs: dict | None = None,
    fco_kwargs: dict | None = None,
    tau_area: float = 0.01,
    dbscan_eps: float = 0.05,
    dbscan_min: int = 3,
    sample_budget: int = 25,
) -> dict:
    """
    Run the deterministic PASG primitive builder.

    The returned payload contains point primitives, axis primitives, and a
    combined operation-primitive view. The historical ``primitives`` key is
    retained as a primary-axis alias for older evaluation helpers.
    """

    refined_top = refine_mask(mask_top, tau_area)
    refined_bot = refine_mask(mask_bot, tau_area)

    clustered_pts, _ = cluster_pipeline(
        raw_keypoints, eps=dbscan_eps, min_samples=dbscan_min
    )
    candidates = strategic_sample(clustered_pts, budget=sample_budget)

    frame = calibrate_from_masks(
        refined_top,
        refined_bot,
        depth_top,
        depth_bot,
        K_top,
        K_bot,
        extrinsic_top,
        extrinsic_bot,
    )
    v_vert = frame["v_vert"]

    keypoints = _keypoint_lookup(candidates)
    point_primitives = build_point_primitives_from_gpt(gpt_response, keypoints)
    axis_primitives = build_primitives_from_gpt(gpt_response, keypoints, v_vert)

    point_payload = {
        name: [primitive.to_dict() for primitive in primitives]
        for name, primitives in point_primitives.items()
    }
    axis_payload = {
        name: [primitive.to_dict() for primitive in primitives]
        for name, primitives in axis_primitives.items()
    }
    primary_axis_payload = {
        name: primitives[0]
        for name, primitives in axis_payload.items()
        if primitives
    }

    result: dict = {
        "coordinate_frame": frame,
        "keypoints": candidates,
        "point_primitives": point_payload,
        "axis_primitives": axis_payload,
        "operation_primitives": {
            "points": point_payload,
            "axes": axis_payload,
        },
        "primitives": primary_axis_payload,
    }

    if object_class == "rjo":
        if rjo_kwargs is None:
            raise ValueError("rjo_kwargs must be provided for RJO objects.")
        rjo_result: RJOResult = process_rjo(**rjo_kwargs)
        result["rjo"] = rjo_result.to_dict()
    elif object_class == "fco":
        if fco_kwargs is None:
            raise ValueError("fco_kwargs must be provided for FCO objects.")
        fco_result: FCOResult = process_fco(**fco_kwargs)
        result["fco"] = fco_result.to_dict()

    return result


def build_primitives_closed_loop(**kwargs) -> dict:
    """Run the explicit PASG closed-loop controller around ``build_primitives``."""

    from .closed_loop_controller import run_closed_loop_refinement

    controller_keys = {
        "refine_fn",
        "required_primitives",
        "confidence_threshold",
        "max_iterations",
        "sample_budget_step",
    }
    controller_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs.keys()) if key in controller_keys
    }
    object_class = kwargs.get("object_class", "normal")
    return run_closed_loop_refinement(
        build_kwargs=kwargs,
        object_class=object_class,
        **controller_kwargs,
    ).to_dict()
