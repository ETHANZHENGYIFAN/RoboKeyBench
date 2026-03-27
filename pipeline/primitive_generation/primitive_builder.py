"""
Primitive Builder – Top-Level Entry Point

Orchestrates the full primitive generation pipeline for a single object:

  1. Refine segmentation mask        (connected_component)
  2. Cluster keypoint candidates     (dbscan_clustering)
  3. Strategically sample candidates (strategic_sampling)
  4. Calibrate principal axes        (axis_calibration)
  5. Build functional axes           (functional_axis)
  6. Handle RJO / FCO if applicable  (rjo / fco)
"""

from __future__ import annotations
import numpy as np
from typing import Literal

from ..data_processing.connected_component import refine_mask
from ..data_processing.dbscan_clustering import cluster_pipeline
from ..data_processing.strategic_sampling import strategic_sample
from ..data_processing.axis_calibration import calibrate_from_masks
from .functional_axis import build_primitives_from_gpt
from .rjo import process_rjo, RJOResult
from .fco import process_fco, FCOResult


ObjectClass = Literal["normal", "rjo", "fco"]


def build_primitives(
    # ── Segmentation masks (top + bottom view) ──────────────────────────
    mask_top:       np.ndarray,
    mask_bot:       np.ndarray,
    depth_top:      np.ndarray,
    depth_bot:      np.ndarray,
    K_top:          np.ndarray,
    K_bot:          np.ndarray,
    extrinsic_top:  np.ndarray,
    extrinsic_bot:  np.ndarray,
    # ── Raw keypoint candidates from multi-view projections ─────────────
    raw_keypoints:  np.ndarray,
    # ── GPT-4o annotation response ───────────────────────────────────────
    gpt_response:   dict,
    # ── Object class ─────────────────────────────────────────────────────
    object_class:   ObjectClass = "normal",
    # ── Optional RJO / FCO arguments ─────────────────────────────────────
    rjo_kwargs:     dict | None = None,
    fco_kwargs:     dict | None = None,
    # ── Hyper-parameters ─────────────────────────────────────────────────
    tau_area:       float = 0.01,
    dbscan_eps:     float = 0.05,
    dbscan_min:     int   = 3,
    sample_budget:  int   = 25,
) -> dict:
    """
    Full primitive generation pipeline.

    Args:
        mask_top / mask_bot:        Binary foreground masks for top/bottom views.
        depth_top / depth_bot:      Corresponding depth maps.
        K_top / K_bot:              Camera intrinsic matrices (3×3).
        extrinsic_top / _bot:       Camera-to-world transforms (4×4).
        raw_keypoints:              (N, 3) raw candidate keypoints.
        gpt_response:               GPT-4o axis annotation response dict.
        object_class:               'normal', 'rjo', or 'fco'.
        rjo_kwargs:                 Extra args forwarded to process_rjo().
        fco_kwargs:                 Extra args forwarded to process_fco().
        tau_area:                   Connected component area threshold.
        dbscan_eps / dbscan_min:    DBSCAN hyperparameters.
        sample_budget:              Strategic sampling target size.

    Returns:
        result dict with keys:
            'coordinate_frame' – {'v_vert', 'v_horiz1', 'v_horiz2', 'c_top', 'c_bot'}
            'keypoints'        – (K, 3) sampled keypoint candidates
            'primitives'       – {'grasp': ..., 'place': ..., 'utility': ...}
            'rjo'              – RJOResult (only when object_class == 'rjo')
            'fco'              – FCOResult (only when object_class == 'fco')
    """
    # ── Step 1: Refine segmentation masks ────────────────────────────────
    refined_top = refine_mask(mask_top, tau_area)
    refined_bot = refine_mask(mask_bot, tau_area)

    # ── Step 2: Cluster raw keypoints ────────────────────────────────────
    clustered_pts, cluster_labels = cluster_pipeline(
        raw_keypoints, eps=dbscan_eps, min_samples=dbscan_min
    )

    # ── Step 3: Strategic sampling ───────────────────────────────────────
    candidates = strategic_sample(clustered_pts, budget=sample_budget)

    # ── Step 4: Calibrate principal axes ─────────────────────────────────
    frame = calibrate_from_masks(
        refined_top, refined_bot,
        depth_top, depth_bot,
        K_top, K_bot,
        extrinsic_top, extrinsic_bot,
    )
    v_vert = frame["v_vert"]

    # ── Step 5: Build functional axes from GPT-4o response ───────────────
    keypoint_lookup = {i: pt for i, pt in enumerate(candidates)}
    primitives = build_primitives_from_gpt(gpt_response, keypoint_lookup, v_vert)

    result: dict = {
        "coordinate_frame": frame,
        "keypoints":        candidates,
        "primitives":       {k: v.to_dict() for k, v in primitives.items()},
    }

    # ── Step 6: Special handling ─────────────────────────────────────────
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
