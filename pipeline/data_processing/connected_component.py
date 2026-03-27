"""
Connected Component Analysis (Section B.1)

Refines a binary segmentation mask produced by Semantic-SAM by discarding
small spurious fragments whose area falls below a relative threshold τ_area
computed against the object's axis-aligned bounding box.
"""

import cv2
import numpy as np


def refine_mask(
    mask: np.ndarray,
    tau_area: float = 0.01,
) -> np.ndarray:
    """
    Filter connected components of a binary mask.

    Args:
        mask:      Binary mask M ∈ {0,1}^{H×W} (uint8 or bool).
        tau_area:  Relative area threshold τ_area (default 0.01).
                   A component C_k is retained iff
                   |C_k| / |BBox(M)| >= tau_area.

    Returns:
        refined_mask: Binary uint8 mask M_hat = ∪ { C_k | retained }.
    """
    mask = (mask > 0).astype(np.uint8)

    # Bounding box area of the full foreground region
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    if bbox_area == 0:
        return mask

    # 8-connectivity labeling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    refined = np.zeros_like(mask)
    for k in range(1, num_labels):          # label 0 is background
        area = stats[k, cv2.CC_STAT_AREA]
        if area / bbox_area >= tau_area:
            refined[labels == k] = 1

    return refined


def get_components(
    mask: np.ndarray,
    tau_area: float = 0.01,
) -> list[dict]:
    """
    Return retained connected components as a list of dicts with keys:
        'mask'   – binary sub-mask for this component
        'area'   – pixel count
        'bbox'   – (x, y, w, h) bounding box
    """
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components = []
    for k in range(1, num_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area / bbox_area >= tau_area:
            components.append({
                "mask": (labels == k).astype(np.uint8),
                "area": int(area),
                "bbox": (
                    int(stats[k, cv2.CC_STAT_LEFT]),
                    int(stats[k, cv2.CC_STAT_TOP]),
                    int(stats[k, cv2.CC_STAT_WIDTH]),
                    int(stats[k, cv2.CC_STAT_HEIGHT]),
                ),
            })

    # Largest component first
    components.sort(key=lambda c: c["area"], reverse=True)
    return components
