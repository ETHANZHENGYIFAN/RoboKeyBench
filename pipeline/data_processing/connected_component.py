"""
Connected component analysis for mask refinement.

Semantic segmentation masks may contain small fragments unrelated to the
foreground object. This module keeps connected components whose relative area
exceeds a threshold measured against the object's axis-aligned bounding box.
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
        mask: Binary mask with shape (H, W), stored as uint8 or bool.
        tau_area: Relative area threshold. A component C_k is retained when
            area(C_k) / area(BBox(mask)) >= tau_area.

    Returns:
        Binary uint8 mask containing all retained components.
    """
    mask = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    if bbox_area == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    refined = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area / bbox_area >= tau_area:
            refined[labels == label_idx] = 1

    return refined


def get_components(
    mask: np.ndarray,
    tau_area: float = 0.01,
) -> list[dict]:
    """
    Return retained connected components.

    Each component is represented as a dict with:
        mask: binary sub-mask for this component.
        area: pixel count.
        bbox: (x, y, w, h) bounding box.
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
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area / bbox_area >= tau_area:
            components.append(
                {
                    "mask": (labels == label_idx).astype(np.uint8),
                    "area": int(area),
                    "bbox": (
                        int(stats[label_idx, cv2.CC_STAT_LEFT]),
                        int(stats[label_idx, cv2.CC_STAT_TOP]),
                        int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                        int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
                    ),
                }
            )

    components.sort(key=lambda component: component["area"], reverse=True)
    return components
