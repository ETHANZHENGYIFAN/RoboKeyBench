"""
Density-Based DBSCAN Clustering (Section B.2)

Groups raw multi-view keypoint candidates into dense clusters and discards
noise points.  All coordinates are assumed to be normalized to the object's
bounding box so that ε = 0.05 corresponds to ~5 % of the object diagonal.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_keypoints(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply DBSCAN to a set of candidate keypoints.

    Args:
        points:      (N, 3) array of 3-D keypoint coordinates normalized to
                     the object bounding box (values in [0, 1]).
        eps:         ε-neighbourhood radius (default 0.05).
        min_samples: Minimum neighbourhood size m_min to qualify as a core
                     point (default 3).

    Returns:
        filtered_points: (M, 3) array with noise points removed.
        labels:          (M,) integer cluster labels for each retained point.
    """
    if len(points) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    all_labels = db.fit_predict(points)

    # Discard noise (label == -1)
    keep = all_labels != -1
    return points[keep], all_labels[keep]


def normalize_to_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize 3-D points to the unit bounding box [0,1]^3.

    Returns:
        normalized:  (N, 3) normalized coordinates.
        bbox_min:    (3,) lower corner of the original bounding box.
        bbox_scale:  (3,) per-axis extent (for inverse transform).
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_scale = bbox_max - bbox_min
    bbox_scale[bbox_scale == 0] = 1.0          # avoid division by zero
    normalized = (points - bbox_min) / bbox_scale
    return normalized, bbox_min, bbox_scale


def denormalize(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_scale: np.ndarray,
) -> np.ndarray:
    """Inverse of normalize_to_bbox."""
    return points * bbox_scale + bbox_min


def cluster_pipeline(
    raw_points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: normalize → DBSCAN → denormalize.

    Returns:
        filtered_points: (M, 3) in original coordinate space.
        labels:          (M,) cluster labels.
    """
    normalized, bbox_min, bbox_scale = normalize_to_bbox(raw_points)
    filtered_norm, labels = cluster_keypoints(normalized, eps, min_samples)
    filtered_points = denormalize(filtered_norm, bbox_min, bbox_scale)
    return filtered_points, labels
