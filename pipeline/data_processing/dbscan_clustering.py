"""
Density-based DBSCAN clustering for raw keypoint candidates.

The input coordinates are normalized to the object bounding box before
clustering, making the default epsilon value approximately scale-invariant
across objects.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_keypoints(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply DBSCAN to candidate keypoints.

    Args:
        points: (N, 3) normalized 3D keypoint coordinates.
        eps: Neighborhood radius in normalized coordinates.
        min_samples: Minimum neighborhood size for a core point.

    Returns:
        filtered_points: points with DBSCAN noise removed.
        labels: cluster labels for the retained points.
    """
    if len(points) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    all_labels = db.fit_predict(points)

    keep = all_labels != -1
    return points[keep], all_labels[keep]


def normalize_to_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize 3D points to the unit bounding box [0, 1]^3.

    Returns:
        normalized: normalized coordinates.
        bbox_min: lower corner of the original bounding box.
        bbox_scale: per-axis extent for inverse transformation.
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_scale = bbox_max - bbox_min
    bbox_scale[bbox_scale == 0] = 1.0
    normalized = (points - bbox_min) / bbox_scale
    return normalized, bbox_min, bbox_scale


def denormalize(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_scale: np.ndarray,
) -> np.ndarray:
    """Invert normalize_to_bbox."""
    return points * bbox_scale + bbox_min


def cluster_pipeline(
    raw_points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize raw points, apply DBSCAN, and map retained points back.

    Returns:
        filtered_points: retained points in the original coordinate space.
        labels: DBSCAN cluster labels for retained points.
    """
    normalized, bbox_min, bbox_scale = normalize_to_bbox(raw_points)
    filtered_norm, labels = cluster_keypoints(normalized, eps, min_samples)
    filtered_points = denormalize(filtered_norm, bbox_min, bbox_scale)
    return filtered_points, labels
