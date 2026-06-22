"""
Strategic sampling for keypoint candidate selection.

The sampler selects a compact representative subset from clustered keypoint
candidates before semantic alignment. Three priority levels are applied:

1. Geometric anchors: centroid and six PCA extremal points.
2. Saliency-guided ranking: high-curvature and peripheral points.
3. Spatial coverage: farthest-point sampling.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def geometric_anchors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the object centroid and six PCA extremal points.

    Args:
        points: (N, 3) point cloud.

    Returns:
        anchors: seven anchor coordinates. The first row is the centroid.
        indices: indices of the six extremal points in the input array.
    """
    centroid = points.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(points)
    axes = pca.components_

    extremal_indices = []
    for axis in axes:
        projections = points @ axis
        extremal_indices.append(int(np.argmax(projections)))
        extremal_indices.append(int(np.argmin(projections)))

    extremal_pts = points[extremal_indices]
    anchors = np.vstack([centroid[np.newaxis], extremal_pts])
    return anchors, np.array(extremal_indices)


def estimate_curvature(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Estimate local surface curvature for each point via PCA on k-nearest
    neighbors.

    Curvature is approximated as the ratio between the smallest local
    covariance eigenvalue and the sum of all local eigenvalues.
    """
    tree = KDTree(points)
    curvature = np.zeros(len(points))

    _, inds = tree.query(points, k=min(k + 1, len(points)))

    for idx, neighbours in enumerate(inds):
        nbr = points[neighbours]
        cov = np.cov(nbr.T)
        if cov.ndim < 2:
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))
        total = eigvals.sum()
        if total > 0:
            curvature[idx] = eigvals[0] / total

    return curvature


def saliency_scores(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Score each point by combined curvature and distance from centroid.

    The score favors locally distinctive and spatially peripheral points.
    """
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_dist = distances.max()
    norm_dist = distances / max_dist if max_dist > 0 else distances

    curvature = estimate_curvature(points, k=k)
    return curvature + norm_dist


def farthest_point_sampling(
    points: np.ndarray,
    selected: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Iteratively add the point farthest from the current selected set.

    Args:
        points: candidate pool.
        selected: already selected points.
        n_samples: number of additional points to select.

    Returns:
        Indices into points of the newly selected samples.
    """
    fps_indices = []
    current = selected.copy()

    for _ in range(min(n_samples, len(points))):
        diffs = points[:, np.newaxis, :] - current[np.newaxis, :, :]
        min_dists = np.linalg.norm(diffs, axis=2).min(axis=1)

        farthest = int(np.argmax(min_dists))
        fps_indices.append(farthest)
        current = np.vstack([current, points[farthest]])

    return np.array(fps_indices)


def strategic_sample(
    points: np.ndarray,
    budget: int = 25,
    curvature_k: int = 10,
) -> np.ndarray:
    """
    Select up to budget representative keypoint candidates from points.

    Args:
        points: clustered keypoint candidates.
        budget: target number of output candidates.
        curvature_k: neighborhood size for curvature estimation.

    Returns:
        Selected keypoint candidates.
    """
    if len(points) <= budget:
        return points.copy()

    anchors, anchor_idx = geometric_anchors(points)
    selected_idx = set(anchor_idx.tolist())
    selected_pts = anchors

    remaining_budget = budget - len(selected_pts)
    if remaining_budget <= 0:
        return selected_pts[:budget]

    mask = np.ones(len(points), dtype=bool)
    mask[list(selected_idx)] = False
    pool_idx = np.where(mask)[0]
    pool_pts = points[pool_idx]

    if len(pool_pts) == 0:
        return selected_pts

    scores = saliency_scores(pool_pts, k=curvature_k)
    ranked = pool_idx[np.argsort(-scores)]

    saliency_budget = remaining_budget // 2
    saliency_pick = ranked[:saliency_budget]
    selected_idx.update(saliency_pick.tolist())
    selected_pts = np.vstack([selected_pts, points[saliency_pick]])

    remaining_budget -= len(saliency_pick)
    if remaining_budget <= 0:
        return selected_pts

    mask2 = np.ones(len(points), dtype=bool)
    mask2[list(selected_idx)] = False
    pool2_idx = np.where(mask2)[0]
    pool2_pts = points[pool2_idx]

    if len(pool2_pts) == 0:
        return selected_pts

    fps_local = farthest_point_sampling(pool2_pts, selected_pts, remaining_budget)
    fps_global = pool2_idx[fps_local]
    selected_pts = np.vstack([selected_pts, points[fps_global]])

    return selected_pts
