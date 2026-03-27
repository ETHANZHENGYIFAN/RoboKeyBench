"""
Strategic Sampling for Keypoint Candidate Selection (Section B.3)

Selects a compact, representative subset S ⊂ P from a clustered point cloud
to forward to GPT-4o.  Three priority levels are applied in order:

  Level 1 – Geometric Anchors   : centroid + 6 PCA extremal points (7 total)
  Level 2 – Saliency-Guided     : high-curvature / high-distance points
  Level 3 – Spatial Coverage    : farthest-point sampling (FPS)

Target output size is 20–30 candidates per object.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


# ---------------------------------------------------------------------------
# Level 1 – Geometric Anchors
# ---------------------------------------------------------------------------

def geometric_anchors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the object centroid and the six PCA extremal points.

    Args:
        points: (N, 3) point cloud.

    Returns:
        anchors:  (7, 3) anchor point coordinates.
        indices:  (7,) indices into `points` for the six extremal points
                  (centroid is virtual and has index -1 in the full array).
    """
    centroid = points.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(points)
    axes = pca.components_           # (3, 3) – each row is a principal axis

    extremal_indices = []
    for axis in axes:
        projections = points @ axis
        extremal_indices.append(int(np.argmax(projections)))
        extremal_indices.append(int(np.argmin(projections)))

    extremal_pts = points[extremal_indices]
    anchors = np.vstack([centroid[np.newaxis], extremal_pts])
    return anchors, np.array(extremal_indices)


# ---------------------------------------------------------------------------
# Level 2 – Saliency-Guided Ranking
# ---------------------------------------------------------------------------

def estimate_curvature(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Estimate local surface curvature for each point via PCA on its k-NN.

    Curvature is approximated as the smallest eigenvalue ratio:
        κ(p) = λ_min / (λ_1 + λ_2 + λ_3)

    Args:
        points: (N, 3) point cloud.
        k:      Number of nearest neighbours.

    Returns:
        curvature: (N,) array of curvature estimates in [0, 1].
    """
    tree = KDTree(points)
    curvature = np.zeros(len(points))

    # query k+1 because the point itself is included
    dists, inds = tree.query(points, k=min(k + 1, len(points)))

    for i, neighbours in enumerate(inds):
        nbr = points[neighbours]
        cov = np.cov(nbr.T)
        if cov.ndim < 2:
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))
        total = eigvals.sum()
        if total > 0:
            curvature[i] = eigvals[0] / total

    return curvature


def saliency_scores(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Score each point by combined curvature and distance from centroid.

    score(p) = κ(p) + ||p - c|| / max_dist

    Returns:
        scores: (N,) non-negative saliency scores.
    """
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_dist = distances.max()
    norm_dist = distances / max_dist if max_dist > 0 else distances

    curvature = estimate_curvature(points, k=k)
    scores = curvature + norm_dist
    return scores


# ---------------------------------------------------------------------------
# Level 3 – Farthest-Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(
    points: np.ndarray,
    selected: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Iteratively add the point from `points` that is farthest from any point
    already in `selected`.

    Args:
        points:    (N, 3) candidate pool.
        selected:  (M, 3) already-chosen points.
        n_samples: Number of additional points to select.

    Returns:
        fps_indices: indices into `points` of the newly selected points.
    """
    fps_indices = []
    current = selected.copy()

    for _ in range(min(n_samples, len(points))):
        # Minimum distance from each candidate to the nearest selected point
        diffs = points[:, np.newaxis, :] - current[np.newaxis, :, :]   # (N,M,3)
        min_dists = np.linalg.norm(diffs, axis=2).min(axis=1)          # (N,)

        farthest = int(np.argmax(min_dists))
        fps_indices.append(farthest)
        current = np.vstack([current, points[farthest]])

    return np.array(fps_indices)


# ---------------------------------------------------------------------------
# Full Strategic Sampling Pipeline
# ---------------------------------------------------------------------------

def strategic_sample(
    points: np.ndarray,
    budget: int = 25,
    curvature_k: int = 10,
) -> np.ndarray:
    """
    Select up to `budget` representative keypoint candidates from `points`.

    Args:
        points:      (N, 3) clustered keypoint candidates.
        budget:      Target number of output candidates (default 25, ≈20–30).
        curvature_k: Neighbourhood size for curvature estimation.

    Returns:
        candidates: (K, 3) selected keypoints, K ≤ budget.
    """
    if len(points) <= budget:
        return points.copy()

    # ── Level 1: Geometric Anchors ────────────────────────────────────────
    anchors, anchor_idx = geometric_anchors(points)
    selected_idx = set(anchor_idx.tolist())
    selected_pts = anchors                        # includes virtual centroid

    remaining_budget = budget - len(selected_pts)
    if remaining_budget <= 0:
        return selected_pts[:budget]

    # Candidate pool: all points not yet selected
    mask = np.ones(len(points), dtype=bool)
    mask[list(selected_idx)] = False
    pool_idx = np.where(mask)[0]
    pool_pts = points[pool_idx]

    if len(pool_pts) == 0:
        return selected_pts

    # ── Level 2: Saliency-Guided Ranking ─────────────────────────────────
    scores = saliency_scores(pool_pts, k=curvature_k)
    ranked = pool_idx[np.argsort(-scores)]          # descending score order

    saliency_budget = remaining_budget // 2
    saliency_pick = ranked[:saliency_budget]
    selected_idx.update(saliency_pick.tolist())
    selected_pts = np.vstack([selected_pts, points[saliency_pick]])

    remaining_budget -= len(saliency_pick)
    if remaining_budget <= 0:
        return selected_pts

    # ── Level 3: Farthest-Point Sampling ─────────────────────────────────
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
