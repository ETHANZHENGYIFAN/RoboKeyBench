"""
Rotational Joint Object (RJO) Processing (Section B.6.1)

Three-stage pipeline:
  Stage 1 – Joint Region Identification      (GPT-4o + multi-view projection)
  Stage 2 – Rotation Axis Extraction         (PCA + kinematic feasibility check)
  Stage 3 – Angular Limit Point Detection    (extreme-articulation comparison)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage 1 – Joint Region Identification
# ---------------------------------------------------------------------------

def select_joint_vertices(
    vertices: np.ndarray,
    joint_regions: list[dict],
) -> np.ndarray:
    """
    Collect mesh vertices whose projections fall inside the annotated 2-D
    joint regions across multiple views.

    Args:
        vertices:      (V, 3) mesh vertex positions in world space.
        joint_regions: List of per-view dicts, each containing:
                         'projection_fn'  – callable (V,3) → (V,2) pixels
                         'region_mask'    – (H, W) binary mask R_joint

    Returns:
        V_joint: (M, 3) subset of vertices identified as the joint region.
    """
    hit_counts = np.zeros(len(vertices), dtype=int)

    for view in joint_regions:
        proj_fn  = view["projection_fn"]
        region   = view["region_mask"]        # (H, W) binary
        H, W     = region.shape

        pixels = proj_fn(vertices)            # (V, 2) — (col, row)
        cols = np.round(pixels[:, 0]).astype(int)
        rows = np.round(pixels[:, 1]).astype(int)

        in_image = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        in_region = np.zeros(len(vertices), dtype=bool)
        in_region[in_image] = region[rows[in_image], cols[in_image]] > 0

        hit_counts += in_region.astype(int)

    # Retain vertices seen in at least one view
    return vertices[hit_counts > 0]


# ---------------------------------------------------------------------------
# Stage 2 – Rotation Axis Extraction
# ---------------------------------------------------------------------------

def pca_rotation_axis(v_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA to the joint-region vertex set and return the candidate
    rotation axis (leading eigenvector of the empirical covariance).

    Args:
        v_joint: (M, 3) joint-region vertices.

    Returns:
        axis:       (3,) unit rotation axis candidate a_rot.
        eigenvalues: (3,) eigenvalues in ascending order.
    """
    v_bar = v_joint.mean(axis=0)
    cov = np.cov((v_joint - v_bar).T)              # (3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; leading axis = last column
    axis = eigenvectors[:, -1]
    return axis / np.linalg.norm(axis), eigenvalues


def check_kinematic_feasibility(
    axis_candidate: np.ndarray,
    articulation_states: list[np.ndarray],
    consistency_threshold: float = 0.95,
) -> bool:
    """
    Verify that the candidate rotation axis is consistent across multiple
    rendered articulation states by checking that the axis direction remains
    stable (dot product with mean axis > threshold).

    Args:
        axis_candidate:        (3,) candidate axis.
        articulation_states:   List of (M_i, 3) vertex sets for the joint
                               region at different articulation angles.
        consistency_threshold: Minimum |cos θ| for consistency.

    Returns:
        True if the axis passes the kinematic feasibility check.
    """
    axes = [axis_candidate]
    for state_verts in articulation_states:
        a, _ = pca_rotation_axis(state_verts)
        axes.append(a)

    axes = np.array(axes)            # (K, 3)
    mean_axis = axes.mean(axis=0)
    mean_axis /= np.linalg.norm(mean_axis)

    for a in axes:
        if abs(np.dot(a, mean_axis)) < consistency_threshold:
            return False
    return True


@dataclass
class RJOResult:
    rotation_axis:        np.ndarray          # (3,) unit vector
    rotation_center:      np.ndarray          # (3,) mean of joint vertices
    angular_limit_minus:  np.ndarray          # (3,) l⁻ limit point
    angular_limit_plus:   np.ndarray          # (3,) l⁺ limit point
    confidence:           float = 1.0
    refined:              bool  = False       # True if adaptive refinement ran

    def to_dict(self) -> dict:
        return {
            "rotation_axis":       self.rotation_axis.tolist(),
            "rotation_center":     self.rotation_center.tolist(),
            "angular_limit_minus": self.angular_limit_minus.tolist(),
            "angular_limit_plus":  self.angular_limit_plus.tolist(),
            "confidence":          self.confidence,
            "refined":             self.refined,
        }


# ---------------------------------------------------------------------------
# Stage 3 – Angular Limit Point Detection
# ---------------------------------------------------------------------------

def detect_angular_limits(
    v_joint_open:   np.ndarray,
    v_joint_closed: np.ndarray,
    rotation_axis:  np.ndarray,
    rotation_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify angular limit points l⁻ and l⁺ by finding vertices that are
    spatially invariant w.r.t. the rotation axis yet exhibit maximum
    displacement between the two extreme articulation states.

    Strategy:
      1. Compute radial distance to the rotation axis for all vertices in
         both states to identify axis-stable vertices (small Δr).
      2. Among those, find the pair (one from each state) with maximum
         Euclidean displacement.

    Args:
        v_joint_open:    (M, 3) joint vertices at fully-open state.
        v_joint_closed:  (M, 3) joint vertices at fully-closed state (same
                         topology / correspondence assumed).
        rotation_axis:   (3,) unit rotation axis.
        rotation_center: (3,) point on the rotation axis.

    Returns:
        l_minus: (3,) angular limit point at fully-closed state.
        l_plus:  (3,) angular limit point at fully-open state.
    """
    def radial_distance(pts: np.ndarray) -> np.ndarray:
        """Distance of each point to the rotation axis line."""
        d = pts - rotation_center
        proj = (d @ rotation_axis)[:, np.newaxis] * rotation_axis
        return np.linalg.norm(d - proj, axis=1)

    r_open   = radial_distance(v_joint_open)
    r_closed = radial_distance(v_joint_closed)

    # Axis-stable: small change in radial distance
    delta_r = np.abs(r_open - r_closed)
    stable  = delta_r < np.percentile(delta_r, 25)

    # Among axis-stable vertices, find the pair with maximum displacement
    displacements = np.linalg.norm(v_joint_open - v_joint_closed, axis=1)
    displacements[~stable] = -np.inf
    best_idx = int(np.argmax(displacements))

    return v_joint_closed[best_idx], v_joint_open[best_idx]


# ---------------------------------------------------------------------------
# Full RJO Pipeline
# ---------------------------------------------------------------------------

def process_rjo(
    vertices:             np.ndarray,
    joint_regions:        list[dict],
    articulation_states:  list[np.ndarray],
    v_joint_open:         np.ndarray,
    v_joint_closed:       np.ndarray,
    consistency_threshold: float = 0.95,
    confidence_threshold:  float = 0.7,
    gpt_refine_fn=None,
) -> RJOResult:
    """
    Full three-stage RJO processing pipeline.

    Args:
        vertices:              (V, 3) full mesh vertices.
        joint_regions:         Per-view dicts for Stage 1 (see select_joint_vertices).
        articulation_states:   List of (M_i, 3) joint vertex arrays for feasibility check.
        v_joint_open:          (M, 3) joint vertices at fully-open articulation.
        v_joint_closed:        (M, 3) joint vertices at fully-closed articulation.
        consistency_threshold: Kinematic feasibility threshold.
        confidence_threshold:  If GPT-4o confidence < this, trigger adaptive refinement.
        gpt_refine_fn:         Optional callable(vertices) → (axis, confidence) for
                               adaptive re-querying when feasibility check fails.

    Returns:
        RJOResult with all extracted primitives.
    """
    # Stage 1 – Joint Region
    v_joint = select_joint_vertices(vertices, joint_regions)
    if len(v_joint) < 3:
        raise ValueError("Insufficient joint vertices; check region annotations.")

    rotation_center = v_joint.mean(axis=0)

    # Stage 2 – Rotation Axis + Feasibility
    axis, _ = pca_rotation_axis(v_joint)
    feasible = check_kinematic_feasibility(axis, articulation_states,
                                           consistency_threshold)
    refined = False
    confidence = 1.0

    if not feasible and gpt_refine_fn is not None:
        axis, confidence = gpt_refine_fn(v_joint)
        refined = True

    # Stage 3 – Angular Limits
    l_minus, l_plus = detect_angular_limits(
        v_joint_open, v_joint_closed, axis, rotation_center
    )

    return RJOResult(
        rotation_axis=axis,
        rotation_center=rotation_center,
        angular_limit_minus=l_minus,
        angular_limit_plus=l_plus,
        confidence=confidence,
        refined=refined,
    )
