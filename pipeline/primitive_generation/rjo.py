"""
Rotational Joint Object (RJO) processing.

The RJO pipeline extracts a rotation axis, verifies that it is a persistent
linear feature orthogonal to the object's motion plane across rendered
articulation states, and detects angular limit points from extreme states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("Degenerate vector cannot be normalized.")
    return vec / norm


def estimate_motion_plane_normal(
    motion_states: list[np.ndarray],
    min_displacement: float = 1e-6,
) -> np.ndarray | None:
    """Estimate the motion-plane normal from multi-state vertex trajectories."""

    states = [np.asarray(s, dtype=float) for s in motion_states if len(s) > 0]
    states = [s for s in states if s.ndim == 2 and s.shape[1] == 3]
    if len(states) < 2:
        return None

    motion_vectors: list[np.ndarray] = []
    same_shape = all(s.shape == states[0].shape for s in states)
    if same_shape:
        base = states[0]
        for state in states[1:]:
            displacement = state - base
            moving = np.linalg.norm(displacement, axis=1) > min_displacement
            if np.any(moving):
                motion_vectors.append(displacement[moving])
        if len(states) >= 3:
            trajectory = np.stack(states, axis=0)
            centered = trajectory - trajectory.mean(axis=0, keepdims=True)
            centered = centered.reshape(-1, 3)
            moving = np.linalg.norm(centered, axis=1) > min_displacement
            if np.any(moving):
                motion_vectors.append(centered[moving])
    else:
        centers = np.asarray([s.mean(axis=0) for s in states])
        displacement = centers[1:] - centers[0]
        moving = np.linalg.norm(displacement, axis=1) > min_displacement
        if np.any(moving):
            motion_vectors.append(displacement[moving])
        centered = centers - centers.mean(axis=0, keepdims=True)
        moving = np.linalg.norm(centered, axis=1) > min_displacement
        if np.any(moving):
            motion_vectors.append(centered[moving])

    if not motion_vectors:
        return None
    vectors = np.concatenate(motion_vectors, axis=0)
    if vectors.shape[0] < 2:
        return None

    cov = vectors.T @ vectors / vectors.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.count_nonzero(eigenvalues > min_displacement**2) < 2:
        return None
    return _unit_vector(eigenvectors[:, 0])


def select_joint_vertices(vertices: np.ndarray, joint_regions: list[dict]) -> np.ndarray:
    """Collect mesh vertices projected inside annotated 2D joint regions."""

    hit_counts = np.zeros(len(vertices), dtype=int)
    for view in joint_regions:
        proj_fn = view["projection_fn"]
        region = view["region_mask"]
        height, width = region.shape
        pixels = proj_fn(vertices)
        cols = np.round(pixels[:, 0]).astype(int)
        rows = np.round(pixels[:, 1]).astype(int)
        in_image = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        in_region = np.zeros(len(vertices), dtype=bool)
        in_region[in_image] = region[rows[in_image], cols[in_image]] > 0
        hit_counts += in_region.astype(int)
    return vertices[hit_counts > 0]


def pca_rotation_axis(v_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading PCA direction of the joint-region vertices."""

    v_bar = v_joint.mean(axis=0)
    cov = np.cov((v_joint - v_bar).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, -1]
    return _unit_vector(axis), eigenvalues


def check_kinematic_feasibility(
    axis_candidate: np.ndarray,
    joint_axis_states: list[np.ndarray],
    motion_states: list[np.ndarray] | None = None,
    consistency_threshold: float = 0.95,
    orthogonality_threshold: float = 0.90,
) -> bool:
    """
    Verify persistent-axis consistency and motion-plane orthogonality.

    Axis signs from PCA are first aligned to the candidate axis before the
    consistency average is computed, avoiding false failures caused by PCA's
    arbitrary sign convention.
    """

    if len(joint_axis_states) < 2:
        return False
    axis_candidate = _unit_vector(np.asarray(axis_candidate, dtype=float))

    aligned_axes = [axis_candidate]
    for state_verts in joint_axis_states:
        if len(state_verts) < 3:
            return False
        axis, _ = pca_rotation_axis(np.asarray(state_verts, dtype=float))
        if np.dot(axis, axis_candidate) < 0:
            axis = -axis
        aligned_axes.append(axis)

    mean_axis = _unit_vector(np.asarray(aligned_axes).mean(axis=0))
    for axis in aligned_axes:
        if np.dot(_unit_vector(axis), mean_axis) < consistency_threshold:
            return False

    plane_states = motion_states if motion_states is not None else joint_axis_states
    motion_plane_normal = estimate_motion_plane_normal(plane_states)
    if motion_plane_normal is None:
        return False

    return abs(float(np.dot(axis_candidate, motion_plane_normal))) >= orthogonality_threshold


@dataclass
class RJOResult:
    rotation_axis: np.ndarray
    rotation_center: np.ndarray
    angular_limit_minus: np.ndarray
    angular_limit_plus: np.ndarray
    confidence: float = 1.0
    refined: bool = False

    def to_dict(self) -> dict:
        return {
            "rotation_axis": self.rotation_axis.tolist(),
            "rotation_center": self.rotation_center.tolist(),
            "angular_limit_minus": self.angular_limit_minus.tolist(),
            "angular_limit_plus": self.angular_limit_plus.tolist(),
            "confidence": self.confidence,
            "refined": self.refined,
        }


def detect_angular_limits(
    v_joint_open: np.ndarray,
    v_joint_closed: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect angular limit points by comparing extreme articulation states."""

    rotation_axis = _unit_vector(rotation_axis)

    def radial_distance(points: np.ndarray) -> np.ndarray:
        delta = points - rotation_center
        proj = (delta @ rotation_axis)[:, np.newaxis] * rotation_axis
        return np.linalg.norm(delta - proj, axis=1)

    r_open = radial_distance(v_joint_open)
    r_closed = radial_distance(v_joint_closed)
    delta_r = np.abs(r_open - r_closed)
    stable = delta_r < np.percentile(delta_r, 25)

    displacements = np.linalg.norm(v_joint_open - v_joint_closed, axis=1)
    displacements[~stable] = -np.inf
    best_idx = int(np.argmax(displacements))
    return v_joint_closed[best_idx], v_joint_open[best_idx]


def process_rjo(
    vertices: np.ndarray,
    joint_regions: list[dict],
    articulation_states: list[np.ndarray],
    v_joint_open: np.ndarray,
    v_joint_closed: np.ndarray,
    motion_states: list[np.ndarray] | None = None,
    consistency_threshold: float = 0.95,
    orthogonality_threshold: float = 0.90,
    confidence_threshold: float = 0.7,
    max_refine_iter: int = 2,
    gpt_refine_fn=None,
) -> RJOResult:
    """Run the full RJO processing pipeline."""

    v_joint = select_joint_vertices(vertices, joint_regions)
    if len(v_joint) < 3:
        raise ValueError("Insufficient joint vertices; check region annotations.")

    rotation_center = v_joint.mean(axis=0)
    axis, _ = pca_rotation_axis(v_joint)
    joint_axis_states = list(articulation_states or [])
    if len(joint_axis_states) < 2:
        joint_axis_states = [v_joint_closed, v_joint_open]
    plane_motion_states = motion_states if motion_states is not None else [v_joint_closed, v_joint_open]

    feasible = check_kinematic_feasibility(
        axis,
        joint_axis_states,
        motion_states=plane_motion_states,
        consistency_threshold=consistency_threshold,
        orthogonality_threshold=orthogonality_threshold,
    )
    refined = False
    confidence = 1.0

    refine_iter = 0
    while (not feasible or confidence < confidence_threshold) and gpt_refine_fn is not None:
        if refine_iter >= max_refine_iter:
            break
        axis, confidence = gpt_refine_fn(v_joint)
        axis = _unit_vector(np.asarray(axis, dtype=float))
        feasible = check_kinematic_feasibility(
            axis,
            joint_axis_states,
            motion_states=plane_motion_states,
            consistency_threshold=consistency_threshold,
            orthogonality_threshold=orthogonality_threshold,
        )
        refined = True
        refine_iter += 1

    l_minus, l_plus = detect_angular_limits(
        v_joint_open,
        v_joint_closed,
        axis,
        rotation_center,
    )

    return RJOResult(
        rotation_axis=axis,
        rotation_center=rotation_center,
        angular_limit_minus=l_minus,
        angular_limit_plus=l_plus,
        confidence=confidence,
        refined=refined,
    )
