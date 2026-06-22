"""
Functional Control Object (FCO) processing.

FCOs require actuation-point localization, feedback-region identification, and
signed actuation-axis extraction. The implementation follows the manuscript's
affordance-semantic coupling: operational gestures are grounded in surface
normals and feedback regions are selected by deformation response aligned with
the actuation direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("Degenerate vector cannot be normalized.")
    return vec / norm


def select_actuation_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
    actuation_regions: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Select mesh vertices projected inside annotated 2D actuation regions."""

    hit_counts = np.zeros(len(vertices), dtype=int)
    for view in actuation_regions:
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

    selected = hit_counts > 0
    return vertices[selected], normals[selected]


def infer_actuation_direction(n_fco: np.ndarray) -> np.ndarray:
    """Infer the actuation gesture direction from local surface normals."""

    if len(n_fco) == 0:
        raise ValueError("Cannot determine actuation direction from an empty region.")
    return _unit_vector(np.asarray(n_fco, dtype=float).mean(axis=0))


def _sample_pixels(proj_fn, vertices: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = proj_fn(vertices)
    cols = np.clip(np.round(pixels[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, height - 1)
    return rows, cols


def identify_feedback_region(
    vertices: np.ndarray,
    normals: np.ndarray,
    d_act: np.ndarray,
    actuation_center: np.ndarray,
    deformation_views: list[dict],
    spatial_radius: float = 0.1,
) -> np.ndarray:
    """
    Identify feedback vertices coupled to the actuation gesture.

    Vector displacement maps are scored by the positive projection of local
    deformation onto the actuation direction. Scalar deformation maps remain
    supported as a release-compatible fallback, with an additional surface
    normal consistency check to preserve affordance semantics.
    """

    dists = np.linalg.norm(vertices - actuation_center, axis=1)
    in_radius = dists < spatial_radius
    if not np.any(in_radius):
        return np.empty((0, 3))

    d_act = _unit_vector(np.asarray(d_act, dtype=float))
    normals = np.asarray(normals, dtype=float)
    deformation_score = np.zeros(len(vertices), dtype=float)
    has_directional_signal = False

    for view in deformation_views:
        proj_fn = view["projection_fn"]
        if "displacement_vectors" in view:
            disp_vectors = np.asarray(view["displacement_vectors"], dtype=float)
            if disp_vectors.ndim != 3 or disp_vectors.shape[-1] != 3:
                raise ValueError("displacement_vectors must have shape (H, W, 3).")
            height, width = disp_vectors.shape[:2]
            rows, cols = _sample_pixels(proj_fn, vertices, height, width)
            vectors = disp_vectors[rows, cols]
            magnitudes = np.linalg.norm(vectors, axis=1)
            valid = magnitudes > 1e-8
            alignment = np.zeros(len(vertices), dtype=float)
            alignment[valid] = np.maximum(0.0, (vectors[valid] @ d_act) / magnitudes[valid])
            deformation_score += alignment * magnitudes
            has_directional_signal = True
        elif "displacement_map" in view:
            disp_map = np.asarray(view["displacement_map"], dtype=float)
            height, width = disp_map.shape
            rows, cols = _sample_pixels(proj_fn, vertices, height, width)
            deformation_score += disp_map[rows, cols]

    local_scores = deformation_score[in_radius]
    threshold = float(local_scores.mean()) if len(local_scores) else 0.0
    positive_response = deformation_score > 1e-8
    if has_directional_signal:
        return vertices[in_radius & positive_response & (deformation_score >= threshold)]

    normal_alignment = np.abs(normals @ d_act)
    return vertices[
        in_radius
        & positive_response
        & (deformation_score >= threshold)
        & (normal_alignment > 0.25)
    ]


def extract_actuation_axis(v_fco: np.ndarray, d_act: np.ndarray) -> np.ndarray:
    """Extract the PCA axis and resolve its sign by the actuation direction."""

    v_bar = v_fco.mean(axis=0)
    cov = np.cov((v_fco - v_bar).T)
    _, eigenvectors = np.linalg.eigh(cov)
    e1 = eigenvectors[:, -1]
    sign = np.sign(np.dot(e1, d_act))
    if sign == 0:
        sign = 1.0
    return _unit_vector(sign * e1)


@dataclass
class FCOResult:
    actuation_axis: np.ndarray
    actuation_center: np.ndarray
    actuation_dir: np.ndarray
    feedback_vertices: np.ndarray
    confidence: float = 1.0
    refined: bool = False

    def to_dict(self) -> dict:
        return {
            "actuation_axis": self.actuation_axis.tolist(),
            "actuation_center": self.actuation_center.tolist(),
            "actuation_dir": self.actuation_dir.tolist(),
            "feedback_vertices": self.feedback_vertices.tolist(),
            "confidence": self.confidence,
            "refined": self.refined,
        }


def process_fco(
    vertices: np.ndarray,
    normals: np.ndarray,
    actuation_regions: list[dict],
    deformation_views: list[dict],
    confidence_threshold: float = 0.7,
    spatial_radius: float = 0.1,
    max_refine_iter: int = 2,
    gpt_refine_fn=None,
) -> FCOResult:
    """Run the full three-stage FCO processing pipeline."""

    v_fco, n_fco = select_actuation_vertices(vertices, normals, actuation_regions)
    if len(v_fco) < 3:
        raise ValueError("Insufficient actuation vertices; check region annotations.")

    actuation_center = v_fco.mean(axis=0)
    d_act = infer_actuation_direction(n_fco)
    confidence = 1.0
    refined = False

    if gpt_refine_fn is not None and max_refine_iter > 0:
        for _ in range(max_refine_iter):
            candidate_dir, confidence = gpt_refine_fn(v_fco)
            d_act = _unit_vector(np.asarray(candidate_dir, dtype=float))
            refined = True
            if confidence >= confidence_threshold:
                break

    v_feedback = identify_feedback_region(
        vertices,
        normals,
        d_act,
        actuation_center,
        deformation_views,
        spatial_radius,
    )
    a_fco = extract_actuation_axis(v_fco, d_act)

    return FCOResult(
        actuation_axis=a_fco,
        actuation_center=actuation_center,
        actuation_dir=d_act,
        feedback_vertices=v_feedback,
        confidence=confidence,
        refined=refined,
    )
