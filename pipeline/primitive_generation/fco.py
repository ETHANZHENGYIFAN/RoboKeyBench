"""
Functional Control Object (FCO) Processing (Section B.6.2)

Three-stage pipeline:
  Stage 1 – Actuation Point Identification  (GPT-4o + surface normal grounding)
  Stage 2 – Feedback Region Identification  (affordance-semantic coupling)
  Stage 3 – Actuation Axis Extraction       (PCA + sign resolution via d_act)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage 1 – Actuation Point Identification
# ---------------------------------------------------------------------------

def select_actuation_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
    actuation_regions: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect mesh vertices whose projections fall within the annotated 2-D
    actuation regions across multiple views.

    Args:
        vertices:          (V, 3) mesh vertices.
        normals:           (V, 3) per-vertex surface normals (unit length).
        actuation_regions: List of per-view dicts, each containing:
                             'projection_fn' – callable (V,3) → (V,2) pixels
                             'region_mask'   – (H, W) binary mask R_act

    Returns:
        v_fco:   (M, 3) selected actuation vertices.
        n_fco:   (M, 3) corresponding surface normals.
    """
    hit_counts = np.zeros(len(vertices), dtype=int)

    for view in actuation_regions:
        proj_fn = view["projection_fn"]
        region  = view["region_mask"]
        H, W    = region.shape

        pixels = proj_fn(vertices)
        cols = np.round(pixels[:, 0]).astype(int)
        rows = np.round(pixels[:, 1]).astype(int)

        in_image  = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        in_region = np.zeros(len(vertices), dtype=bool)
        in_region[in_image] = region[rows[in_image], cols[in_image]] > 0

        hit_counts += in_region.astype(int)

    selected = hit_counts > 0
    return vertices[selected], normals[selected]


def infer_actuation_direction(n_fco: np.ndarray) -> np.ndarray:
    """
    Derive the actuation direction d_act as the mean surface normal of the
    actuation region, constrained to be spatially coincident with the local
    surface geometry.

    Args:
        n_fco: (M, 3) surface normals of selected actuation vertices.

    Returns:
        d_act: (3,) unit actuation direction.
    """
    mean_normal = n_fco.mean(axis=0)
    norm = np.linalg.norm(mean_normal)
    if norm < 1e-8:
        raise ValueError("Cannot determine actuation direction: "
                         "degenerate normal distribution.")
    return mean_normal / norm


# ---------------------------------------------------------------------------
# Stage 2 – Feedback Region Identification
# ---------------------------------------------------------------------------

def identify_feedback_region(
    vertices: np.ndarray,
    normals: np.ndarray,
    d_act: np.ndarray,
    actuation_center: np.ndarray,
    deformation_views: list[dict],
    spatial_radius: float = 0.1,
) -> np.ndarray:
    """
    Identify the feedback region: vertices that exhibit deformation signatures
    in response to the actuation gesture (button depression, slider travel, etc.)

    Strategy:
      1. Restrict candidates to a spatial neighbourhood of the actuation center.
      2. Among candidates, retain those whose projected displacement across
         rendered articulation views is aligned with d_act (affordance coupling).

    Args:
        vertices:          (V, 3) full mesh.
        normals:           (V, 3) per-vertex normals.
        d_act:             (3,) actuation direction.
        actuation_center:  (3,) mean of actuation vertices.
        deformation_views: List of per-view dicts with:
                             'projection_fn'     – (V,3) → (V,2) projection
                             'displacement_map'  – (H,W) scalar deformation map
        spatial_radius:    Normalized radius around actuation_center.

    Returns:
        v_feedback: (K, 3) feedback region vertices.
    """
    # Spatial neighbourhood filter
    dists = np.linalg.norm(vertices - actuation_center, axis=1)
    in_radius = dists < spatial_radius

    # Affordance-semantic coupling: deformation aligned with d_act
    deformation_score = np.zeros(len(vertices))
    for view in deformation_views:
        proj_fn   = view["projection_fn"]
        disp_map  = view["displacement_map"]
        H, W = disp_map.shape

        pixels = proj_fn(vertices)
        cols = np.clip(np.round(pixels[:, 0]).astype(int), 0, W - 1)
        rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, H - 1)
        deformation_score += disp_map[rows, cols]

    combined = in_radius & (deformation_score > deformation_score[in_radius].mean())
    return vertices[combined]


# ---------------------------------------------------------------------------
# Stage 3 – Actuation Axis Extraction
# ---------------------------------------------------------------------------

def extract_actuation_axis(
    v_fco: np.ndarray,
    d_act: np.ndarray,
) -> np.ndarray:
    """
    Apply PCA to V_fco and resolve the sign of the leading eigenvector by
    alignment with d_act.

        a_fco = sgn(e_1(Σ_fco) · d_act) · e_1(Σ_fco)

    Args:
        v_fco: (M, 3) actuation region vertices.
        d_act: (3,) actuation direction for sign disambiguation.

    Returns:
        a_fco: (3,) signed unit actuation axis.
    """
    v_bar = v_fco.mean(axis=0)
    cov   = np.cov((v_fco - v_bar).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    e1 = eigenvectors[:, -1]                       # leading eigenvector

    sign = np.sign(np.dot(e1, d_act))
    if sign == 0:
        sign = 1.0
    return sign * e1 / np.linalg.norm(e1)


# ---------------------------------------------------------------------------
# Full FCO Result
# ---------------------------------------------------------------------------

@dataclass
class FCOResult:
    actuation_axis:    np.ndarray          # (3,) signed unit axis a_fco
    actuation_center:  np.ndarray          # (3,) centroid of actuation region
    actuation_dir:     np.ndarray          # (3,) d_act from surface normals
    feedback_vertices: np.ndarray          # (K, 3) feedback region
    confidence:        float = 1.0
    refined:           bool  = False

    def to_dict(self) -> dict:
        return {
            "actuation_axis":   self.actuation_axis.tolist(),
            "actuation_center": self.actuation_center.tolist(),
            "actuation_dir":    self.actuation_dir.tolist(),
            "confidence":       self.confidence,
            "refined":          self.refined,
        }


def process_fco(
    vertices:            np.ndarray,
    normals:             np.ndarray,
    actuation_regions:   list[dict],
    deformation_views:   list[dict],
    confidence_threshold: float = 0.7,
    spatial_radius:      float = 0.1,
    gpt_refine_fn=None,
) -> FCOResult:
    """
    Full three-stage FCO processing pipeline.

    Args:
        vertices:             (V, 3) mesh vertices.
        normals:              (V, 3) per-vertex surface normals.
        actuation_regions:    Per-view dicts for Stage 1.
        deformation_views:    Per-view dicts for Stage 2.
        confidence_threshold: Trigger adaptive resampling below this value.
        spatial_radius:       Feedback region search radius (normalized).
        gpt_refine_fn:        Optional callable(v_fco) → (d_act, confidence)
                              for adaptive refinement.

    Returns:
        FCOResult with all extracted primitives.
    """
    # Stage 1 – Actuation Point
    v_fco, n_fco = select_actuation_vertices(vertices, normals, actuation_regions)
    if len(v_fco) < 3:
        raise ValueError("Insufficient actuation vertices; check region annotations.")

    actuation_center = v_fco.mean(axis=0)
    d_act = infer_actuation_direction(n_fco)
    confidence = 1.0
    refined = False

    # Adaptive refinement if GPT-4o confidence is low
    if gpt_refine_fn is not None:
        d_act_gpt, confidence = gpt_refine_fn(v_fco)
        if confidence < confidence_threshold:
            d_act_gpt, confidence = gpt_refine_fn(v_fco)   # re-query with extra views
            refined = True
        d_act = d_act_gpt

    # Stage 2 – Feedback Region
    v_feedback = identify_feedback_region(
        vertices, normals, d_act, actuation_center,
        deformation_views, spatial_radius
    )

    # Stage 3 – Actuation Axis
    a_fco = extract_actuation_axis(v_fco, d_act)

    return FCOResult(
        actuation_axis=a_fco,
        actuation_center=actuation_center,
        actuation_dir=d_act,
        feedback_vertices=v_feedback,
        confidence=confidence,
        refined=refined,
    )
