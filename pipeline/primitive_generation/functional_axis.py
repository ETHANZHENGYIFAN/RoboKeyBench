"""
Functional Axis Construction from Keypoint Correspondences (Section B.5)

Builds manipulation primitive axes in one of two representations:
  - Standard orientation  : a global coordinate direction e ∈ {±e_x, ±e_y, ±e_z}
  - Keypoint-pair orientation : vector from keypoint p_i towards keypoint p_j

Covers all axis types in the primitive taxonomy:
  a_grasp  – end-effector approach direction for grasping
  a_place  – principal vertical direction (always v_vert)
  a_util   – functional motion direction (e.g., pouring, cutting)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class AxisType(str, Enum):
    GRASP   = "grasp"
    PLACE   = "place"
    UTILITY = "utility"


class AxisRepr(str, Enum):
    STANDARD      = "standard"       # global coordinate direction
    KEYPOINT_PAIR = "keypoint_pair"  # vector from p_i to p_j


@dataclass
class AxisPrimitive:
    """
    A single manipulation axis primitive.

    Attributes:
        axis_type:    One of GRASP, PLACE, UTILITY.
        repr_type:    How the direction is encoded.
        direction:    (3,) unit vector (always populated after resolve()).
        ori_id:       Raw GPT-4o output.
                        [i, j]      for keypoint-pair  ("ori_ID": [i, j])
                        [dx, dy, dz] for standard       ("ori_ID": [0,0,1])
        confidence:   Score s ∈ [0, 1] returned by GPT-4o.
        keypoints:    Keypoint coordinate lookup {id: (3,) array}.
    """
    axis_type:  AxisType
    repr_type:  AxisRepr
    ori_id:     list
    confidence: float = 1.0
    direction:  np.ndarray = field(default=None, repr=False)
    keypoints:  dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def resolve(self) -> "AxisPrimitive":
        """
        Compute the unit direction vector from ori_id and stored keypoints.
        Modifies self.direction in-place and returns self.
        """
        if self.repr_type == AxisRepr.STANDARD:
            vec = np.array(self.ori_id, dtype=float)
        else:
            i, j = self.ori_id
            pi = self.keypoints[i]
            pj = self.keypoints[j]
            vec = pj - pi

        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            raise ValueError(f"Zero-length axis vector for {self.axis_type}.")
        self.direction = vec / norm
        return self

    def to_dict(self) -> dict:
        d = {
            "axis_type":  self.axis_type.value,
            "repr_type":  self.repr_type.value,
            "ori_id":     self.ori_id,
            "confidence": self.confidence,
        }
        if self.direction is not None:
            d["direction"] = self.direction.tolist()
        return d


def parse_gpt_axis(
    raw: dict,
    axis_type: AxisType,
    keypoints: dict[int, np.ndarray],
) -> AxisPrimitive:
    """
    Parse a single axis entry returned by GPT-4o.

    Expected raw format:
        {"ori_ID": [i, j],      "confidence": 0.9}   # keypoint-pair
        {"ori_ID": [0, 0, 1],   "confidence": 0.95}  # standard

    Args:
        raw:       Dict from GPT-4o response.
        axis_type: Semantic role of this axis.
        keypoints: {id: (3,) array} lookup for keypoint-pair resolution.

    Returns:
        Resolved AxisPrimitive.
    """
    ori = raw["ori_ID"]
    confidence = float(raw.get("confidence", 1.0))

    # Distinguish keypoint-pair (two integer indices) from standard direction
    if all(isinstance(v, int) for v in ori) and len(ori) == 2:
        repr_type = AxisRepr.KEYPOINT_PAIR
    else:
        repr_type = AxisRepr.STANDARD

    primitive = AxisPrimitive(
        axis_type=axis_type,
        repr_type=repr_type,
        ori_id=ori,
        confidence=confidence,
        keypoints=keypoints,
    )
    return primitive.resolve()


def build_place_axis(v_vert: np.ndarray) -> AxisPrimitive:
    """
    The placing axis a_place is always the calibrated principal vertical axis.

    Args:
        v_vert: (3,) unit vertical axis from CrossViewAxisCalibration.

    Returns:
        Resolved AxisPrimitive of type PLACE.
    """
    primitive = AxisPrimitive(
        axis_type=AxisType.PLACE,
        repr_type=AxisRepr.STANDARD,
        ori_id=v_vert.tolist(),
        confidence=1.0,
    )
    primitive.direction = v_vert / np.linalg.norm(v_vert)
    return primitive


def build_primitives_from_gpt(
    gpt_response: dict,
    keypoints: dict[int, np.ndarray],
    v_vert: np.ndarray,
) -> dict[str, AxisPrimitive]:
    """
    Build all axis primitives for one object from a GPT-4o annotation response.

    Expected gpt_response keys (all optional except 'grasp'):
        "grasp"   : {"ori_ID": ..., "confidence": ...}
        "utility" : {"ori_ID": ..., "confidence": ...}

    The placing axis is always constructed from v_vert.

    Returns:
        primitives: {"grasp": ..., "place": ..., "utility": ...}
    """
    primitives: dict[str, AxisPrimitive] = {}

    if "grasp" in gpt_response:
        primitives["grasp"] = parse_gpt_axis(
            gpt_response["grasp"], AxisType.GRASP, keypoints
        )

    # Place axis is deterministic — never inferred from GPT-4o
    primitives["place"] = build_place_axis(v_vert)

    if "utility" in gpt_response:
        primitives["utility"] = parse_gpt_axis(
            gpt_response["utility"], AxisType.UTILITY, keypoints
        )

    return primitives
