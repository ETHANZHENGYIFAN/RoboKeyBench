"""
Functional point and axis primitive construction.

This module converts GPT-4o semantic alignment records into the operation
primitive representation described in the manuscript. Point primitives and
axis primitives are both first-class outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AxisType(str, Enum):
    GRASP = "grasp"
    PLACE = "place"
    UTILITY = "utility"


class PointType(str, Enum):
    GRASP = "grasp"
    PLACE = "place"
    UTILITY = "utility"


class AxisRepr(str, Enum):
    STANDARD = "standard"
    KEYPOINT_PAIR = "keypoint_pair"


POINT_KEY_ALIASES = {
    PointType.GRASP: ("grasp", "grasping", "grasp_point", "Grasp", "Grasping"),
    PointType.PLACE: ("place", "placing", "place_point", "Place", "Placing"),
    PointType.UTILITY: ("utility", "util", "utility_point", "Utility"),
}

AXIS_KEY_ALIASES = {
    AxisType.GRASP: ("grasp", "grasping", "grasp_axis", "Grasp", "Grasping"),
    AxisType.PLACE: ("place", "placing", "place_axis", "Place", "Placing"),
    AxisType.UTILITY: ("utility", "util", "utility_axis", "Utility"),
}


@dataclass
class PointPrimitive:
    """A functional point primitive with geometry and semantic metadata."""

    point_type: PointType
    pos_id: int
    confidence: float = 1.0
    position: np.ndarray | None = field(default=None, repr=False)
    stage: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        payload = {
            "point_type": self.point_type.value,
            "pos_id": self.pos_id,
            "confidence": self.confidence,
        }
        if self.position is not None:
            payload["position"] = self.position.tolist()
        if self.stage:
            payload["stage"] = self.stage
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class AxisPrimitive:
    """A functional axis primitive in standard or keypoint-pair form."""

    axis_type: AxisType
    repr_type: AxisRepr
    ori_id: list
    confidence: float = 1.0
    direction: np.ndarray | None = field(default=None, repr=False)
    keypoints: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def resolve(self) -> "AxisPrimitive":
        if self.repr_type == AxisRepr.STANDARD:
            vec = np.asarray(self.ori_id, dtype=float)
        else:
            i, j = self.ori_id
            if i not in self.keypoints and i + 1 in self.keypoints:
                i = i + 1
            if j not in self.keypoints and j + 1 in self.keypoints:
                j = j + 1
            if i not in self.keypoints or j not in self.keypoints:
                raise KeyError(f"Axis keypoint pair {self.ori_id} is not in the keypoint lookup.")
            vec = self.keypoints[j] - self.keypoints[i]

        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            raise ValueError(f"Zero-length axis vector for {self.axis_type}.")
        self.direction = vec / norm
        return self

    def to_dict(self) -> dict:
        payload = {
            "axis_type": self.axis_type.value,
            "repr_type": self.repr_type.value,
            "ori_id": self.ori_id,
            "confidence": self.confidence,
        }
        if self.direction is not None:
            payload["direction"] = self.direction.tolist()
        return payload


def _as_records(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        records = [value]
        for key in ("candidates", "points", "axes", "orientations"):
            nested = value.get(key)
            if isinstance(nested, list):
                records.extend(item for item in nested if isinstance(item, dict))
        return records
    return []


def _records_for_aliases(response: dict, aliases: tuple[str, ...]) -> list[dict]:
    records: list[dict] = []
    for key in aliases:
        records.extend(_as_records(response.get(key)))
    return records


def _parse_id(record: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = record.get(key)
        if value in (None, "None", "Error"):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _parse_confidence(record: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key not in record:
            continue
        try:
            return float(record[key])
        except (TypeError, ValueError):
            pass
    return 1.0


def build_point_primitives_from_gpt(
    gpt_response: dict,
    keypoints: dict[int, np.ndarray],
) -> dict[str, list[PointPrimitive]]:
    """Build all available point primitives from GPT-4o alignment records."""

    primitives: dict[str, list[PointPrimitive]] = {}
    for point_type in PointType:
        parsed: list[PointPrimitive] = []
        records = _records_for_aliases(gpt_response, POINT_KEY_ALIASES[point_type])
        for record in records:
            pos_id = _parse_id(record, ("pos_ID", "pos_id", "point_ID", "point_id", "id"))
            if pos_id is None:
                continue
            parsed.append(
                PointPrimitive(
                    point_type=point_type,
                    pos_id=pos_id,
                    confidence=_parse_confidence(
                        record,
                        ("pos_Probability", "pos_probability", "point_probability", "confidence"),
                    ),
                    position=keypoints.get(pos_id, keypoints.get(pos_id + 1)),
                    stage=str(record.get("Stage", record.get("stage", ""))),
                    description=str(record.get("Description", record.get("description", ""))),
                )
            )
        if parsed:
            primitives[point_type.value] = parsed
    return primitives


def _parse_axis_record(
    raw: dict,
    axis_type: AxisType,
    keypoints: dict[int, np.ndarray],
) -> AxisPrimitive | None:
    if "ori_ID" not in raw and "ori_id" not in raw:
        return None
    ori = raw.get("ori_ID", raw.get("ori_id"))
    if not isinstance(ori, list):
        return None
    confidence = _parse_confidence(
        raw,
        ("ori_Probability", "ori_probability", "axis_probability", "confidence"),
    )
    if len(ori) == 2 and all(isinstance(v, int) for v in ori):
        repr_type = AxisRepr.KEYPOINT_PAIR
    else:
        repr_type = AxisRepr.STANDARD
    return AxisPrimitive(
        axis_type=axis_type,
        repr_type=repr_type,
        ori_id=ori,
        confidence=confidence,
        keypoints=keypoints,
    ).resolve()


def build_place_axis(v_vert: np.ndarray) -> AxisPrimitive:
    """Build the deterministic placing axis from the calibrated vertical axis."""

    vec = np.asarray(v_vert, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Degenerate calibrated vertical axis.")
    primitive = AxisPrimitive(
        axis_type=AxisType.PLACE,
        repr_type=AxisRepr.STANDARD,
        ori_id=(vec / norm).tolist(),
        confidence=1.0,
    )
    primitive.direction = vec / norm
    return primitive


def build_primitives_from_gpt(
    gpt_response: dict,
    keypoints: dict[int, np.ndarray],
    v_vert: np.ndarray,
) -> dict[str, list[AxisPrimitive]]:
    """
    Build axis primitives from GPT-4o records.

    Multiple orientation candidates are preserved when GPT-4o provides them,
    matching the prompt instruction that several valid axis representations may
    be listed for the same semantic axis.
    """

    primitives: dict[str, list[AxisPrimitive]] = {"place": [build_place_axis(v_vert)]}
    for axis_type in (AxisType.GRASP, AxisType.UTILITY):
        parsed: list[AxisPrimitive] = []
        for record in _records_for_aliases(gpt_response, AXIS_KEY_ALIASES[axis_type]):
            primitive = _parse_axis_record(record, axis_type, keypoints)
            if primitive is not None:
                parsed.append(primitive)
        if parsed:
            primitives[axis_type.value] = parsed
    return primitives
