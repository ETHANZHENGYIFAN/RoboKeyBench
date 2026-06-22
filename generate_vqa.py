"""
VQA Candidate-Pool Generation

Generates six RoboKeyBench question types from *_arrows.json annotation files:
  PSA - Part Semantic Anchoring
  FTC - Functional Type Classification
  FTR - Functional Type Retrieval
  GTV - Geometry Task Verification
  TGV - Task Geometry Verification
  PCV - Physical Constraint Validation

Both point primitives and axis primitives are treated as first-class items.
Point primitives use point[i]. Axis primitives use either standard axes
(axis[+x], axis[-z], etc.) or object-specific directed keypoint pairs
(point[i][j]), following the representation used in the manuscript.

This script builds a dense VQA candidate pool from the released annotation
files. The 14,958-question RoboKeyBench benchmark reported in the manuscript
corresponds to the curated/sampled benchmark split built from the full object
corpus, not necessarily to every candidate emitted by this utility.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE / "img"
SEED = 42
random.seed(SEED)

# Vocabulary -----------------------------------------------------------------
ALL_PARTS = [
    "blade", "body", "bottom", "button", "edge",
    "handle", "head", "interior", "lid", "opening",
    "pivot", "spout", "surface", "tines",
]

ALL_INTERACTION_TYPES = ["Grasping", "Placing", "Utility"]

POINT_LABEL = {
    "Grasping": "Grasping Point",
    "Placing": "Placing Point",
    "Utility": "Utility Point",
}

AXIS_LABEL = {
    "Grasping": "Grasping Axis",
    "Placing": "Placing Axis",
    "Utility": "Utility Axis",
}

AXIS_ROLE = {
    "Grasping": "grasping axis",
    "Placing": "placing axis",
    "Utility": "utility axis",
}

ALL_PRIMITIVE_TYPE_LABELS = list(POINT_LABEL.values()) + list(AXIS_LABEL.values())
STANDARD_AXIS_OPTIONS = ["axis[+x]", "axis[-x]", "axis[+y]", "axis[-y]", "axis[+z]", "axis[-z]"]
CANONICAL_ANGLES = [0, 45, 90, 135, 180]

ALL_TASKS = [
    "Cut", "Fill teapot", "Grasp alcohol", "Grasp fork", "Grasp kettle",
    "Grasp mug", "Grasp pan", "Grasp plate", "Grasp scissors", "Grasp spoon",
    "Grasp teapot", "Open lid", "Pierce food", "Place alcohol", "Place food",
    "Place fork", "Place kettle", "Place mug", "Place pan", "Place plate",
    "Place spoon", "Place teapot", "Pour water", "Press heating button",
    "Rotate pivot", "Scoop",
]

ANNOTATION_RULES = (
    "Annotation Rules Summary:\n"
    "Axes Identification: Red = x-axis, Green = y-axis, Blue = z-axis.\n"
    "Point Mapping: Label points numerically "
    "(e.g., point[1] refers to marked point 1 in the image).\n"
    "Direction Representation:\n"
    "For standard axes, use axis[+x/-x/+y/-y/+z/-z] "
    "(e.g., axis[+z] = positive Z-axis).\n"
    "For object-specific functional axes, use point[X][Y] to denote the "
    "direction from point X to point Y (e.g., point[1][2] = direction from "
    "point 1 to point 2)."
)
ANSWER_RULE = "Answer strictly from the provided options. Do not add explanations or extra text."


# Helpers --------------------------------------------------------------------
def unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fmt_options(opts: list[str]) -> str:
    return "[" + ", ".join(opts) + "]"


def sample_distractors(pool: list[str], exclude: set[str], n: int) -> list[str]:
    candidates = [x for x in unique(pool) if x not in exclude]
    if len(candidates) < n:
        return []
    return random.sample(candidates, n)


def shuffle_with_correct(correct: str, distractors: list[str]) -> tuple[list[str], str] | None:
    if len(distractors) < 3:
        return None
    opts = [correct] + distractors[:3]
    opts = unique(opts)
    if len(opts) < 4:
        return None
    random.shuffle(opts)
    return opts, correct


def build_question(core: str, options: list[str]) -> str:
    return (
        f"{core}\n"
        f"{ANNOTATION_RULES}\n"
        f"Options:\n{fmt_options(options)}\n"
        f"{ANSWER_RULE}"
    )


def make_entry(entry_id: int, image: str, image_path: str,
               question: str, answer: str, qtype: str) -> dict[str, Any]:
    return {
        "id": entry_id,
        "type": qtype,
        "image": image,
        "image_path": image_path,
        "question": question,
        "answer": answer,
    }


def usage(arrow: dict[str, Any]) -> dict[str, Any]:
    return arrow.get("usage_context", {}) or {}


def interaction_type(arrow: dict[str, Any]) -> str:
    return usage(arrow).get("interaction_type", "")


def task_name(arrow: dict[str, Any]) -> str:
    return usage(arrow).get("task", "")


def point_repr(arrow: dict[str, Any]) -> str:
    return f"point[{arrow['id']}]"


def arrow_start(arrow: dict[str, Any]) -> tuple[float, float]:
    pos = arrow["position"]["2D_position"]
    return float(pos["start_x"]), float(pos["start_y"])


def arrow_2d_vec(arrow: dict[str, Any]) -> tuple[float, float]:
    """Unit vector of the primitive direction in 2D image space."""
    pos = arrow["position"]["2D_position"]
    dx = float(pos["end_x"]) - float(pos["start_x"])
    dy = float(pos["end_y"]) - float(pos["start_y"])
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 0 else (1.0, 0.0)


def angle_between_arrows_deg(a1: dict[str, Any], a2: dict[str, Any]) -> float:
    v1 = arrow_2d_vec(a1)
    v2 = arrow_2d_vec(a2)
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.degrees(math.acos(dot))


def nearest_canonical(angle: float) -> int:
    return min(CANONICAL_ANGLES, key=lambda c: abs(c - angle))


def standard_axis_repr(arrow: dict[str, Any]) -> str:
    """
    Approximate a rendered 2D primitive direction with a canonical axis label.

    Placing axes are usually vertical support directions, so they are encoded
    as z-axis labels when the rendered direction is mostly vertical. Horizontal
    projected directions are encoded as x-axis labels. Object-specific axes are
    represented with point-pair labels when possible.
    """
    dx, dy = arrow_2d_vec(arrow)
    if abs(dx) >= abs(dy):
        return "axis[+x]" if dx >= 0 else "axis[-x]"
    return "axis[+z]" if dy < 0 else "axis[-z]"


def infer_axis_target_id(arrow: dict[str, Any], arrows: list[dict[str, Any]]) -> int | None:
    """Find a keypoint lying along the annotated direction for point[i][j]."""
    sx, sy = arrow_start(arrow)
    vx, vy = arrow_2d_vec(arrow)
    best_id = None
    best_score = -2.0

    for other in arrows:
        if other.get("id") == arrow.get("id"):
            continue
        ox, oy = arrow_start(other)
        dx = ox - sx
        dy = oy - sy
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            continue
        align = (dx / dist) * vx + (dy / dist) * vy
        # Prefer points in the arrow direction; distance only breaks ties.
        score = align - 0.0005 * dist
        if score > best_score:
            best_score = score
            best_id = int(other["id"])

    if best_score < 0.15:
        return None
    return best_id


def axis_repr(arrow: dict[str, Any], arrows: list[dict[str, Any]]) -> str:
    itype = interaction_type(arrow)
    if itype == "Placing":
        return standard_axis_repr(arrow)

    target_id = infer_axis_target_id(arrow, arrows)
    if target_id is not None:
        return f"point[{arrow['id']}][{target_id}]"
    return standard_axis_repr(arrow)


def point_primitive(arrow: dict[str, Any]) -> dict[str, Any] | None:
    itype = interaction_type(arrow)
    if itype not in POINT_LABEL:
        return None
    return {
        "kind": "point",
        "repr": point_repr(arrow),
        "label": POINT_LABEL[itype],
        "role": POINT_LABEL[itype].lower(),
        "interaction_type": itype,
        "task": task_name(arrow),
        "part": arrow.get("part", ""),
        "arrow": arrow,
    }


def axis_primitive(arrow: dict[str, Any], arrows: list[dict[str, Any]]) -> dict[str, Any] | None:
    itype = interaction_type(arrow)
    if itype not in AXIS_LABEL:
        return None
    return {
        "kind": "axis",
        "repr": axis_repr(arrow, arrows),
        "label": AXIS_LABEL[itype],
        "role": AXIS_ROLE[itype],
        "interaction_type": itype,
        "task": task_name(arrow),
        "part": arrow.get("part", ""),
        "arrow": arrow,
    }


def build_primitive_sets(arrows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = [p for a in arrows if (p := point_primitive(a)) is not None]
    axes = [p for a in arrows if (p := axis_primitive(a, arrows)) is not None]
    return points, axes


def primitive_pool(primitives: list[dict[str, Any]]) -> list[str]:
    return unique([p["repr"] for p in primitives])


# Question generators ---------------------------------------------------------
def gen_psa(prim: dict[str, Any], category: str) -> tuple[str, str] | None:
    """PSA: given an operation primitive, determine part-level semantics."""
    part = prim.get("part", "")
    if not part:
        return None

    distractors = sample_distractors(ALL_PARTS, {part}, 3)
    choice = shuffle_with_correct(part, distractors)
    if choice is None:
        return None

    if prim["kind"] == "axis":
        core = f"Which part of the {category} is {prim['repr']} functionally associated with?"
    else:
        core = f"Which part of the {category} does {prim['repr']} represent?"
    opts, ans = choice
    return build_question(core, opts), ans


def gen_ftc(prim: dict[str, Any], category: str) -> tuple[str, str] | None:
    """FTC: classify a point or axis primitive into functional type labels."""
    correct = prim["label"]
    distractors = sample_distractors(ALL_PRIMITIVE_TYPE_LABELS, {correct}, 3)
    choice = shuffle_with_correct(correct, distractors)
    if choice is None:
        return None

    part = prim.get("part", "")
    core = f"What is the functional type of {prim['repr']} in the {part} section of the {category}?"
    opts, ans = choice
    return build_question(core, opts), ans


def gen_ftr(primitives: list[dict[str, Any]], category: str,
            itype: str, kind: str) -> tuple[str, str] | None:
    """FTR: given a functional type, retrieve the matching primitive."""
    correct_prims = [p for p in primitives if p["kind"] == kind and p["interaction_type"] == itype]
    wrong_prims = [p for p in primitives if p["kind"] == kind and p["interaction_type"] != itype]
    if not correct_prims:
        return None

    correct_prim = random.choice(correct_prims)
    correct = correct_prim["repr"]
    distractor_pool = primitive_pool(wrong_prims)
    if kind == "axis":
        distractor_pool += STANDARD_AXIS_OPTIONS
    distractors = sample_distractors(distractor_pool, {correct}, 3)
    choice = shuffle_with_correct(correct, distractors)
    if choice is None:
        return None

    if kind == "axis":
        core = f"Which axis serves as the {AXIS_ROLE[itype]} in the {category}?"
    else:
        core = f"Which point serves as the {POINT_LABEL[itype]} in the {category}?"
    opts, ans = choice
    return build_question(core, opts), ans


def gen_gtv(prim: dict[str, Any], category: str) -> tuple[str, str] | None:
    """GTV: given a primitive, select the associated task."""
    task = prim.get("task", "")
    if not task:
        return None

    distractors = sample_distractors(ALL_TASKS, {task}, 3)
    choice = shuffle_with_correct(task, distractors)
    if choice is None:
        return None

    part = prim.get("part", "")
    core = f"Which task is associated with {prim['repr']} in the {part} section of the {category}?"
    opts, ans = choice
    return build_question(core, opts), ans


def gen_tgv(primitives: list[dict[str, Any]], category: str,
            task: str, preferred_kind: str | None = None) -> tuple[str, str] | None:
    """TGV: given a task, retrieve a required point or axis primitive."""
    correct_prims = [p for p in primitives if p.get("task") == task]
    if preferred_kind is not None:
        correct_prims = [p for p in correct_prims if p["kind"] == preferred_kind]
    wrong_prims = [p for p in primitives if p.get("task") != task]
    if not correct_prims:
        return None

    correct_prim = random.choice(correct_prims)
    correct = correct_prim["repr"]
    distractor_pool = primitive_pool(wrong_prims)
    distractors = sample_distractors(distractor_pool, {correct}, 3)
    choice = shuffle_with_correct(correct, distractors)
    if choice is None:
        return None

    core = f'Which primitive is associated with the task "{task}" in the {category}?'
    opts, ans = choice
    return build_question(core, opts), ans


def pcv_task_context(axis1: dict[str, Any], axis2: dict[str, Any]) -> str:
    if axis1.get("task") and axis1.get("task") == axis2.get("task"):
        return axis1["task"]

    for axis in (axis1, axis2):
        if axis["interaction_type"] == "Utility" and axis.get("task"):
            return axis["task"]

    for axis in (axis1, axis2):
        if axis.get("task"):
            return axis["task"]

    return ""


def gen_pcv(axis1: dict[str, Any], axis2: dict[str, Any], category: str) -> tuple[str, str] | None:
    """PCV: validate task-conditioned angle constraints between functional axes."""
    task = pcv_task_context(axis1, axis2)
    if not task:
        return None

    real_angle = angle_between_arrows_deg(axis1["arrow"], axis2["arrow"])
    correct_deg = nearest_canonical(real_angle)
    correct = f"{correct_deg} degrees"
    canonical_strs = [f"{a} degrees" for a in CANONICAL_ANGLES]
    distractors = sample_distractors(canonical_strs, {correct}, 3)
    choice = shuffle_with_correct(correct, distractors)
    if choice is None:
        return None

    core = (
        f'When performing "{task}", what task-conditioned angle is feasible between '
        f'the {axis1["role"]} ({axis1["repr"]}) and the {axis2["role"]} '
        f'({axis2["repr"]}) in the {category}?'
    )
    opts, ans = choice
    return build_question(core, opts), ans


def pcv_axis_pairs(axes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Prioritize semantically meaningful PCV pairs.

    PCV is intended to test whether a model can reason about a task-relevant
    functional axis under a physical constraint. Therefore, pairs involving a
    utility axis and a placing reference axis are preferred first, followed by
    utility-grasping references. Same-role axis pairs are kept only as a last-resort
    fallback for sparse annotations.
    """
    utility_placing = []
    utility_grasping = []
    same_task_cross_type = []
    other_cross_type = []
    same_type_fallback = []
    seen = set()

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a1, a2 = axes[i], axes[j]
            pair_key = tuple(sorted((id(a1), id(a2))))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            pair = (a1, a2)
            types = {a1["interaction_type"], a2["interaction_type"]}
            same_task = bool(a1.get("task") and a1.get("task") == a2.get("task"))

            if types == {"Utility", "Placing"}:
                utility_placing.append(pair)
            elif types == {"Utility", "Grasping"}:
                utility_grasping.append(pair)
            elif same_task and len(types) > 1:
                same_task_cross_type.append(pair)
            elif len(types) > 1:
                other_cross_type.append(pair)
            else:
                same_type_fallback.append(pair)

    groups = (utility_placing, utility_grasping, same_task_cross_type, other_cross_type, same_type_fallback)
    for group in groups:
        random.shuffle(group)

    return utility_placing + utility_grasping + same_task_cross_type + other_cross_type + same_type_fallback


# Main ------------------------------------------------------------------------
def generate_all(output_path: Path, max_pcv_per_file: int = 3) -> None:
    entries: list[dict[str, Any]] = []
    eid = 0

    for json_path in sorted(ROOT.rglob("*_arrows.json")):
        category = json_path.parts[json_path.parts.index("img") + 1]
        img_file = json_path.name.replace("_arrows.json", "_marked.png")
        rel_dir = json_path.parent.relative_to(ROOT)
        image_path = str(rel_dir / img_file).replace("\\", "/")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        arrows = data.get("arrows", [])
        if not arrows:
            continue

        points, axes = build_primitive_sets(arrows)
        all_primitives = points + axes

        def add(result: tuple[str, str] | None, qtype: str) -> None:
            nonlocal eid
            if result is None:
                return
            q, a = result
            entries.append(make_entry(eid, img_file, image_path, q, a, qtype))
            eid += 1

        # Primitive understanding layer: PSA and FTC for both points and axes.
        for prim in all_primitives:
            add(gen_psa(prim, category), "PSA")
            add(gen_ftc(prim, category), "FTC")
            add(gen_gtv(prim, category), "GTV")

        # Functional type retrieval: point and axis variants.
        for itype in ALL_INTERACTION_TYPES:
            add(gen_ftr(all_primitives, category, itype, "point"), "FTR")
            add(gen_ftr(all_primitives, category, itype, "axis"), "FTR")

        # Task geometry verification: retrieve point and axis primitives.
        tasks_in_file = {p["task"] for p in all_primitives if p.get("task")}
        for task in tasks_in_file:
            add(gen_tgv(all_primitives, category, task, "point"), "TGV")
            add(gen_tgv(all_primitives, category, task, "axis"), "TGV")

        # Physical constraint validation: prioritize utility-vs-reference axis
        # pairs so PCV reflects function-aware constraints, not raw keypoint
        # angle calculation alone.
        pairs = pcv_axis_pairs(axes)
        if pairs:
            for axis1, axis2 in pairs[:max_pcv_per_file]:
                add(gen_pcv(axis1, axis2, category), "PCV")

    by_type = defaultdict(int)
    for entry in entries:
        by_type[entry["type"]] += 1

    print(f"Total candidate entries: {len(entries)}")
    for qtype, count in sorted(by_type.items()):
        print(f"  {qtype}: {count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a RoboKeyBench VQA candidate pool.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "vqa_candidate_pool.json",
        help="Output JSON path for the generated VQA candidate pool.",
    )
    parser.add_argument(
        "--max-pcv-per-file",
        type=int,
        default=3,
        help="Maximum number of PCV candidate questions generated per annotation file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(args.output, max_pcv_per_file=args.max_pcv_per_file)
