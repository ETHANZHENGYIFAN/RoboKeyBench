"""
VQA Dataset Generation
Generates six question types from *_arrows.json annotation files:
  PSA – Part Semantic Anchoring
  FTC – Functional Type Classification
  FTR – Functional Type Retrieval
  GTV – Geometry Task Verification
  TGV – Task Geometry Verification
  PCV – Physical Constraint Validation
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict

ROOT    = Path("./RoboKeyBench/img")
SEED    = 42
random.seed(SEED)

# ── Vocabulary ────────────────────────────────────────────────────────────────
ALL_PARTS = [
    "blade", "body", "bottom", "button", "edge",
    "handle", "head", "interior", "lid", "opening",
    "pivot", "spout", "surface", "tines",
]

ALL_INTERACTION_TYPES = ["Grasping", "Placing", "Utility"]

INTERACTION_LABEL = {
    "Grasping": "Grasp Point",
    "Placing":  "Place Point",
    "Utility":  "Utility Point",
}

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
    "For standard axes, use axis[±x/y/z] (e.g., axis[+z] = positive Z-axis).\n"
    "For object-specific directions, use point[X][Y] to denote the direction "
    "from point X to point Y (e.g., point[1][2] = direction from point 1 to "
    "point 2)."
)
ANSWER_RULE = "Answer strictly from the provided options. Do not add any explanations or additional text."


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_options(opts: list[str]) -> str:
    return "[" + ", ".join(opts) + "]"


def shuffle_with_correct(correct: str, distractors: list[str]) -> tuple[list[str], str]:
    """Return shuffled option list and the (unchanged) correct answer."""
    opts = [correct] + distractors[:3]
    random.shuffle(opts)
    return opts, correct


def sample_distractors(pool: list, exclude: set, n: int) -> list:
    candidates = [x for x in pool if x not in exclude]
    return random.sample(candidates, min(n, len(candidates)))


def build_question(core: str, options: list[str]) -> str:
    return (
        f"{core}\n"
        f"{ANNOTATION_RULES}\n"
        f"Options:\n{fmt_options(options)}\n"
        f"{ANSWER_RULE}"
    )


def arrow_2d_vec(arrow: dict) -> tuple[float, float]:
    """Unit vector of the arrow in 2-D image space."""
    p = arrow["position"]["2D_position"]
    dx = p["end_x"] - p["start_x"]
    dy = p["end_y"] - p["start_y"]
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 0 else (1.0, 0.0)


def angle_between_arrows_deg(a1: dict, a2: dict) -> float:
    v1 = arrow_2d_vec(a1)
    v2 = arrow_2d_vec(a2)
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1]))
    return math.degrees(math.acos(dot))


def nearest_canonical(angle: float) -> int:
    """Round angle to nearest of {0, 45, 90, 135, 180}."""
    canonical = [0, 45, 90, 135, 180]
    return min(canonical, key=lambda c: abs(c - angle))


def make_entry(entry_id: int, image: str, image_path: str,
               question: str, answer: str, qtype: str) -> dict:
    return {
        "id":         entry_id,
        "type":       qtype,
        "image":      image,
        "image_path": image_path,
        "question":   question,
        "answer":     answer,
    }


# ── Question Generators ───────────────────────────────────────────────────────

def gen_psa(arrow: dict, arrows: list, category: str) -> tuple[str, str] | None:
    """PSA – Part Semantic Anchoring."""
    aid  = arrow["id"]
    part = arrow.get("part", "")
    if not part:
        return None

    distractors = sample_distractors(ALL_PARTS, {part}, 3)
    if len(distractors) < 3:
        return None

    core = (f"Which part of the {category} does point[{aid}] represent?")
    opts, ans = shuffle_with_correct(part, distractors)
    return build_question(core, opts), ans


def gen_ftc(arrow: dict, category: str) -> tuple[str, str] | None:
    """FTC – Functional Type Classification."""
    aid    = arrow["id"]
    itype  = arrow.get("usage_context", {}).get("interaction_type", "")
    part   = arrow.get("part", "")
    if not itype:
        return None

    correct = INTERACTION_LABEL[itype]
    all_labels = list(INTERACTION_LABEL.values())
    distractors = sample_distractors(all_labels, {correct}, 3)

    core = (f"What is the functional type of point[{aid}] "
            f"in the {part} section of the {category}?")
    opts, ans = shuffle_with_correct(correct, distractors)
    return build_question(core, opts), ans


def gen_ftr(arrows: list, category: str, itype: str) -> tuple[str, str] | None:
    """FTR – Functional Type Retrieval.
    'Which point is used as the {itype} point in the {category}?'
    Picks one correct arrow; others are distractors.
    """
    correct_arrows = [a for a in arrows
                      if a.get("usage_context", {}).get("interaction_type") == itype]
    wrong_arrows   = [a for a in arrows
                      if a.get("usage_context", {}).get("interaction_type") != itype]

    if not correct_arrows or len(wrong_arrows) < 3:
        return None

    correct_a = random.choice(correct_arrows)
    wrong_a   = random.sample(wrong_arrows, min(3, len(wrong_arrows)))

    correct   = f"point[{correct_a['id']}]"
    distractors = [f"point[{a['id']}]" for a in wrong_a]

    label = INTERACTION_LABEL[itype]
    core  = f"Which point serves as the {label} in the {category}?"
    opts, ans = shuffle_with_correct(correct, distractors)
    return build_question(core, opts), ans


def gen_gtv(arrow: dict, category: str) -> tuple[str, str] | None:
    """GTV – Geometry Task Verification.
    'Which task is point[X] associated with?'
    """
    aid  = arrow["id"]
    task = arrow.get("usage_context", {}).get("task", "")
    part = arrow.get("part", "")
    if not task:
        return None

    distractors = sample_distractors(ALL_TASKS, {task}, 3)
    if len(distractors) < 3:
        return None

    core = (f"Which task is associated with point[{aid}] "
            f"in the {part} section of the {category}?")
    opts, ans = shuffle_with_correct(task, distractors)
    return build_question(core, opts), ans


def gen_tgv(arrows: list, category: str, task: str) -> tuple[str, str] | None:
    """TGV – Task Geometry Verification.
    'Which point is associated with the task "{task}"?'
    """
    correct_arrows = [a for a in arrows
                      if a.get("usage_context", {}).get("task") == task]
    wrong_arrows   = [a for a in arrows
                      if a.get("usage_context", {}).get("task") != task]

    if not correct_arrows or len(wrong_arrows) < 3:
        return None

    correct_a   = random.choice(correct_arrows)
    wrong_a     = random.sample(wrong_arrows, min(3, len(wrong_arrows)))

    correct     = f"point[{correct_a['id']}]"
    distractors = [f"point[{a['id']}]" for a in wrong_a]

    core = f'Which point is associated with the task "{task}" in the {category}?'
    opts, ans = shuffle_with_correct(correct, distractors)
    return build_question(core, opts), ans


def gen_pcv(a1: dict, a2: dict, category: str) -> tuple[str, str] | None:
    """PCV – Physical Constraint Validation.
    'When performing "{task}", what is the optimal angle between
     point[X] and point[Y]?'
    """
    t1 = a1.get("usage_context", {}).get("task", "")
    if not t1:
        return None

    real_angle  = angle_between_arrows_deg(a1, a2)
    correct_deg = nearest_canonical(real_angle)
    correct     = f"{correct_deg} degrees"

    canonical_strs = ["0 degrees", "45 degrees", "90 degrees",
                      "135 degrees", "180 degrees"]
    distractors = sample_distractors(canonical_strs, {correct}, 3)
    if len(distractors) < 3:
        return None

    core = (f'When performing "{t1}", what is the optimal angle between '
            f'point[{a1["id"]}] and point[{a2["id"]}] in the {category}?')
    opts, ans = shuffle_with_correct(correct, distractors)
    return build_question(core, opts), ans


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(output_path: Path, max_pcv_per_file: int = 3):
    entries = []
    eid = 0

    for json_path in sorted(ROOT.rglob("*_arrows.json")):
        # Derive paths
        category   = json_path.parts[json_path.parts.index("img") + 1]
        img_file   = json_path.name.replace("_arrows.json", "_marked.png")
        # Relative path from ROOT parent (i.e., from RoboKeyBench/img/)
        rel_dir    = json_path.parent.relative_to(ROOT)
        image_path = str(rel_dir / img_file).replace("\\", "/")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        arrows = data.get("arrows", [])
        if not arrows:
            continue

        def add(result, qtype):
            nonlocal eid
            if result is None:
                return
            q, a = result
            entries.append(make_entry(eid, img_file, image_path, q, a, qtype))
            eid += 1

        # PSA + FTC + GTV per arrow
        for arrow in arrows:
            add(gen_psa(arrow, arrows, category), "PSA")
            add(gen_ftc(arrow, category),         "FTC")
            add(gen_gtv(arrow, category),         "GTV")

        # FTR per interaction type present in this file
        for itype in ALL_INTERACTION_TYPES:
            add(gen_ftr(arrows, category, itype), "FTR")

        # TGV per unique task present in this file
        tasks_in_file = {a["usage_context"]["task"]
                         for a in arrows if a.get("usage_context", {}).get("task")}
        for task in tasks_in_file:
            add(gen_tgv(arrows, category, task), "TGV")

        # PCV – sample a few arrow pairs
        if len(arrows) >= 2:
            pairs = [(arrows[i], arrows[j])
                     for i in range(len(arrows))
                     for j in range(i + 1, len(arrows))]
            for a1, a2 in random.sample(pairs, min(max_pcv_per_file, len(pairs))):
                add(gen_pcv(a1, a2, category), "PCV")

    # Summary
    by_type = defaultdict(int)
    for e in entries:
        by_type[e["type"]] += 1

    print(f"Total entries: {len(entries)}")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    out = Path("./vqa_dataset.json")
    generate_all(out)
