"""RoboKeyBench result statistics.

This script summarizes evaluation files under the current RoboKeyBench
taxonomy:

  Layer 1: PSA, FTC, FTR
  Layer 2: GTV, TGV
  Layer 3: PCV

It accepts both current candidate-pool style records and model prediction
records that contain `ground_truth`, `prediction`, and `correct` fields.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


QUESTION_TYPES = ("PSA", "FTC", "FTR", "GTV", "TGV", "PCV")
LAYER_BY_TYPE = {
    "PSA": "Layer 1",
    "FTC": "Layer 1",
    "FTR": "Layer 1",
    "GTV": "Layer 2",
    "TGV": "Layer 2",
    "PCV": "Layer 3",
}

POINT_LABELS = {"Grasping Point", "Placing Point", "Utility Point"}
AXIS_LABELS = {"Grasping Axis", "Placing Axis", "Utility Axis"}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON structure in {path}")
    return data


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value).strip()
    text = re.sub(r"^\(?[A-D]\)?[\.:)\-\s]+", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def ground_truth(item: dict[str, Any]) -> str:
    return normalize_text(item.get("answer", item.get("ground_truth", "")))


def prediction(item: dict[str, Any]) -> str:
    return normalize_text(
        item.get("prediction", item.get("pred", item.get("response", "")))
    )


def infer_question_type(item: dict[str, Any]) -> str:
    qtype = str(item.get("type", "")).upper()
    if qtype in QUESTION_TYPES:
        return qtype

    question = normalize_text(item.get("question", "")).lower()
    if question.startswith("which part"):
        return "PSA"
    if question.startswith("what is the functional type") or question.startswith(
        "what type of primitive"
    ):
        return "FTC"
    if question.startswith("which point serves") or question.startswith(
        "which axis serves"
    ):
        return "FTR"
    if question.startswith("which task is associated"):
        return "GTV"
    if question.startswith("which primitive is associated") or question.startswith(
        "which primitive is related"
    ):
        return "TGV"
    if "task-conditioned angle" in question or "degrees" in ground_truth(item).lower():
        return "PCV"
    return "UNKNOWN"


def question_core(item: dict[str, Any]) -> str:
    question = normalize_text(item.get("question", ""))
    return question.split("Annotation Rules Summary:", 1)[0].strip()


def infer_primitive_family(item: dict[str, Any]) -> str:
    core = question_core(item)
    answer_text = ground_truth(item)
    text = f"{core} {answer_text}"

    if any(label in answer_text for label in AXIS_LABELS):
        return "axis"
    if any(label in answer_text for label in POINT_LABELS):
        return "point"
    if re.search(r"point\[\d+\]\[\d+\]", text):
        return "axis"
    if "axis[" in text or "axis serves" in core.lower() or "which axis" in core.lower():
        return "axis"
    if re.search(r"point\[\d+\]", text) or "point serves" in core.lower() or "which point" in core.lower():
        return "point"
    return "unknown"


def record_correct(item: dict[str, Any]) -> bool:
    if isinstance(item.get("correct"), bool):
        return bool(item["correct"])
    if item.get("status") and item.get("status") != "success":
        return False

    gt = ground_truth(item)
    pred = prediction(item)
    if not gt or not pred:
        return False

    gt_options = [normalize_text(x) for x in gt.split(",")]
    return pred == gt or pred in gt_options


def new_bucket() -> dict[str, float]:
    return {"correct": 0, "total": 0, "accuracy": 0.0}


def update_bucket(bucket: dict[str, float], is_correct: bool) -> None:
    bucket["total"] += 1
    if is_correct:
        bucket["correct"] += 1


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    total = bucket["total"]
    bucket["accuracy"] = round(bucket["correct"] * 100.0 / total, 2) if total else 0.0
    bucket["correct"] = int(bucket["correct"])
    bucket["total"] = int(bucket["total"])
    return bucket


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_type = defaultdict(new_bucket)
    by_layer = defaultdict(new_bucket)
    by_family = defaultdict(new_bucket)
    overall = new_bucket()

    for item in records:
        qtype = infer_question_type(item)
        family = infer_primitive_family(item)
        ok = record_correct(item)

        update_bucket(overall, ok)
        update_bucket(by_type[qtype], ok)
        update_bucket(by_layer[LAYER_BY_TYPE.get(qtype, "UNKNOWN")], ok)
        update_bucket(by_family[family], ok)

    return {
        "overall": finalize_bucket(overall),
        "by_question_type": {
            key: finalize_bucket(by_type[key])
            for key in sorted(by_type, key=lambda x: QUESTION_TYPES.index(x) if x in QUESTION_TYPES else 99)
        },
        "by_capability_layer": {
            key: finalize_bucket(by_layer[key]) for key in sorted(by_layer)
        },
        "by_point_axis": {
            key: finalize_bucket(by_family[key]) for key in sorted(by_family)
        },
    }


def summarize_paths(paths: Iterable[Path]) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    files = []
    for path in paths:
        records = load_records(path)
        all_records.extend(records)
        files.append({"path": str(path), "records": len(records)})

    summary = summarize_records(all_records)
    summary["files"] = files
    return summary


def resolve_inputs(repo_root: Path, inputs: list[str], default_glob: str | None) -> list[Path]:
    if inputs:
        paths = [Path(x) for x in inputs]
    elif default_glob:
        paths = sorted(repo_root.glob(default_glob))
    else:
        paths = [
            repo_root / "results" / "gpt4o" / "predictions_test.json",
            repo_root / "results" / "gpt4o" / "predictions_val.json",
        ]
    resolved = [p if p.is_absolute() else repo_root / p for p in paths]
    return [p for p in resolved if p.exists()]


def build_parser(default_glob: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RoboKeyBench result files.")
    parser.add_argument("--inputs", nargs="*", default=[], help="Result JSON files.")
    parser.add_argument("--output", type=Path, help="Optional path to save the summary JSON.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root used to resolve relative input paths.",
    )
    parser.set_defaults(default_glob=default_glob)
    return parser


def main(default_glob: str | None = None) -> None:
    parser = build_parser(default_glob=default_glob)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    inputs = resolve_inputs(repo_root, args.inputs, args.default_glob)
    if not inputs:
        raise FileNotFoundError("No result files found. Pass --inputs explicitly.")

    summary = summarize_paths(inputs)
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)

    if args.output:
        out = args.output if args.output.is_absolute() else repo_root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
