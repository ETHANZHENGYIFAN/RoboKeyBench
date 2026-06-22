"""Convenience entry point for open-source model result statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_statistics_module():
    module_path = Path(__file__).with_name("statistics.py")
    spec = importlib.util.spec_from_file_location("robokeybench_statistics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load statistics module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _load_statistics_module().main(default_glob="results/open_source/*_results_*.json")
