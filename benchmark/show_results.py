#!/usr/bin/env python3
"""Print experiment status and tracked final metrics without GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmark/results.json"
METHODS = ("ours", "input_level", "eagle", "tam", "llavacam")
DATASETS = ("coco", "mmvp")
MODELS = ("qwen", "internvl")


def value(number: object) -> str:
    return "—" if number is None else f"{float(number):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--json", action="store_true", help="Print matching registry entries as JSON")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selected: dict[str, dict] = {}
    for key, experiment in registry["experiments"].items():
        model, dataset, method = key.split(".")
        if args.model and model != args.model:
            continue
        if args.dataset and dataset != args.dataset:
            continue
        if args.method and method != args.method:
            continue
        selected[key] = experiment
    if args.json:
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return

    columns = ("experiment", "status", "ins", "del", "high", "PG-box", "PG-mask", "eval GPU-s/sample")
    rows = []
    for key, experiment in selected.items():
        metrics = experiment.get("metrics", {})
        rows.append(
            (
                key,
                experiment["status"],
                value(metrics.get("insertion_auc")),
                value(metrics.get("deletion_auc")),
                value(metrics.get("highest_confidence")),
                value(metrics.get("point_game_box")),
                value(metrics.get("point_game_mask")),
                value(metrics.get("evaluation_gpu_seconds_per_sample")),
            )
        )
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    print("  ".join(columns[i].ljust(widths[i]) for i in range(len(columns))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    for key, experiment in selected.items():
        if experiment.get("notes"):
            print(f"\n{key}: {experiment['notes']}")


if __name__ == "__main__":
    main()
