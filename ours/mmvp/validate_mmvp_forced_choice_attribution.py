#!/usr/bin/env python3
"""Validate exact MMVP forced-choice 150 attribution coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest) != 150:
        raise ValueError(f"expected 150 manifest entries, found {len(manifest)}")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for path in args.csv:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                grouped[int(row["dataset_sample_index"])].append(row)

    invalid: list[str] = []
    for index, item in enumerate(manifest):
        rows = grouped.get(index, [])
        if not rows:
            invalid.append(f"sample={index}: missing")
            continue
        expected = int(rows[0]["source_grid_h"]) * int(rows[0]["source_grid_w"])
        indices = {int(row["center_patch_index"]) for row in rows}
        ranks = {int(row["rank_by_target_probability"]) for row in rows}
        filenames = {row["image_path"] for row in rows}
        predictions = {row["predicted_option_text"] for row in rows}
        if (
            len(rows) != expected
            or indices != set(range(expected))
            or ranks != set(range(1, expected + 1))
            or filenames != {item["image_filename"]}
            or not predictions.issubset(set(item["allowed_outputs"]))
            or len(predictions) != 1
        ):
            invalid.append(f"sample={index}: invalid rows={len(rows)}")

    unexpected = sorted(set(grouped) - set(range(150)))
    if invalid or unexpected:
        print(f"invalid={len(invalid)} unexpected={len(unexpected)}")
        print("\n".join(invalid[:20]))
        raise SystemExit(1)
    predictions = [grouped[index][0] for index in range(150)]
    accuracy = sum(row["prediction_correct"].lower() == "true" for row in predictions) / 150
    predicted_balance = Counter(row["predicted_option_text"] for row in predictions)
    print(f"expected=150 actual=150 invalid=0 accuracy={accuracy:.6f}")
    print(f"predicted option texts={dict(predicted_balance)}")
    print("MMVP forced-choice 150 attribution coverage passed")


if __name__ == "__main__":
    main()
