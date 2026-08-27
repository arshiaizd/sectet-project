#!/usr/bin/env python3
"""Validate exact, non-overlapping MMVP-300 attribution coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest) != 300:
        raise ValueError(f"expected 300 manifest entries, found {len(manifest)}")

    rows_by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
    for csv_path in args.csv:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rows_by_sample[int(row["dataset_sample_index"])].append(row)

    expected_indices = set(range(300))
    actual_indices = set(rows_by_sample)
    missing = sorted(expected_indices - actual_indices)
    unexpected = sorted(actual_indices - expected_indices)
    invalid: list[str] = []
    for index, rows in sorted(rows_by_sample.items()):
        grid_h = int(rows[0]["source_grid_h"])
        grid_w = int(rows[0]["source_grid_w"])
        expected_patches = grid_h * grid_w
        patch_indices = [int(row["center_patch_index"]) for row in rows]
        ranks = [int(row["rank_by_target_probability"]) for row in rows]
        expected_filename = str(manifest[index]["image_filename"])
        filenames = {row["image_path"] for row in rows}
        if (
            len(rows) != expected_patches
            or set(patch_indices) != set(range(expected_patches))
            or set(ranks) != set(range(1, expected_patches + 1))
            or filenames != {expected_filename}
        ):
            invalid.append(
                f"sample={index} rows={len(rows)} expected={expected_patches} "
                f"files={sorted(filenames)}"
            )

    print(
        f"expected=300 actual={len(actual_indices)} missing={len(missing)} "
        f"unexpected={len(unexpected)} invalid={len(invalid)}"
    )
    if missing:
        print(f"first missing indices: {missing[:20]}")
    if unexpected:
        print(f"unexpected indices: {unexpected[:20]}")
    if invalid:
        print("first invalid samples:")
        print("\n".join(invalid[:20]))
    if missing or unexpected or invalid:
        raise SystemExit(1)
    print("MMVP-300 attribution coverage passed")


if __name__ == "__main__":
    main()
