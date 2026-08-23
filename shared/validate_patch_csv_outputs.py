#!/usr/bin/env python3
"""Validate complete patch-grid CSV groups for every manifest image."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    records = json.loads(args.eval_list.read_text(encoding="utf-8"))
    expected = {Path(record["image_path"]).name for record in records}
    groups = defaultdict(list)
    for csv_path in args.csv:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                groups[Path(row["image_path"]).name].append(row)

    actual = set(groups)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    invalid = []
    for name in sorted(expected & actual):
        rows = groups[name]
        try:
            grid_h = int(rows[0]["source_grid_h"])
            grid_w = int(rows[0]["source_grid_w"])
            indices = {int(row["center_patch_index"]) for row in rows}
            if len(rows) != grid_h * grid_w or indices != set(range(grid_h * grid_w)):
                raise ValueError(f"incomplete {grid_h}x{grid_w} patch grid")
        except Exception as error:
            invalid.append(f"{name}: {error}")

    print(
        f"expected={len(expected)} complete={len(expected)-len(missing)-len(invalid)} "
        f"missing={len(missing)} unexpected={len(unexpected)} invalid={len(invalid)}"
    )
    if missing:
        print("first missing IDs:", ", ".join(missing[:10]))
    if unexpected:
        print("first unexpected IDs:", ", ".join(unexpected[:10]))
    if invalid:
        print("first invalid outputs:", " | ".join(invalid[:10]))
    if missing or unexpected or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
