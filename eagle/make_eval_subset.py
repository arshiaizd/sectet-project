#!/usr/bin/env python3
"""Create a reproducible random subset of the canonical EAGLE evaluation list."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not 0 < args.size <= len(records):
        raise ValueError(f"subset size must be between 1 and {len(records)}")

    image_paths = [record["image_path"] for record in records]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("input evaluation list contains duplicate image IDs")

    indices = random.Random(args.seed).sample(range(len(records)), args.size)
    subset = [records[index] for index in indices]
    args.output.write_text(
        json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"selected {len(subset)} of {len(records)} records with seed {args.seed}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
