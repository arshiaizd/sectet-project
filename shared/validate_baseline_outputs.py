#!/usr/bin/env python3
"""Validate that a baseline stage produced one complete artifact per canonical ID."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["npy", "json"], required=True)
    args = parser.parse_args()

    records = json.loads(args.eval_list.read_text(encoding="utf-8"))
    expected = {Path(record["image_path"]).stem for record in records}
    actual = {path.stem for path in args.output_dir.glob(f"*.{args.kind}")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    invalid = []

    for image_id in sorted(expected & actual):
        path = args.output_dir / f"{image_id}.{args.kind}"
        try:
            if args.kind == "npy":
                array = np.load(path, mmap_mode="r")
                if array.ndim != 2 or array.size == 0:
                    raise ValueError(f"unexpected shape {array.shape}")
                del array
            else:
                saved = json.loads(path.read_text(encoding="utf-8"))
                lengths = [
                    len(saved["insertion_score"]),
                    len(saved["deletion_score"]),
                    len(saved["insertion_word_score"]),
                    len(saved["deletion_word_score"]),
                    len(saved["region_area"]),
                ]
                if lengths[0] != 64 or len(set(lengths)) != 1:
                    raise ValueError(f"unexpected curve lengths {lengths}")
        except Exception as error:
            invalid.append(f"{image_id}: {error}")

    print(f"expected={len(expected)} actual={len(actual)}")
    print(f"missing={len(missing)} unexpected={len(unexpected)} invalid={len(invalid)}")
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
