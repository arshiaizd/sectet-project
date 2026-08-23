#!/usr/bin/env python3
"""Validate all per-image outputs from the patch8 faithfulness evaluation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patches-per-step", type=int, default=8)
    args = parser.parse_args()

    records = json.loads(args.eval_list.read_text(encoding="utf-8"))
    expected = {Path(record["image_path"]).stem for record in records}
    json_dir = args.output_dir / "json"
    npy_dir = args.output_dir / "npy"
    actual_json = {path.stem for path in json_dir.glob("*.json")}
    actual_npy = {path.stem for path in npy_dir.glob("*.npy")}
    missing_json = sorted(expected - actual_json)
    missing_npy = sorted(expected - actual_npy)
    unexpected = sorted((actual_json | actual_npy) - expected)
    invalid = []

    for image_id in sorted(expected & actual_json & actual_npy):
        try:
            saved = json.loads((json_dir / f"{image_id}.json").read_text(encoding="utf-8"))
            total = int(saved["total_patches"])
            expected_steps = math.ceil(total / args.patches_per_step)
            lengths = [
                len(saved["insertion_score"]),
                len(saved["deletion_score"]),
                len(saved["insertion_word_score"]),
                len(saved["deletion_word_score"]),
                len(saved["region_area"]),
                len(saved["patches_changed_per_step"]),
            ]
            if lengths != [expected_steps] * 6:
                raise ValueError(f"curve lengths={lengths}, expected={expected_steps}")
            if int(saved["patches_per_step"]) != args.patches_per_step:
                raise ValueError("wrong patches_per_step")
            if any(value != args.patches_per_step for value in saved["patches_changed_per_step"][:-1]):
                raise ValueError("a non-final step did not change exactly 8 patches")
            if not 1 <= int(saved["patches_changed_per_step"][-1]) <= args.patches_per_step:
                raise ValueError("invalid final patch count")
            if sum(map(int, saved["patches_changed_per_step"])) != total:
                raise ValueError("changed patch counts do not cover the grid")
            if abs(float(saved["region_area"][-1]) - 1.0) > 1e-12:
                raise ValueError("final region area is not 1")
            saliency = np.load(npy_dir / f"{image_id}.npy", mmap_mode="r")
            expected_shape = (int(saved["grid_height"]), int(saved["grid_width"]))
            if saliency.shape != expected_shape or not np.isfinite(saliency).all():
                raise ValueError(
                    f"invalid saliency shape/values: {saliency.shape}, expected {expected_shape}"
                )
            del saliency
        except Exception as error:
            invalid.append(f"{image_id}: {error}")

    print(f"expected={len(expected)} json={len(actual_json)} npy={len(actual_npy)}")
    print(
        f"missing_json={len(missing_json)} missing_npy={len(missing_npy)} "
        f"unexpected={len(unexpected)} invalid={len(invalid)}"
    )
    if missing_json:
        print("first missing JSON IDs:", ", ".join(missing_json[:10]))
    if missing_npy:
        print("first missing NPY IDs:", ", ".join(missing_npy[:10]))
    if unexpected:
        print("first unexpected IDs:", ", ".join(unexpected[:10]))
    if invalid:
        print("first invalid outputs:", " | ".join(invalid[:10]))
    if missing_json or missing_npy or unexpected or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
