#!/usr/bin/env python3
"""Validate one complete EAGLE JSON/NPY output pair per manifest image."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "insertion_score",
    "deletion_score",
    "smdl_score",
    "region_area",
    "sub-region_number",
    "selected_interpretation_token_id",
    "selected_interpretation_token_word_id",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.eval_list.read_text(encoding="utf-8"))
    expected = {Path(record["image_path"]).stem for record in records}
    json_dir = args.output_dir / "json"
    npy_dir = args.output_dir / "npy"
    actual_json = {path.stem for path in json_dir.glob("*.json")}
    actual_npy = {path.stem for path in npy_dir.glob("*.npy")}
    missing = sorted(expected - (actual_json & actual_npy))
    unexpected = sorted((actual_json | actual_npy) - expected)
    invalid = []

    for stem in sorted(expected & actual_json & actual_npy):
        try:
            saved = json.loads((json_dir / f"{stem}.json").read_text(encoding="utf-8"))
            regions = np.load(npy_dir / f"{stem}.npy", mmap_mode="r")
            if not REQUIRED.issubset(saved):
                raise ValueError("missing required JSON fields")
            if regions.ndim != 4 or regions.shape[0] != int(saved["sub-region-number"] if "sub-region-number" in saved else saved["sub-region_number"]):
                raise ValueError(f"invalid region array shape {regions.shape}")
            del regions
        except Exception as error:
            invalid.append(f"{stem}: {error}")

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
