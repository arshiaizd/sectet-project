#!/usr/bin/env python3
"""Filter Point Game results after dropping targets with the fewest mask pixels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_method(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("METHOD must have the form NAME=CSV") from exc
    return name, Path(path)


def binary_se(mean: float, n: int) -> float:
    return math.sqrt(mean * (1.0 - mean) / n) if n else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--drop", type=int, default=50)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text())
    if not 0 <= args.drop < len(records):
        raise ValueError(f"--drop must be in [0, {len(records) - 1}]")
    for record in records:
        if "mask_pixels" not in record:
            raise KeyError(f"mask_pixels missing for {record.get('image_path')}")

    # Deterministic exact-size exclusion: raw white-pixel count, then image ID.
    ranked = sorted(records, key=lambda r: (int(r["mask_pixels"]), int(r["image_id"])))
    excluded = ranked[: args.drop]
    excluded_names = {record["image_path"] for record in excluded}
    retained = [record for record in records if record["image_path"] not in excluded_names]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(retained, indent=2) + "\n")

    with (args.output_dir / "excluded_50_smallest_masks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "image_path", "image_id", "select_category", "mask_pixels", "mask_fraction"],
        )
        writer.writeheader()
        for rank, record in enumerate(excluded, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "image_path": record["image_path"],
                    "image_id": record["image_id"],
                    "select_category": record["select_category"],
                    "mask_pixels": record["mask_pixels"],
                    "mask_fraction": record["mask_fraction"],
                }
            )

    expected_names = {record["image_path"] for record in records}
    summaries = []
    for method_name, csv_path in args.method:
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        names = {row["image_path"] for row in rows}
        if len(rows) != len(records) or names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            raise ValueError(
                f"{method_name}: expected {len(records)} exact rows; got {len(rows)}; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        filtered = [row for row in rows if row["image_path"] not in excluded_names]
        box_values = [int(row["pg_box"]) for row in filtered]
        mask_values = [int(row["pg_mask"]) for row in filtered]
        box_mean = sum(box_values) / len(box_values)
        mask_mean = sum(mask_values) / len(mask_values)

        safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
        filtered_path = args.output_dir / f"{safe_name}_point_game_per_image.csv"
        with filtered_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(filtered)

        summary = {
            "method": method_name,
            "num_scored": len(filtered),
            "point_game_box_mean": box_mean,
            "point_game_box_standard_error": binary_se(box_mean, len(filtered)),
            "point_game_mask_mean": mask_mean,
            "point_game_mask_standard_error": binary_se(mask_mean, len(filtered)),
            "excluded_by": "50 lowest mask_pixels; ties broken by image_id",
            "source_per_image_csv": str(csv_path),
            "filtered_per_image_csv": str(filtered_path),
        }
        (args.output_dir / f"{safe_name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)

    (args.output_dir / "all_methods_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for summary in summaries:
        print(
            f"{summary['method']}: n={summary['num_scored']} "
            f"box={summary['point_game_box_mean']:.3f} "
            f"mask={summary['point_game_mask_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
