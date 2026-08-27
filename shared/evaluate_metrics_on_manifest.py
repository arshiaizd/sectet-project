#!/usr/bin/env python3
"""Aggregate saved faithfulness and Point Game results on an exact manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


SENSITIVITY_THRESHOLD = 0.4


def parse_method(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("METHOD must have the form NAME=RESULT_DIR=PG_CSV")
    return parts[0], Path(parts[1]), Path(parts[2])


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Equivalent to sklearn.metrics.auc for monotonic x."""
    return float(abs(np.trapz(y, x)))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "standard_error": float("nan"), "n": 0}
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "standard_error": se, "n": int(array.size)}


def sample_metrics(data: dict) -> dict[str, float | None]:
    area = np.asarray([0.0] + data["region_area"], dtype=float)
    insertion = np.asarray([data["deletion_score"][-1]] + data["insertion_score"], dtype=float)
    deletion = np.asarray([data["insertion_score"][-1]] + data["deletion_score"], dtype=float)
    if not (len(area) == len(insertion) == len(deletion)):
        raise ValueError("inconsistent curve lengths")

    result: dict[str, float | None] = {
        "insertion_auc": auc(area, insertion),
        "deletion_auc": auc(1.0 - area, deletion),
        "highest_confidence": float(insertion.max()),
        "sensitive_insertion_auc": None,
        "sensitive_deletion_auc": None,
        "sensitive_highest_confidence": None,
    }

    insertion_words = np.asarray(data["insertion_word_score"], dtype=float)
    deletion_words = np.asarray(data["deletion_word_score"], dtype=float)
    sensitive = (insertion_words[-1] - deletion_words[-1]) > SENSITIVITY_THRESHOLD
    if sensitive.any():
        sensitive_insertion = np.concatenate(
            ([deletion_words[-1, sensitive].mean()], insertion_words[:, sensitive].mean(axis=1))
        )
        sensitive_deletion = np.concatenate(
            ([insertion_words[-1, sensitive].mean()], deletion_words[:, sensitive].mean(axis=1))
        )
        result["sensitive_insertion_auc"] = auc(area, sensitive_insertion)
        result["sensitive_deletion_auc"] = auc(1.0 - area, sensitive_deletion)
        result["sensitive_highest_confidence"] = float(sensitive_insertion.max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    image_names = [record["image_path"] for record in manifest]
    if len(image_names) != len(set(image_names)):
        raise ValueError("manifest contains duplicate image paths")
    expected = set(image_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "insertion_auc",
        "deletion_auc",
        "highest_confidence",
        "sensitive_insertion_auc",
        "sensitive_deletion_auc",
        "sensitive_highest_confidence",
        "point_game_box",
        "point_game_mask",
    ]
    all_summaries = []
    summary_csv_rows = []

    for method_name, result_dir, pg_csv in args.method:
        json_dir = result_dir / "json"
        available = {path.stem: path for path in json_dir.glob("*.json")}
        missing_json = [Path(name).stem for name in image_names if Path(name).stem not in available]
        if missing_json:
            raise ValueError(f"{method_name}: missing JSON for {missing_json[:5]}")

        with pg_csv.open(newline="") as handle:
            pg_rows = list(csv.DictReader(handle))
        pg_by_image = {row["image_path"]: row for row in pg_rows}
        missing_pg = sorted(expected - set(pg_by_image))
        if missing_pg:
            raise ValueError(f"{method_name}: missing PG for {missing_pg[:5]}")

        per_image = []
        accumulated: dict[str, list[float]] = {name: [] for name in metric_names}
        for image_name in image_names:
            values = sample_metrics(json.loads(available[Path(image_name).stem].read_text()))
            values["point_game_box"] = float(pg_by_image[image_name]["pg_box"])
            values["point_game_mask"] = float(pg_by_image[image_name]["pg_mask"])
            row = {"image_path": image_name, **values}
            per_image.append(row)
            for metric_name, value in values.items():
                if value is not None:
                    accumulated[metric_name].append(float(value))

        summaries = {name: summarize(values) for name, values in accumulated.items()}
        safe_name = re.sub(r"[^a-z0-9]+", "_", method_name.lower()).strip("_")
        per_image_path = args.output_dir / f"{safe_name}_per_image_metrics.csv"
        with per_image_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_image[0].keys())
            writer.writeheader()
            writer.writerows(per_image)

        method_summary = {
            "method": method_name,
            "cohort_manifest": str(args.manifest),
            "num_images": len(image_names),
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "source_result_directory": str(result_dir),
            "source_point_game_csv": str(pg_csv),
            "per_image_metrics_csv": str(per_image_path),
            "metrics": summaries,
        }
        (args.output_dir / f"{safe_name}_summary.json").write_text(
            json.dumps(method_summary, indent=2, allow_nan=True) + "\n"
        )
        all_summaries.append(method_summary)
        summary_csv_rows.append(
            {
                "method": method_name,
                **{
                    f"{metric}_{field}": summaries[metric][field]
                    for metric in metric_names
                    for field in ("mean", "standard_error", "n")
                },
            }
        )

    (args.output_dir / "all_methods_metrics.json").write_text(
        json.dumps(all_summaries, indent=2, allow_nan=True) + "\n"
    )
    with (args.output_dir / "all_methods_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_csv_rows)

    for method in all_summaries:
        metrics = method["metrics"]
        print(
            f"{method['method']}: n={method['num_images']} "
            f"ins={metrics['insertion_auc']['mean']:.4f} "
            f"del={metrics['deletion_auc']['mean']:.4f} "
            f"high={metrics['highest_confidence']['mean']:.4f} "
            f"pg_box={metrics['point_game_box']['mean']:.4f} "
            f"pg_mask={metrics['point_game_mask']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
