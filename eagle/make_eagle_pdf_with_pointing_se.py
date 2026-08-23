#!/usr/bin/env python3
"""Generate the consolidated EAGLE report with Pointing Game and standard errors."""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from make_eagle_pdf_report import (
    add_plot_pages,
    add_summary_page,
    add_table_pages,
    evaluate_record,
    file_sha256,
    mean_present,
)


def standard_error(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def segmentation_to_mask(segmentation, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not segmentation:
        return mask
    polygons = segmentation if isinstance(segmentation[0], (list, tuple)) else [segmentation]
    for polygon in polygons:
        if polygon is None or len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask


def eagle_attribution_map(regions, saved):
    """Exact reconstruction used by EAGLE visualization.add_value."""
    attribution = np.zeros_like(regions[0]).astype(np.float16)
    current_scores = np.asarray(saved["smdl_score"], dtype=float)
    previous_scores = np.asarray(
        [
            np.mean(
                1
                - np.asarray(saved["org_score"], dtype=float)
                + np.asarray(saved["baseline_score"], dtype=float)
            )
        ]
        + saved["smdl_score"][:-1],
        dtype=float,
    )
    value = 0.0
    for region, score_delta in zip(regions, current_scores - previous_scores):
        value -= abs(score_delta)
        attribution[region == 1] = value
    attribution = attribution - attribution.min()
    maximum = attribution.max()
    if maximum == 0 or not np.isfinite(maximum):
        return np.full(attribution.shape, np.nan, dtype=float)
    return attribution.astype(float) / maximum


def pointing_game(regions, saved, image_path):
    """Match official EAGLE eval_point_game box and mask decisions."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    attribution = eagle_attribution_map(regions, saved)
    attribution = cv2.resize(attribution, (width, height))
    if not np.isfinite(attribution).all():
        return 0, 0
    attribution = attribution - attribution.min()
    attribution = attribution / (attribution.max() + 1e-8)
    attribution = cv2.resize(
        attribution, (width, height), interpolation=cv2.INTER_LINEAR
    )

    x1, y1, x2, y2 = map(int, saved["location"])
    box = np.zeros((height, width), dtype=np.uint8)
    box[y1:y2, x1:x2] = 1
    segmentation = segmentation_to_mask(saved["segmentation"], width, height)
    maximum = attribution.max()
    return (
        int((attribution * box).max() == maximum),
        int((attribution * segmentation).max() == maximum),
    )


def add_statistics_page(pdf, rows):
    specifications = [
        ("Insertion AUC", "insertion_auc"),
        ("Deletion AUC", "deletion_auc"),
        ("Highest confidence", "highest_confidence"),
        ("Sensitive insertion AUC", "sensitive_insertion_auc"),
        ("Sensitive deletion AUC", "sensitive_deletion_auc"),
        ("Sensitive highest confidence", "sensitive_highest_confidence"),
        ("Pointing Game (Box)", "pointing_box"),
        ("Pointing Game (Mask)", "pointing_mask"),
        ("Runtime (seconds/image/GPU)", "runtime_seconds"),
    ]
    table_rows = []
    for label, key in specifications:
        values = [row[key] for row in rows if row[key] is not None]
        table_rows.append(
            [label, f"{np.mean(values):.6f}", f"{standard_error(values):.6f}", str(len(values))]
        )

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title("Statistical Results", fontsize=20, weight="bold", pad=24)
    ax.text(
        0.0,
        0.93,
        "Values are reported as sample mean and standard error (sample SD / sqrt(n)).\n"
        "Pointing Game follows the official EAGLE box- and mask-level evaluator.",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Mean", "Standard error", "n"],
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.48, 1.0, 0.38],
        colWidths=[0.48, 0.20, 0.22, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#EAF0F8")

    box_hits = sum(row["pointing_box"] for row in rows)
    mask_hits = sum(row["pointing_mask"] for row in rows)
    ax.text(
        0.02,
        0.39,
        f"Pointing Game box hits:  {box_hits}/{len(rows)}\n"
        f"Pointing Game mask hits: {mask_hits}/{len(rows)}\n\n"
        "A hit is recorded when the maximum reconstructed attribution value occurs inside "
        "the target COCO bounding box or segmentation mask. Ties follow the official "
        "implementation: a maximum inside the annotation counts as a hit.",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity", type=float, default=0.4)
    args = parser.parse_args()

    canonical = json.loads(args.eval_list.read_text(encoding="utf-8"))
    rows = []
    for record in canonical:
        image_id = Path(record["image_path"]).stem
        json_path = args.results / "json" / f"{image_id}.json"
        npy_path = args.results / "npy" / f"{image_id}.npy"
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        regions = np.load(npy_path, mmap_mode="r")
        if regions.ndim != 4 or regions.shape[0] != saved["sub-region_number"]:
            raise ValueError(f"invalid result pair for {image_id}")
        scores = evaluate_record(saved, args.sensitivity)
        box_hit, mask_hit = pointing_game(
            regions, saved, args.datasets / f"{image_id}.jpg"
        )
        target = saved.get("words", "")
        if isinstance(target, list):
            target = " ".join(map(str, target))
        rows.append(
            {
                "image_id": image_id,
                "target": target,
                "rank": int(saved["worker_rank"]),
                "runtime_seconds": float(saved["runtime_seconds"]),
                "pointing_box": box_hit,
                "pointing_mask": mask_hit,
                **scores,
            }
        )
        del regions

    rank_stats = defaultdict(list)
    for row in rows:
        rank_stats[row["rank"]].append(row["runtime_seconds"])
    runtimes = np.asarray([row["runtime_seconds"] for row in rows])
    critical_path = max(sum(values) for values in rank_stats.values())
    summary = {
        "count": len(rows),
        "canonical_count": len(canonical),
        "invalid_count": 0,
        "eval_sha256": file_sha256(args.eval_list),
        "insertion_auc": float(np.mean([row["insertion_auc"] for row in rows])),
        "deletion_auc": float(np.mean([row["deletion_auc"] for row in rows])),
        "highest_confidence": float(np.mean([row["highest_confidence"] for row in rows])),
        "sensitive_insertion_auc": mean_present(rows, "sensitive_insertion_auc"),
        "sensitive_deletion_auc": mean_present(rows, "sensitive_deletion_auc"),
        "sensitive_highest_confidence": mean_present(rows, "sensitive_highest_confidence"),
        "sensitivity": args.sensitivity,
        "runtime_mean": float(runtimes.mean()),
        "runtime_median": float(np.median(runtimes)),
        "runtime_min": float(runtimes.min()),
        "runtime_max": float(runtimes.max()),
        "runtime_total": float(runtimes.sum()),
        "parallel_critical_path": float(critical_path),
        "throughput": len(rows) / (critical_path / 3600.0),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "EAGLE Qwen2.5-VL-7B complete COCO report"
        metadata["Subject"] = "Faithfulness, Pointing Game, standard errors, and runtimes"
        add_statistics_page(pdf, rows)
        add_summary_page(pdf, summary)
        add_plot_pages(pdf, rows, rank_stats)
        add_table_pages(pdf, rows)

    metric_keys = [
        "insertion_auc",
        "deletion_auc",
        "highest_confidence",
        "sensitive_insertion_auc",
        "sensitive_deletion_auc",
        "sensitive_highest_confidence",
        "pointing_box",
        "pointing_mask",
        "runtime_seconds",
    ]
    for key in metric_keys:
        values = [row[key] for row in rows if row[key] is not None]
        print(f"{key}: mean={np.mean(values):.6f} se={standard_error(values):.6f} n={len(values)}")
    print(f"PDF: {args.output}")


if __name__ == "__main__":
    main()
