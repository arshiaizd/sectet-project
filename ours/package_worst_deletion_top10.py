#!/usr/bin/env python3
"""Package the ten highest (worst) deletion-AUC samples for our method."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/vilab/scratch/arshia/projects/izadi")
ATTRIBUTION_DIR = ROOT / "ours/results/encoder_250_seed_20260815"
EVALUATION_DIR = ATTRIBUTION_DIR / "evaluation_patch8"
METRICS_CSV = EVALUATION_DIR / "ours_patch8_per_image_metrics.csv"
CANONICAL_JSON = ROOT / "shared/coco_eval_250_seed_20260815.json"
COCO_DIR = Path("/mnt/vilab/scratch/arshia/datasets/coco/val2017")
OUTPUT_ZIP = ROOT / "ours/reports/ours_top10_worst_deletion.zip"


def read_metrics():
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = [
        "insertion_auc",
        "deletion_auc",
        "highest_confidence",
        "sensitive_insertion_auc",
        "sensitive_deletion_auc",
        "sensitive_highest_confidence",
        "pointing_game_box",
        "pointing_game_mask",
    ]
    for row in rows:
        for key in numeric:
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return sorted(rows, key=lambda row: row["deletion_auc"], reverse=True)[:10]


def collect_patch_rows(selected_ids):
    selected = {image_id: [] for image_id in selected_ids}
    for rank in range(4):
        path = ATTRIBUTION_DIR / f"rank{rank}.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                image_id = Path(row["image_path"]).stem
                if image_id in selected:
                    selected[image_id].append(row)
    for image_id, rows in selected.items():
        if not rows:
            raise ValueError(f"no patch attribution rows for {image_id}")
        rows.sort(key=lambda row: int(row["center_patch_index"]))
    return selected


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_overview(image_path, saliency_path, evaluation, metrics, output_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    saliency = np.load(saliency_path)
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    resized = cv2.resize(saliency, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].imshow(image)
    axes[0].set_title("COCO image")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(resized, cmap="jet", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Patch attribution")
    axes[1].axis("off")

    area = np.asarray(evaluation["region_area"], dtype=float)
    insertion = np.asarray(evaluation["insertion_score"], dtype=float)
    deletion = np.asarray(evaluation["deletion_score"], dtype=float)
    insertion_x = np.concatenate(([0.0], area))
    insertion_y = np.concatenate(([deletion[-1]], insertion))
    deletion_x = np.concatenate(([0.0], area))
    deletion_y = np.concatenate(([insertion[-1]], deletion))
    axes[2].plot(insertion_x, insertion_y, label="Insertion", color="#2E933C", linewidth=2)
    axes[2].plot(deletion_x, deletion_y, label="Deletion", color="#D7263D", linewidth=2)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("Fraction of patches inserted/deleted")
    axes[2].set_ylabel("Summed yes-token probability")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    axes[2].set_title(
        f"Insertion AUC={metrics['insertion_auc']:.4f}\n"
        f"Deletion AUC={metrics['deletion_auc']:.4f}"
    )
    figure.suptitle(f"{metrics['rank']}. {metrics['image_id']} — worst deletion ranking", weight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    top = read_metrics()
    canonical = {
        Path(row["image_path"]).stem: row
        for row in json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
    }
    patch_rows = collect_patch_rows([row["image_id"] for row in top])
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    enriched = []
    for rank, row in enumerate(top, start=1):
        image_id = row["image_id"]
        first_patch = patch_rows[image_id][0]
        enriched.append(
            {
                "rank": rank,
                **row,
                "image_path": f"{image_id}.jpg",
                "object_label": first_patch["object_label"],
                "target_token_dataset": first_patch["target_token_dataset"],
                "prompt": first_patch["prompt"],
                "select_category": first_patch["select_category"],
                "gt_caption": canonical[image_id].get("gt_caption", ""),
                "generated_sentence": canonical[image_id].get("generate_sentence", ""),
            }
        )

    with tempfile.TemporaryDirectory(prefix="ours_top10_deletion_") as temp_name:
        package = Path(temp_name) / "ours_top10_worst_deletion"
        package.mkdir()
        write_csv(package / "top10_per_example_scores.csv", enriched)
        (package / "top10_per_example_scores.json").write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        readme = (
            "Our method: 10 worst deletion examples\n"
            "=======================================\n\n"
            "Ranking criterion: deletion AUC in descending order. Higher deletion AUC is worse.\n"
            "Evaluation: summed yes-token probability; cumulative insertion/deletion of eight\n"
            "ranked 28x28 merged vision patches per regular step (final step uses the remainder).\n\n"
            "top10_per_example_scores.csv/json contain exactly one row/object per example.\n"
            "Each numbered directory contains the COCO image, every patch-level attribution score,\n"
            "the complete per-step insertion/deletion curves, the saliency grid, per-example metrics,\n"
            "and an overview visualization.\n"
        )
        (package / "README.txt").write_text(readme, encoding="utf-8")

        for metrics in enriched:
            image_id = metrics["image_id"]
            sample = package / f"{metrics['rank']:02d}_{image_id}"
            sample.mkdir()
            image_source = COCO_DIR / f"{image_id}.jpg"
            evaluation_source = EVALUATION_DIR / "json" / f"{image_id}.json"
            saliency_source = EVALUATION_DIR / "npy" / f"{image_id}.npy"
            shutil.copy2(image_source, sample / f"{image_id}.jpg")
            shutil.copy2(evaluation_source, sample / "evaluation_curves.json")
            shutil.copy2(saliency_source, sample / "saliency_grid.npy")
            write_csv(sample / "patch_scores.csv", patch_rows[image_id])
            (sample / "per_example_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            evaluation = json.loads(evaluation_source.read_text(encoding="utf-8"))
            make_overview(
                image_source,
                saliency_source,
                evaluation,
                metrics,
                sample / "overview.png",
            )

        temporary_zip = OUTPUT_ZIP.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package.parent))
        temporary_zip.replace(OUTPUT_ZIP)

    print(OUTPUT_ZIP)
    for row in enriched:
        print(
            f"{row['rank']:02d} {row['image_id']} "
            f"deletion_auc={row['deletion_auc']:.6f} "
            f"insertion_auc={row['insertion_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
