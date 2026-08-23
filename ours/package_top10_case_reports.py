#!/usr/bin/env python3
"""Build deletion/insertion top-10 ZIPs with one consolidated per-case PDF each."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/mnt/vilab/scratch/arshia/projects/izadi")
ATTRIBUTION_DIR = ROOT / "ours/results/encoder_250_seed_20260815"
EVALUATION_DIR = ATTRIBUTION_DIR / "evaluation_patch8"
METRICS_CSV = EVALUATION_DIR / "ours_patch8_per_image_metrics.csv"
CANONICAL_JSON = ROOT / "shared/coco_eval_250_seed_20260815.json"
COCO_DIR = Path("/mnt/vilab/scratch/arshia/datasets/coco/val2017")
REPORT_DIR = ROOT / "ours/reports"
EAGLE_DIR = ROOT / "eagle/results/eagle_250_seed_20260815/slico-1.0-1.0-division-number-64"
sys.path.insert(0, str(ROOT / "eagle"))
from make_eagle_pdf_report import evaluate_record as evaluate_eagle_record
from make_eagle_pdf_with_pointing_se import pointing_game as eagle_pointing_game
METRIC_KEYS = [
    "insertion_auc",
    "deletion_auc",
    "highest_confidence",
    "sensitive_insertion_auc",
    "sensitive_deletion_auc",
    "sensitive_highest_confidence",
    "pointing_game_box",
    "pointing_game_mask",
]


def load_all_metrics():
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in METRIC_KEYS:
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return rows


def collect_patch_rows(selected_ids):
    selected = {image_id: [] for image_id in selected_ids}
    for rank in range(4):
        with (ATTRIBUTION_DIR / f"rank{rank}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                image_id = Path(row["image_path"]).stem
                if image_id in selected:
                    selected[image_id].append(row)
    for image_id, rows in selected.items():
        if not rows:
            raise ValueError(f"no patch rows found for {image_id}")
        rows.sort(key=lambda row: int(row["center_patch_index"]))
    return selected


def compute_eagle_metrics(selected_ids, canonical):
    results = {}
    for image_id in sorted(set(selected_ids)):
        saved = json.loads((EAGLE_DIR / "json" / f"{image_id}.json").read_text(encoding="utf-8"))
        values = evaluate_eagle_record(saved, 0.4)
        regions = np.load(EAGLE_DIR / "npy" / f"{image_id}.npy", mmap_mode="r")
        box_hit, mask_hit = eagle_pointing_game(regions, saved, COCO_DIR / canonical[image_id]["image_path"])
        del regions
        results[image_id] = {**values, "pointing_game_box": float(box_hit), "pointing_game_mask": float(mask_hit)}
    return results


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_saliency(image_id, image):
    saliency = np.load(EVALUATION_DIR / "npy" / f"{image_id}.npy")
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    return cv2.resize(
        saliency,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def curve_arrays(evaluation):
    area = np.asarray(evaluation["region_area"], dtype=float)
    insertion = np.asarray(evaluation["insertion_score"], dtype=float)
    deletion = np.asarray(evaluation["deletion_score"], dtype=float)
    return (
        np.concatenate(([0.0], area)),
        np.concatenate(([deletion[-1]], insertion)),
        np.concatenate(([insertion[-1]], deletion)),
    )


def metric_text(value):
    return "N/A" if value is None else f"{value:.6f}"


def add_summary_page(pdf, rows, criterion, direction):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    label = "Deletion AUC (highest is worst)" if criterion == "deletion_auc" else "Insertion AUC (lowest is worst)"
    ax.set_title(f"Our Method — 10 Worst {direction.title()} Cases", fontsize=20, weight="bold", pad=20)
    ax.text(
        0.02,
        0.91,
        f"Ranking criterion: {label}. All values below are per-example values.",
        fontsize=10,
        transform=ax.transAxes,
    )
    columns = ["Rank", "Image ID", "Object", "Our ins.", "EAGLE ins.", "Our del.", "EAGLE del."]
    data = [
        [
            row["rank"],
            row["image_id"],
            row["object_label"],
            f"{row['insertion_auc']:.4f}",
            f"{row['eagle_insertion_auc']:.4f}",
            f"{row['deletion_auc']:.4f}",
            f"{row['eagle_deletion_auc']:.4f}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.25, 0.98, 0.60],
        colWidths=[0.07, 0.15, 0.16, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C2CC")
        if r == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF3F8")
    ax.text(
        0.02,
        0.15,
        "Method: vision-encoder activation patching. Evaluation target: summed yes-token probability. "
        "Each regular evaluation step inserts or deletes eight ranked 28×28 merged vision patches; "
        "the final step uses the remaining patches. EAGLE scores use the exact same image IDs.",
        fontsize=9,
        transform=ax.transAxes,
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_case_page(pdf, row, evaluation, eagle_evaluation):
    image_bgr = cv2.imread(str(COCO_DIR / f"{row['image_id']}.jpg"))
    if image_bgr is None:
        raise FileNotFoundError(row["image_id"])
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    saliency = normalized_saliency(row["image_id"], image)
    area, insertion, deletion = curve_arrays(evaluation)
    eagle_area, eagle_insertion, eagle_deletion = curve_arrays(eagle_evaluation)

    fig = plt.figure(figsize=(11.69, 8.27))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.85], hspace=0.28, wspace=0.22)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[0, 2])
    ax_eagle_curve = fig.add_subplot(grid[0, 3])
    ax_metrics = fig.add_subplot(grid[1, :])
    fig.suptitle(
        f"Rank {row['rank']} · COCO {row['image_id']} · object: {row['object_label']}",
        fontsize=16,
        weight="bold",
        y=0.98,
    )

    ax_image.imshow(image)
    ax_image.set_title("Original image")
    ax_image.axis("off")
    ax_map.imshow(image)
    ax_map.imshow(saliency, cmap="jet", alpha=0.50, vmin=0, vmax=1)
    ax_map.set_title("Patch attribution")
    ax_map.axis("off")
    ax_curve.plot(area, insertion, color="#2E933C", linewidth=2, label="Insertion")
    ax_curve.plot(area, deletion, color="#D7263D", linewidth=2, label="Deletion")
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.set_xlabel("Fraction of patches changed")
    ax_curve.set_ylabel("Summed yes probability")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=8)
    ax_curve.set_title("Our faithfulness")
    ax_eagle_curve.plot(eagle_area, eagle_insertion, color="#2E933C", linewidth=2, label="Insertion")
    ax_eagle_curve.plot(eagle_area, eagle_deletion, color="#D7263D", linewidth=2, label="Deletion")
    ax_eagle_curve.set_xlim(0, 1)
    ax_eagle_curve.set_ylim(-0.02, 1.02)
    ax_eagle_curve.set_xlabel("Fraction changed")
    ax_eagle_curve.set_ylabel("Canonical-token probability")
    ax_eagle_curve.grid(alpha=0.25)
    ax_eagle_curve.legend(fontsize=8)
    ax_eagle_curve.set_title("EAGLE faithfulness")

    ax_metrics.axis("off")
    labels = {
        "insertion_auc": "Insertion AUC",
        "deletion_auc": "Deletion AUC",
        "highest_confidence": "Highest confidence",
        "sensitive_insertion_auc": "Sensitive insertion AUC",
        "sensitive_deletion_auc": "Sensitive deletion AUC",
        "sensitive_highest_confidence": "Sensitive highest confidence",
        "pointing_game_box": "Pointing Game — box",
        "pointing_game_mask": "Pointing Game — mask",
    }
    metric_rows = [[labels[key], metric_text(row[key]), metric_text(row[f"eagle_{key}"])] for key in METRIC_KEYS]
    table = ax_metrics.table(
        cellText=metric_rows,
        colLabels=["Per-example metric", "Our method", "EAGLE"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.08, 0.62, 0.88],
        colWidths=[0.46, 0.27, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (table_row, table_col), cell in table.get_celld().items():
        cell.set_edgecolor("#C4CCD4")
        if table_row == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif table_col == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
    ax_metrics.text(0.66, 0.91, "Case details", transform=ax_metrics.transAxes, fontsize=10, weight="bold", va="top")
    ax_metrics.text(
        0.66,
        0.82,
        f"Dataset index: {evaluation['dataset_sample_index']}\n"
        f"Patch grid: {evaluation['grid_height']}×{evaluation['grid_width']}\n"
        f"Total patches: {evaluation['total_patches']}\n"
        f"Evaluation steps: {len(evaluation['region_area'])}\n"
        f"Patches/regular step: {evaluation['patches_per_step']}\n\n"
        "Our configuration:\n"
        "vision blocks 0–last\n"
        "square neighborhood + center\n"
        "white full canvas\n"
        "norm-preserving injection\n\n"
        "Our target: summed yes-token mass\n"
        "EAGLE target: canonical token",
        transform=ax_metrics.transAxes,
        fontsize=8.6,
        va="top",
        linespacing=1.35,
    )
    ax_metrics.text(0.66, 0.12, "Prompt: " + row["prompt"], transform=ax_metrics.transAxes, fontsize=8, va="top", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf(path, rows, criterion, direction):
    with PdfPages(path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = f"Our method: 10 worst {direction} cases"
        metadata["Subject"] = "Per-case metrics, patch maps, and faithfulness curves"
        add_summary_page(pdf, rows, criterion, direction)
        for row in rows:
            evaluation = json.loads(
                (EVALUATION_DIR / "json" / f"{row['image_id']}.json").read_text(encoding="utf-8")
            )
            eagle_evaluation = json.loads((EAGLE_DIR / "json" / f"{row['image_id']}.json").read_text(encoding="utf-8"))
            add_case_page(pdf, row, evaluation, eagle_evaluation)


def build_package(all_metrics, canonical, eagle_metrics, criterion, direction):
    reverse = criterion == "deletion_auc"
    selected = sorted(all_metrics, key=lambda row: row[criterion], reverse=reverse)[:10]
    patch_rows = collect_patch_rows([row["image_id"] for row in selected])
    enriched = []
    for rank, metrics in enumerate(selected, start=1):
        image_id = metrics["image_id"]
        first = patch_rows[image_id][0]
        enriched.append( { "rank": rank, **metrics, "object_label": first["object_label"], "prompt": first["prompt"], "select_category": first["select_category"], "target_token_dataset": first["target_token_dataset"], "dataset_sample_index": int(first["dataset_sample_index"]), "gt_caption": canonical[image_id].get("gt_caption", ""), "generated_sentence": canonical[image_id].get("generate_sentence", ""), })
        for key in METRIC_KEYS:
            enriched[-1][f"eagle_{key}"] = eagle_metrics[image_id][key]

    zip_path = REPORT_DIR / f"ours_top10_worst_{direction}.zip"
    with tempfile.TemporaryDirectory(prefix=f"ours_{direction}_top10_") as temporary:
        package = Path(temporary) / f"ours_top10_worst_{direction}"
        package.mkdir(parents=True)
        write_csv(package / "top10_per_example_scores.csv", enriched)
        (package / "top10_per_example_scores.json").write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        criterion_text = "highest deletion AUC" if direction == "deletion" else "lowest insertion AUC"
        (package / "README.txt").write_text(
            f"Our method: 10 worst {direction} cases\n\n"
            f"Ranking criterion: {criterion_text}.\n"
            "All metrics are per-example. Each PDF metric table includes our method and EAGLE for the same image.\n",
            encoding="utf-8",
        )
        pdf_path = package / f"top10_worst_{direction}_per_case_report.pdf"
        create_pdf(pdf_path, enriched, criterion, direction)

        for row in enriched:
            image_id = row["image_id"]
            sample = package / f"{row['rank']:02d}_{image_id}"
            sample.mkdir()
            shutil.copy2(COCO_DIR / f"{image_id}.jpg", sample / f"{image_id}.jpg")
            shutil.copy2(
                EVALUATION_DIR / "json" / f"{image_id}.json",
                sample / "evaluation_curves.json",
            )
            shutil.copy2(
                EAGLE_DIR / "json" / f"{image_id}.json",
                sample / "eagle_evaluation_curves.json",
            )
            shutil.copy2(
                EVALUATION_DIR / "npy" / f"{image_id}.npy",
                sample / "saliency_grid.npy",
            )
            write_csv(sample / "patch_scores.csv", patch_rows[image_id])
            (sample / "per_example_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        temporary_zip = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package.parent))
        temporary_zip.replace(zip_path)
    return zip_path, enriched


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = load_all_metrics()
    canonical = {
        Path(row["image_path"]).stem: row
        for row in json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
    }
    selected_ids = {row["image_id"] for row in sorted(all_metrics, key=lambda row: row["deletion_auc"], reverse=True)[:10]}
    selected_ids.update(row["image_id"] for row in sorted(all_metrics, key=lambda row: row["insertion_auc"])[:10])
    eagle_metrics = compute_eagle_metrics(selected_ids, canonical)
    for criterion, direction in (("deletion_auc", "deletion"), ("insertion_auc", "insertion")):
        path, rows = build_package(all_metrics, canonical, eagle_metrics, criterion, direction)
        print(path)
        for row in rows:
            print(
                f"{direction} rank={row['rank']:02d} image={row['image_id']} "
                f"insertion={row['insertion_auc']:.6f} deletion={row['deletion_auc']:.6f}"
            )


if __name__ == "__main__":
    main()
