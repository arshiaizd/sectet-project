#!/usr/bin/env python3
"""Create an aggregate + per-sample PDF for EAGLE, old ours, and new ours."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from sklearn import metrics


METHODS = ("EAGLE", "Ours (old)", "Ours (new)")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_point_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["image_path"]: row for row in csv.DictReader(handle)}


def standard_error(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0


def curve_metrics(saved: dict[str, Any], sensitivity: float = 0.4) -> dict[str, Any]:
    changed = np.asarray([0.0] + saved["region_area"], dtype=float)
    insertion = np.asarray(
        [saved["deletion_score"][-1]] + saved["insertion_score"], dtype=float
    )
    deletion = np.asarray(
        [saved["insertion_score"][-1]] + saved["deletion_score"], dtype=float
    )
    result: dict[str, Any] = {
        "changed": changed,
        "insertion_curve": insertion,
        "deletion_curve": deletion,
        "insertion_auc": float(metrics.auc(changed, insertion)),
        "deletion_auc": float(metrics.auc(1.0 - changed, deletion)),
        "highest_confidence": float(insertion.max()),
        "sensitivity_insertion_auc": None,
        "sensitivity_deletion_auc": None,
        "sensitivity_highest_confidence": None,
    }

    insertion_words = np.asarray(saved["insertion_word_score"], dtype=float)
    deletion_words = np.asarray(saved["deletion_word_score"], dtype=float)
    if insertion_words.ndim == 1:
        insertion_words = insertion_words[:, None]
    if deletion_words.ndim == 1:
        deletion_words = deletion_words[:, None]
    selected = (insertion_words[-1] - deletion_words[-1]) > sensitivity
    if selected.any():
        sensitivity_insertion = np.asarray(
            [deletion_words[-1, selected].mean()]
            + [row[selected].mean() for row in insertion_words],
            dtype=float,
        )
        sensitivity_deletion = np.asarray(
            [insertion_words[-1, selected].mean()]
            + [row[selected].mean() for row in deletion_words],
            dtype=float,
        )
        result.update(
            {
                "sensitivity_insertion_auc": float(
                    metrics.auc(changed, sensitivity_insertion)
                ),
                "sensitivity_deletion_auc": float(
                    metrics.auc(1.0 - changed, sensitivity_deletion)
                ),
                "sensitivity_highest_confidence": float(
                    sensitivity_insertion.max()
                ),
            }
        )
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def mean_se(values: list[Any]) -> tuple[float, float, int]:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)), standard_error(clean), len(clean)


def add_page_number(fig: plt.Figure, number: int, total: int) -> None:
    fig.text(0.985, 0.018, f"Page {number} of {total}", ha="right", fontsize=8, color="#555555")


def aggregate_page(
    records: list[dict[str, Any]],
    method_data: dict[str, dict[str, dict[str, Any]]],
    pdf: PdfPages,
    page: int,
    total_pages: int,
) -> dict[str, Any]:
    metric_specs = (
        ("Insertion AUC ↑", "insertion_auc"),
        ("Deletion AUC ↓", "deletion_auc"),
        ("Highest confidence ↑", "highest_confidence"),
        ("Sensitivity insertion ↑", "sensitivity_insertion_auc"),
        ("Sensitivity deletion ↓", "sensitivity_deletion_auc"),
        ("Sensitivity highest ↑", "sensitivity_highest_confidence"),
        ("Point Game box ↑", "pg_box"),
        ("Point Game mask ↑", "pg_mask"),
    )
    summary: dict[str, Any] = {}
    table_rows = []
    for method in METHODS:
        summary[method] = {}
        row = [method]
        for label, key in metric_specs:
            values = [method_data[method][record["image_path"]][key] for record in records]
            mean, se, n = mean_se(values)
            summary[method][key] = {"mean": mean, "standard_error": se, "n": n}
            row.append(f"{mean:.4f} ± {se:.4f}")
        table_rows.append(row)

    fig = plt.figure(figsize=(16.5, 11.7), facecolor="white")
    fig.text(0.05, 0.94, "Three-method evaluation — 30-image COCO benchmark", fontsize=23, weight="bold")
    fig.text(
        0.05,
        0.902,
        "EAGLE vs. old activation-patching method vs. new-prompt method",
        fontsize=13,
        color="#444444",
    )
    ax = fig.add_axes([0.035, 0.33, 0.93, 0.52])
    ax.axis("off")
    columns = ["Method"] + [label for label, _ in metric_specs]
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 2.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c8ced6")
        if r == 0:
            cell.set_facecolor("#26364a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 0:
            cell.set_facecolor("#e9eef5")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f9fb")

    notes = (
        "Values are mean ± standard error. Higher is better for insertion, highest confidence, and Point Game; "
        "lower is better for deletion. Sensitivity statistics include only samples satisfying the evaluator's "
        "0.4 sensitivity criterion; their n is recorded in the companion JSON.\n\n"
        "Faithfulness protocol: eight merged Qwen vision patches are inserted/deleted per step for both ours variants; "
        "EAGLE uses its native ordered superpixels. Point Game uses the uploaded fair centroid criterion: top-patch "
        "center for ours and top-superpixel centroid for EAGLE. All methods use the same 30 images, captions, targets, "
        "boxes, and segmentation masks.\n\n"
        "Score interpretation caveat: EAGLE curves measure the selected caption-token confidence, while ours curves "
        "measure summed yes-token probability under their respective yes/no prompts."
    )
    fig.text(0.055, 0.27, textwrap.fill(notes, 165), fontsize=10.5, va="top", linespacing=1.35)
    add_page_number(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return summary


def sample_page(
    record: dict[str, Any],
    case_number: int,
    image_root: Path,
    method_data: dict[str, dict[str, dict[str, Any]]],
    pdf: PdfPages,
    page: int,
    total_pages: int,
) -> None:
    image_name = record["image_path"]
    fig = plt.figure(figsize=(16.5, 11.7), facecolor="white")
    fig.text(0.035, 0.955, f"Case {case_number:02d} — {image_name}", fontsize=20, weight="bold")
    caption = record["selected_coco_caption"]
    target = record["target_caption_phrase"]
    category = record.get("select_category", "")
    fig.text(0.035, 0.915, "Caption: " + textwrap.fill(caption, 145), fontsize=11.5, va="top")
    target_line = f"Target phrase: {target!r}    COCO category: {category!r}    Mask fraction: {record['mask_fraction']:.4f}"
    fig.text(0.035, 0.865, target_line, fontsize=11.5, weight="bold")

    image_ax = fig.add_axes([0.035, 0.445, 0.29, 0.38])
    image_ax.imshow(Image.open(image_root / image_name).convert("RGB"))
    image_ax.set_title("COCO image", fontsize=13)
    image_ax.axis("off")

    plot_lefts = [0.355, 0.565, 0.775]
    for method, left in zip(METHODS, plot_lefts):
        ax = fig.add_axes([left, 0.46, 0.19, 0.34])
        values = method_data[method][image_name]
        ax.plot(values["changed"], values["insertion_curve"], color="#2b8c3e", lw=2.2, label="Insertion")
        ax.plot(values["changed"], values["deletion_curve"], color="#d52b3f", lw=2.2, label="Deletion")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.set_title(method, fontsize=13, weight="bold")
        ax.set_xlabel("Fraction changed", fontsize=9)
        if method == "EAGLE":
            ax.set_ylabel("Confidence / probability", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(loc="best", fontsize=8)

    metric_columns = [
        "Method", "Ins. AUC ↑", "Del. AUC ↓", "Highest ↑",
        "Sens. ins. ↑", "Sens. del. ↓", "Sens. high ↑", "PG box", "PG mask",
    ]
    metric_rows = []
    for method in METHODS:
        values = method_data[method][image_name]
        metric_rows.append(
            [
                method,
                fmt(values["insertion_auc"]),
                fmt(values["deletion_auc"]),
                fmt(values["highest_confidence"]),
                fmt(values["sensitivity_insertion_auc"]),
                fmt(values["sensitivity_deletion_auc"]),
                fmt(values["sensitivity_highest_confidence"]),
                str(int(values["pg_box"])),
                str(int(values["pg_mask"])),
            ]
        )
    table_ax = fig.add_axes([0.035, 0.13, 0.93, 0.23])
    table_ax.axis("off")
    table = table_ax.table(cellText=metric_rows, colLabels=metric_columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c8ced6")
        if r == 0:
            cell.set_facecolor("#26364a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 0:
            cell.set_facecolor("#e9eef5")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f9fb")

    fig.text(
        0.04,
        0.075,
        "PG values are binary per-sample hits. A dash means the sample did not meet the 0.4 sensitivity criterion.",
        fontsize=9,
        color="#555555",
    )
    add_page_number(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--eagle-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--eagle-point", type=Path, required=True)
    parser.add_argument("--old-point", type=Path, required=True)
    parser.add_argument("--new-point", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    records = load_json(args.manifest)
    if len(records) != 30 or len({r["image_path"] for r in records}) != 30:
        raise ValueError("manifest must contain exactly 30 unique images")

    dirs = {"EAGLE": args.eagle_dir, "Ours (old)": args.old_dir, "Ours (new)": args.new_dir}
    point_paths = {"EAGLE": args.eagle_point, "Ours (old)": args.old_point, "Ours (new)": args.new_point}
    method_data: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in METHODS}
    expected = {record["image_path"] for record in records}
    for method in METHODS:
        points = load_point_rows(point_paths[method])
        if set(points) != expected:
            raise ValueError(f"{method} Point Game IDs do not match manifest")
        json_ids = {path.stem + ".jpg" for path in (dirs[method] / "json").glob("*.json")}
        if json_ids != expected:
            raise ValueError(f"{method} faithfulness IDs do not match manifest")
        for record in records:
            image_name = record["image_path"]
            saved = load_json(dirs[method] / "json" / Path(image_name).with_suffix(".json"))
            values = curve_metrics(saved)
            values["pg_box"] = float(points[image_name]["pg_box"])
            values["pg_mask"] = float(points[image_name]["pg_mask"])
            method_data[method][image_name] = values

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    total_pages = 1 + len(records)
    with PdfPages(args.output_pdf) as pdf:
        summary = aggregate_page(records, method_data, pdf, 1, total_pages)
        for index, record in enumerate(records, start=1):
            sample_page(record, index, args.image_root, method_data, pdf, index + 1, total_pages)
        metadata = pdf.infodict()
        metadata["Title"] = "EAGLE vs old ours vs new ours — 30-image evaluation"
        metadata["Author"] = "Experiment report"

    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = [
        "case", "image_path", "caption", "target", "category", "method",
        "insertion_auc", "deletion_auc", "highest_confidence",
        "sensitivity_insertion_auc", "sensitivity_deletion_auc",
        "sensitivity_highest_confidence", "point_game_box", "point_game_mask",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, record in enumerate(records, start=1):
            for method in METHODS:
                values = method_data[method][record["image_path"]]
                writer.writerow(
                    {
                        "case": case,
                        "image_path": record["image_path"],
                        "caption": record["selected_coco_caption"],
                        "target": record["target_caption_phrase"],
                        "category": record["select_category"],
                        "method": method,
                        "insertion_auc": values["insertion_auc"],
                        "deletion_auc": values["deletion_auc"],
                        "highest_confidence": values["highest_confidence"],
                        "sensitivity_insertion_auc": values["sensitivity_insertion_auc"],
                        "sensitivity_deletion_auc": values["sensitivity_deletion_auc"],
                        "sensitivity_highest_confidence": values["sensitivity_highest_confidence"],
                        "point_game_box": int(values["pg_box"]),
                        "point_game_mask": int(values["pg_mask"]),
                    }
                )
    print(f"Wrote {total_pages}-page PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
