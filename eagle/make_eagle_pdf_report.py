#!/usr/bin/env python3
"""Build a single PDF report from completed EAGLE result files."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics


def format_duration(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    if hours:
        return f"{hours}h {minutes:02d}m {remaining:04.1f}s"
    return f"{minutes}m {remaining:04.1f}s"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_record(saved, sensitivity):
    insertion_area = np.asarray([0.0] + saved["region_area"], dtype=float)
    deletion_area = 1.0 - insertion_area
    insertion_score = np.asarray(
        [saved["deletion_score"][-1]] + saved["insertion_score"], dtype=float
    )
    deletion_score = np.asarray(
        [saved["insertion_score"][-1]] + saved["deletion_score"], dtype=float
    )

    result = {
        "insertion_auc": float(metrics.auc(insertion_area, insertion_score)),
        "deletion_auc": float(metrics.auc(deletion_area, deletion_score)),
        "highest_confidence": float(insertion_score.max()),
        "sensitive_insertion_auc": None,
        "sensitive_deletion_auc": None,
        "sensitive_highest_confidence": None,
    }

    insertion_words = np.asarray(saved["insertion_word_score"], dtype=float)
    deletion_words = np.asarray(saved["deletion_word_score"], dtype=float)
    sensitive = (insertion_words[-1] - deletion_words[-1]) > sensitivity
    if sensitive.any():
        sensitive_insertion = np.asarray(
            [deletion_words[-1][sensitive].mean()]
            + [row[sensitive].mean() for row in insertion_words],
            dtype=float,
        )
        sensitive_deletion = np.asarray(
            [insertion_words[-1][sensitive].mean()]
            + [row[sensitive].mean() for row in deletion_words],
            dtype=float,
        )
        result.update(
            sensitive_insertion_auc=float(
                metrics.auc(insertion_area, sensitive_insertion)
            ),
            sensitive_deletion_auc=float(
                metrics.auc(deletion_area, sensitive_deletion)
            ),
            sensitive_highest_confidence=float(sensitive_insertion.max()),
        )
    return result


def mean_present(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else float("nan")


def add_summary_page(pdf, summary):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "EAGLE — Qwen2.5-VL-7B COCO Report", fontsize=20, weight="bold")
    fig.text(0.08, 0.91, "Canonical random 250-image evaluation subset", fontsize=11, color="#555555")

    blocks = [
        (
            "Run integrity",
            [
                f"Completed outputs: {summary['count']}/250 JSON/NPY pairs",
                f"Canonical IDs matched: {summary['canonical_count']}/250",
                f"Corrupt or mismatched pairs: {summary['invalid_count']}",
                f"Evaluation-list seed: 20260815",
                f"Evaluation-list SHA-256: {summary['eval_sha256']}",
            ],
        ),
        (
            "Official EAGLE faithfulness metrics",
            [
                f"Insertion AUC: {summary['insertion_auc']:.4f}",
                f"Deletion AUC: {summary['deletion_auc']:.4f}",
                f"Average highest confidence: {summary['highest_confidence']:.4f}",
                f"Sensitive insertion AUC: {summary['sensitive_insertion_auc']:.4f}",
                f"Sensitive deletion AUC: {summary['sensitive_deletion_auc']:.4f}",
                f"Sensitive average highest confidence: {summary['sensitive_highest_confidence']:.4f}",
                f"Sensitivity threshold: {summary['sensitivity']}",
            ],
        ),
        (
            "Runtime",
            [
                f"Mean per image/GPU: {format_duration(summary['runtime_mean'])}",
                f"Median per image/GPU: {format_duration(summary['runtime_median'])}",
                f"Fastest / slowest: {format_duration(summary['runtime_min'])} / {format_duration(summary['runtime_max'])}",
                f"Total GPU compute: {format_duration(summary['runtime_total'])}",
                f"Parallel compute critical path: {format_duration(summary['parallel_critical_path'])}",
                f"Effective throughput: {summary['throughput']:.2f} images/hour",
            ],
        ),
        (
            "Configuration",
            [
                "Model: Qwen2.5-VL-7B-Instruct",
                "Workers: 4 independent torchrun processes (one model replica/GPU)",
                "Attention: eager; dtype: bfloat16 on A100",
                "Superpixels: SLICO; target division count: 64",
                "lambda1 = 1.0; lambda2 = 1.0",
            ],
        ),
    ]

    y = 0.85
    for title, lines in blocks:
        fig.text(0.08, y, title, fontsize=13, weight="bold", color="#17365D")
        y -= 0.032
        for line in lines:
            fig.text(0.10, y, line, fontsize=9.5, family="monospace")
            y -= 0.026
        y -= 0.025

    fig.text(
        0.08,
        0.035,
        f"Generated {summary['generated_at']} from saved outputs; no model inference was performed.",
        fontsize=8,
        color="#666666",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_plot_pages(pdf, rows, rank_stats):
    runtimes = np.asarray([row["runtime_seconds"] for row in rows]) / 60.0
    insertion = np.asarray([row["insertion_auc"] for row in rows])
    deletion = np.asarray([row["deletion_auc"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Aggregate distributions", fontsize=17, weight="bold")
    axes[0, 0].hist(runtimes, bins=20, color="#4472C4", edgecolor="white")
    axes[0, 0].axvline(runtimes.mean(), color="#C00000", linestyle="--", label=f"mean {runtimes.mean():.2f}m")
    axes[0, 0].set(title="Runtime per image", xlabel="Minutes", ylabel="Images")
    axes[0, 0].legend()

    axes[0, 1].hist(insertion, bins=20, alpha=0.75, label="Insertion", color="#70AD47")
    axes[0, 1].hist(deletion, bins=20, alpha=0.75, label="Deletion", color="#ED7D31")
    axes[0, 1].set(title="Per-image faithfulness AUC", xlabel="AUC", ylabel="Images")
    axes[0, 1].legend()

    axes[1, 0].scatter(deletion, insertion, s=14, alpha=0.65, color="#5B9BD5")
    axes[1, 0].set(title="Insertion versus deletion AUC", xlabel="Deletion AUC", ylabel="Insertion AUC")
    axes[1, 0].grid(alpha=0.2)

    ranks = sorted(rank_stats)
    rank_hours = [sum(rank_stats[rank]) / 3600.0 for rank in ranks]
    axes[1, 1].bar([str(rank) for rank in ranks], rank_hours, color="#8064A2")
    axes[1, 1].set(title="GPU-worker compute time", xlabel="Worker rank", ylabel="Hours")
    for index, value in enumerate(rank_hours):
        axes[1, 1].text(index, value + 0.05, f"{value:.2f}h", ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_pages(pdf, rows, rows_per_page=32):
    columns = ["#", "Image ID", "Target", "Rank", "Runtime", "Ins. AUC", "Del. AUC", "Max conf."]
    page_count = (len(rows) + rows_per_page - 1) // rows_per_page
    for page_index in range(page_count):
        start = page_index * rows_per_page
        page_rows = rows[start : start + rows_per_page]
        cells = []
        for offset, row in enumerate(page_rows, start=start + 1):
            target = str(row["target"]).replace("\n", " ")[:18]
            cells.append(
                [
                    str(offset),
                    row["image_id"],
                    target,
                    str(row["rank"]),
                    f"{row['runtime_seconds'] / 60:.2f}m",
                    f"{row['insertion_auc']:.4f}",
                    f"{row['deletion_auc']:.4f}",
                    f"{row['highest_confidence']:.4f}",
                ]
            )

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(
            f"Per-image results — page {page_index + 1}/{page_count}",
            fontsize=15,
            weight="bold",
            pad=15,
        )
        table = ax.table(
            cellText=cells,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            colWidths=[0.04, 0.14, 0.20, 0.06, 0.10, 0.10, 0.10, 0.11],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.28)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#17365D")
                cell.set_text_props(color="white", weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#EAF0F8")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--eval-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity", type=float, default=0.4)
    args = parser.parse_args()

    canonical_records = json.loads(args.eval_list.read_text(encoding="utf-8"))
    canonical_ids = [Path(record["image_path"]).stem for record in canonical_records]
    json_dir = args.results / "json"
    npy_dir = args.results / "npy"
    rows = []
    invalid = []

    for image_id in canonical_ids:
        json_path = json_dir / f"{image_id}.json"
        npy_path = npy_dir / f"{image_id}.npy"
        try:
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            regions = np.load(npy_path, mmap_mode="r")
            if regions.ndim != 4 or regions.shape[0] != saved["sub-region_number"]:
                raise ValueError("NPY shape does not match JSON region count")
            del regions
            scores = evaluate_record(saved, args.sensitivity)
            target = saved.get("words", "")
            if isinstance(target, list):
                target = " ".join(map(str, target))
            rows.append(
                {
                    "image_id": image_id,
                    "target": target,
                    "rank": int(saved["worker_rank"]),
                    "runtime_seconds": float(saved["runtime_seconds"]),
                    **scores,
                }
            )
        except Exception as error:
            invalid.append((image_id, str(error)))

    if invalid:
        details = "\n".join(f"{image_id}: {message}" for image_id, message in invalid[:10])
        raise RuntimeError(f"{len(invalid)} invalid result pairs:\n{details}")
    if len(rows) != len(canonical_ids):
        raise RuntimeError(f"expected {len(canonical_ids)} results, found {len(rows)}")

    rank_stats = defaultdict(list)
    for row in rows:
        rank_stats[row["rank"]].append(row["runtime_seconds"])
    runtimes = np.asarray([row["runtime_seconds"] for row in rows])
    critical_path = max(sum(values) for values in rank_stats.values())
    summary = {
        "count": len(rows),
        "canonical_count": len(canonical_ids),
        "invalid_count": len(invalid),
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
        metadata["Title"] = "EAGLE Qwen2.5-VL-7B COCO 250-image report"
        metadata["Author"] = "EAGLE evaluation pipeline"
        metadata["Subject"] = "Faithfulness metrics, runtimes, and per-image results"
        add_summary_page(pdf, summary)
        add_plot_pages(pdf, rows, rank_stats)
        add_table_pages(pdf, rows)

    print(json.dumps(summary, indent=2))
    print(f"PDF: {args.output}")


if __name__ == "__main__":
    main()
