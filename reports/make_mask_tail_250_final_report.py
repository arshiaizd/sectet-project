#!/usr/bin/env python3
"""Build the audited COCO mask-tail-250 benchmark PDF from per-image outputs."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "mask_tail_250_final_report.pdf"
SUMMARY_CSV = ROOT / "reports" / "previews" / "mask_tail_250_metrics_summary.csv"
MANIFEST = ROOT / "shared" / "coco_mask_tail_250_benchmark.json"
GPU_COUNT = 4
SENSITIVITY_THRESHOLD = 0.4

# These anchors were captured from the completed jobs before model loading.  The
# end is recovered from the newest per-image JSON, so CPU-only aggregation is
# excluded.  See the runtime page in the report for the exact definition.
METHODS = {
    "Ours": {
        "directory": ROOT / "ours/results/ours_mask_tail_250_v2/evaluation_patch8",
        "start_epoch": 1787603853.383058288,
        "protocol": "patch / white baseline / 8 patches per step",
        "color": "#2E8B57",
    },
    "Input-level": {
        "directory": ROOT / "input_deletion/results/input_deletion_mask_tail_250/evaluation_patch8",
        "start_epoch": 1787573740.329747605,
        "protocol": "patch / white baseline / 8 patches per step",
        "color": "#8E5EA2",
    },
    "TAM": {
        "directory": ROOT / "tam/results/mask_tail_250_portable/TAM",
        "start_epoch": 1787577683.224691035,
        "protocol": "dense map / black baseline / 64 pixel-fraction steps",
        "color": "#F39C12",
    },
    "LLaVA-CAM": {
        "directory": ROOT / "llavacam/results/mask_tail_250_portable/LLaVACAM",
        "start_epoch": 1787580699.003374532,
        "protocol": "dense map / black baseline / 64 pixel-fraction steps",
        "color": "#C0392B",
    },
}


def mean_se(values: list[float]) -> dict[str, float | int]:
    values_array = np.asarray(values, dtype=float)
    return {
        "mean": float(values_array.mean()),
        "standard_error": float(values_array.std(ddof=1) / math.sqrt(values_array.size)),
        "n": int(values_array.size),
    }


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Equivalent to sklearn.metrics.auc for monotonic ascending or descending x."""
    return float(abs(np.trapz(y, x)))


def load_method(name: str, spec: dict) -> dict:
    json_files = sorted((spec["directory"] / "json").glob("*.json"))
    if len(json_files) != 250:
        raise RuntimeError(f"{name}: expected 250 JSON files, found {len(json_files)}")

    raw_curves = []
    values = {
        "insertion_auc": [],
        "deletion_auc": [],
        "highest_confidence": [],
        "sensitive_insertion_auc": [],
        "sensitive_deletion_auc": [],
        "sensitive_highest_confidence": [],
    }
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        x = np.asarray([0.0] + data["region_area"], dtype=float)
        insertion = np.asarray([data["deletion_score"][-1]] + data["insertion_score"], dtype=float)
        deletion = np.asarray([data["insertion_score"][-1]] + data["deletion_score"], dtype=float)
        if not (len(x) == len(insertion) == len(deletion)):
            raise RuntimeError(f"{name}: inconsistent curve lengths in {path}")
        values["insertion_auc"].append(auc(x, insertion))
        values["deletion_auc"].append(auc(1.0 - x, deletion))
        values["highest_confidence"].append(float(insertion.max()))
        raw_curves.append((x, insertion, deletion))

        insertion_words = np.asarray(data["insertion_word_score"], dtype=float)
        deletion_words = np.asarray(data["deletion_word_score"], dtype=float)
        sensitive = (insertion_words[-1] - deletion_words[-1]) > SENSITIVITY_THRESHOLD
        if sensitive.any():
            ins_sensitive = np.concatenate(
                ([deletion_words[-1, sensitive].mean()], insertion_words[:, sensitive].mean(axis=1))
            )
            del_sensitive = np.concatenate(
                ([insertion_words[-1, sensitive].mean()], deletion_words[:, sensitive].mean(axis=1))
            )
            values["sensitive_insertion_auc"].append(auc(x, ins_sensitive))
            values["sensitive_deletion_auc"].append(auc(1.0 - x, del_sensitive))
            values["sensitive_highest_confidence"].append(float(ins_sensitive.max()))

    pg_path = spec["directory"] / "point_game_per_image.csv"
    with pg_path.open(newline="", encoding="utf-8") as handle:
        pg_rows = list(csv.DictReader(handle))
    if len(pg_rows) != 250:
        raise RuntimeError(f"{name}: expected 250 Pointing Game rows, found {len(pg_rows)}")
    values["pointing_game_box"] = [float(row["pg_box"]) for row in pg_rows]
    values["pointing_game_mask"] = [float(row["pg_mask"]) for row in pg_rows]

    metrics = {key: mean_se(metric_values) for key, metric_values in values.items()}
    end_epoch = max(path.stat().st_mtime for path in json_files)
    wall_seconds = end_epoch - spec["start_epoch"]
    if wall_seconds <= 0:
        raise RuntimeError(f"{name}: invalid timing interval")
    return {
        "metrics": metrics,
        "curves": raw_curves,
        "wall_seconds": wall_seconds,
        "gpu_seconds_per_sample": GPU_COUNT * wall_seconds / len(json_files),
        "end_epoch": end_epoch,
    }


def format_metric(item: dict) -> str:
    return f"{item['mean']:.4f} ± {item['standard_error']:.4f}"


def style_table(table, font_size=8.2):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#AEB8C2")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EDF3F8")


def add_overview(pdf: PdfPages, results: dict):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.5, 0.94, "COCO Mask-Tail-250 Benchmark", ha="center", fontsize=24, weight="bold")
    ax.text(
        0.5,
        0.895,
        "Qwen2.5-VL-7B-Instruct · same audited 250-image subset",
        ha="center",
        fontsize=11,
        color="#44515C",
    )
    columns = [
        "Method",
        "Ins. AUC ↑",
        "Del. AUC ↓",
        "Highest conf. ↑",
        "PG box ↑",
        "PG mask ↑",
        "Eval GPU-s/sample ↓",
    ]
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append(
            [
                name,
                format_metric(m["insertion_auc"]),
                format_metric(m["deletion_auc"]),
                format_metric(m["highest_confidence"]),
                format_metric(m["pointing_game_box"]),
                format_metric(m["pointing_game_mask"]),
                f"{result['gpu_seconds_per_sample']:.2f}",
            ]
        )
    rows.append(["EAGLE", "—", "—", "—", "—", "—", "—"])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.49, 0.98, 0.31],
        colWidths=[0.13, 0.155, 0.155, 0.155, 0.135, 0.135, 0.135],
    )
    style_table(table, font_size=7.6)
    ax.text(
        0.025,
        0.42,
        "Values are mean ± sample standard error. N=250 for every primary metric. "
        "Higher is better except deletion AUC and runtime, where lower is better.",
        fontsize=10,
        va="top",
    )
    ax.text(
        0.025,
        0.33,
        "Evaluation GPU-s/sample = 4 GPUs × evaluation wall time / 250. It includes model loading "
        "and perturbation scoring, excludes later CPU aggregation, and is not attribution-generation time.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    ax.text(
        0.025,
        0.235,
        "EAGLE is intentionally blank because it has not been run on this new benchmark. No paper value "
        "or result from an older subset was substituted.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    ax.text(
        0.025,
        0.12,
        "Coverage audit: 250/250 per-image evaluation JSON files and 250/250 Pointing Game rows "
        "for every reported method.",
        fontsize=9.5,
        weight="bold",
        color="#17365D",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_all_metrics(pdf: PdfPages, results: dict):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("All Aggregate Metrics", fontsize=20, weight="bold", pad=18)
    columns = [
        "Method",
        "Sensitive ins. ↑",
        "Sensitive del. ↓",
        "Sensitive high ↑",
        "Sensitive n",
        "Wall min",
        "Eval GPU-s/sample ↓",
    ]
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        rows.append(
            [
                name,
                format_metric(m["sensitive_insertion_auc"]),
                format_metric(m["sensitive_deletion_auc"]),
                format_metric(m["sensitive_highest_confidence"]),
                str(m["sensitive_insertion_auc"]["n"]),
                f"{result['wall_seconds'] / 60:.2f}",
                f"{result['gpu_seconds_per_sample']:.2f}",
            ]
        )
    rows.append(["EAGLE", "—", "—", "—", "—", "—", "—"])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.025, 0.50, 0.95, 0.31],
        colWidths=[0.14, 0.18, 0.18, 0.18, 0.10, 0.10, 0.14],
    )
    style_table(table, font_size=8.1)
    ax.text(
        0.04,
        0.40,
        "Sensitive metrics retain a sample only when its final target-score difference is greater than "
        f"{SENSITIVITY_THRESHOLD:.1f}. The subset is method-dependent, hence n varies.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    ax.text(
        0.04,
        0.29,
        "Pointing Game is reported against both the COCO bounding box and segmentation mask. Patch "
        "methods use their patch representative point; dense-map methods use the dense maximum point.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    ax.text(
        0.04,
        0.17,
        "The complete primary metrics (insertion, deletion, confidence, and both Pointing Game variants) "
        "are on page 1; this page supplies the sensitivity-conditioned metrics and exact runtime summary.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def curve_summary(curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    grid = np.linspace(0.0, 1.0, 101)
    insertion = np.asarray([np.interp(grid, x, ins) for x, ins, _ in curves])
    deletion = np.asarray([np.interp(grid, x, dele) for x, _, dele in curves])
    return grid, insertion.mean(0), insertion.std(0, ddof=1) / math.sqrt(len(curves)), deletion.mean(0), deletion.std(0, ddof=1) / math.sqrt(len(curves))


def add_curves(pdf: PdfPages, results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True, sharey=True)
    fig.suptitle("Mean Faithfulness Curves", fontsize=20, weight="bold", y=0.98)
    for ax, (name, result) in zip(axes.flat, results.items()):
        grid, ins, ins_se, dele, dele_se = curve_summary(result["curves"])
        ax.plot(grid, ins, color="#2E8B57", linewidth=2, label="Insertion")
        ax.fill_between(grid, ins - ins_se, ins + ins_se, color="#2E8B57", alpha=0.16)
        ax.plot(grid, dele, color="#D6273B", linewidth=2, label="Deletion")
        ax.fill_between(grid, dele - dele_se, dele + dele_se, color="#D6273B", alpha=0.16)
        ax.set_title(name, fontsize=13, weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Fraction changed")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target score")
    fig.text(
        0.5,
        0.015,
        "Lines are the mean over 250 samples; shaded bands show ±1 standard error. Curves are linearly interpolated onto 101 common fractions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_protocol(pdf: PdfPages, results: dict):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title("Protocol and Runtime Provenance", fontsize=20, weight="bold", pad=20)
    columns = ["Method", "Attribution / perturbation protocol", "Status"]
    rows = [[name, METHODS[name]["protocol"], "250 complete"] for name in results]
    rows.append(["EAGLE", "—", "Not run"])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        bbox=[0.02, 0.61, 0.96, 0.24],
        colWidths=[0.19, 0.61, 0.20],
    )
    style_table(table, font_size=8.1)
    notes = [
        "All methods use the same audited COCO val2017 mask-tail-250 manifest and Qwen2.5-VL-7B-Instruct model.",
        "Ours and Input-level score summed yes-token probability. TAM and LLaVA-CAM use the audited COCO-caption target token. These target definitions should be considered when comparing absolute AUCs.",
        "Ours and Input-level evaluate merged vision patches and change 8 patches per regular step. TAM and LLaVA-CAM evaluate dense heatmaps with 64 equal pixel-fraction steps.",
        "Runtime begins at the recorded pre-load job anchor and ends at the final per-image evaluation JSON timestamp. All four completed evaluation jobs used four GPUs.",
        "Attribution generation time is not included in the runtime column because an exact uniform start anchor is unavailable for every method.",
        "Every standard error uses the sample standard deviation (ddof=1) divided by sqrt(n).",
    ]
    y = 0.53
    for note in notes:
        ax.text(0.045, y, "• " + note, fontsize=9.5, va="top", wrap=True)
        y -= 0.078
    ax.text(0.025, 0.05, f"Manifest: {MANIFEST.relative_to(ROOT)}", fontsize=8.5, family="monospace")
    ax.text(
        0.025,
        0.025,
        "Generated " + datetime.now().astimezone().isoformat(timespec="seconds"),
        fontsize=8,
        color="#59636E",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(results: dict):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    metric_order = [
        "insertion_auc",
        "deletion_auc",
        "highest_confidence",
        "sensitive_insertion_auc",
        "sensitive_deletion_auc",
        "sensitive_highest_confidence",
        "pointing_game_box",
        "pointing_game_mask",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "mean", "standard_error", "n", "evaluation_gpu_seconds_per_sample"])
        for name, result in results.items():
            for key in metric_order:
                item = result["metrics"][key]
                writer.writerow([name, key, item["mean"], item["standard_error"], item["n"], result["gpu_seconds_per_sample"]])
        for key in metric_order:
            writer.writerow(["EAGLE", key, "", "", "", ""])


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_size = len(manifest) if isinstance(manifest, list) else len(manifest.get("samples", []))
    if manifest_size != 250:
        raise RuntimeError(f"Expected a 250-sample manifest, found {manifest_size}")
    results = {name: load_method(name, spec) for name, spec in METHODS.items()}
    write_summary_csv(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "COCO mask-tail-250 benchmark report"
        metadata["Subject"] = "Ours, input-level, TAM, LLaVA-CAM; EAGLE pending"
        add_overview(pdf, results)
        add_all_metrics(pdf, results)
        add_curves(pdf, results)
        add_protocol(pdf, results)
    print(OUTPUT)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
