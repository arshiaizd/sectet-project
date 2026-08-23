#!/usr/bin/env python3
"""Create a consolidated PDF for all completed 250-image explanation runs."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/mnt/vilab/scratch/arshia/projects/izadi")
OUTPUT = ROOT / "reports/all_methods_250_results.pdf"
SOURCES = {
    "EAGLE": ROOT / "eagle/results/eagle_250_seed_20260815/eagle_final_metrics.json",
    "TAM*": ROOT / "tam/results/qwen25vl7b_coco_object_250_seed_20260815/TAM_249_POSITIONAL/tam_250_mean_imputed_final_metrics.json",
    "LLaVA-CAM": ROOT / "llavacam/results/qwen25vl7b_coco_object_250_seed_20260815/LLaVACAM/llavacam_250_final_metrics.json",
    "Ours†": ROOT / "ours/results/encoder_250_seed_20260815/evaluation_patch8/ours_patch8_final_metrics.json",
}
COLORS = {
    "EAGLE": "#235789",
    "TAM*": "#F1A208",
    "LLaVA-CAM": "#D7263D",
    "Ours†": "#2E933C",
}


def load_results():
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}


def metric(result, key):
    return result["metrics"][key]


def mean_se(result, key):
    value = metric(result, key)
    return f"{value['mean']:.4f} ± {value['standard_error']:.4f}"


def style_table(table, header_color="#17365D", font_size=9):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C2CC")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF3F8")


def add_title_page(pdf, results):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.91, "COCO-250 Explanation Results", ha="center", fontsize=24, weight="bold")
    ax.text(
        0.5,
        0.865,
        "Qwen2.5-VL-7B-Instruct · canonical random subset (seed 20260815)",
        ha="center",
        fontsize=11,
        color="#3D4A57",
    )
    columns = ["Method", "Data", "Insertion AUC ↑", "Deletion AUC ↓", "Box PG ↑", "Mask PG ↑"]
    rows = []
    for name, result in results.items():
        data = "250"
        if name == "TAM*":
            data = "249 + 1 imp."
        rows.append(
            [
                name,
                data,
                mean_se(result, "insertion_auc"),
                mean_se(result, "deletion_auc"),
                mean_se(result, "pointing_game_box"),
                mean_se(result, "pointing_game_mask"),
            ]
        )
    rows.append(["IGOS++", "Not run", "—", "—", "—", "—"])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.49, 0.96, 0.29],
        colWidths=[0.16, 0.13, 0.20, 0.20, 0.16, 0.16],
    )
    style_table(table, font_size=8.5)
    ax.text(
        0.03,
        0.42,
        "Values are mean ± standard error. Higher is better for insertion and Pointing Game; "
        "lower is better for deletion.",
        fontsize=9.5,
        va="top",
        wrap=True,
    )
    ax.text(
        0.03,
        0.34,
        "* TAM: one failed sample was replaced by the mean of the other 249 samples, as requested. "
        "Its official positional runner regenerated captions, so canonical token alignment was not verified.",
        fontsize=9.5,
        va="top",
        wrap=True,
    )
    ax.text(
        0.03,
        0.25,
        "† Ours uses a yes/no object prompt and scores summed yes-token probability. EAGLE, TAM, "
        "and LLaVA-CAM score the canonical generated target token. The resulting AUC values are "
        "reported together for reference but are not a strictly controlled apples-to-apples comparison.",
        fontsize=9.5,
        va="top",
        wrap=True,
    )
    ax.text(
        0.03,
        0.12,
        "Canonical manifest SHA-256:\n8efa57d84af25aab013bc98a2deeed5ee6ffd094fec7bb401379f07ecb678804",
        fontsize=8.5,
        family="monospace",
        va="top",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_main_charts(pdf, results):
    methods = list(results)
    specs = [
        ("insertion_auc", "Insertion AUC ↑", (0, 1)),
        ("deletion_auc", "Deletion AUC ↓", (0, 0.6)),
        ("pointing_game_box", "Pointing Game — Box ↑", (0, 1)),
        ("pointing_game_mask", "Pointing Game — Mask ↑", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Primary Metrics", fontsize=20, weight="bold", y=0.98)
    for ax, (key, title, limits) in zip(axes.flat, specs):
        means = [metric(results[name], key)["mean"] for name in methods]
        errors = [metric(results[name], key)["standard_error"] for name in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=errors, capsize=4, color=[COLORS[name] for name in methods])
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x, methods, rotation=12)
        ax.set_ylim(*limits)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    fig.text(
        0.5,
        0.015,
        "Error bars show standard error across samples. See protocol notes before comparing Ours† directly with canonical-token methods.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_detailed_tables(pdf, results):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Detailed Metrics", fontsize=20, weight="bold", pad=18)
    columns = [
        "Method",
        "Highest conf. ↑",
        "Sensitive ins. ↑",
        "Sensitive del. ↓",
        "Sensitive high ↑",
        "Sensitive n",
        "Box hits",
        "Mask hits",
    ]
    rows = []
    for name, result in results.items():
        sensitive_n = metric(result, "sensitive_insertion_auc")["n"]
        box = metric(result, "pointing_game_box")
        mask = metric(result, "pointing_game_mask")
        box_hits = f"{box.get('hits', '—')}/{box['n']}"
        mask_hits = f"{mask.get('hits', '—')}/{mask['n']}"
        if name == "TAM*":
            box_hits = "128/249 obs."
            mask_hits = "93/249 obs."
        rows.append(
            [
                name,
                mean_se(result, "highest_confidence"),
                mean_se(result, "sensitive_insertion_auc"),
                mean_se(result, "sensitive_deletion_auc"),
                mean_se(result, "sensitive_highest_confidence"),
                str(sensitive_n),
                box_hits,
                mask_hits,
            ]
        )
    rows.append(["IGOS++", "—", "—", "—", "—", "—", "—", "—"])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.43, 0.98, 0.40],
        colWidths=[0.13, 0.16, 0.16, 0.16, 0.16, 0.09, 0.08, 0.08],
    )
    style_table(table, font_size=8)
    ax.text(
        0.02,
        0.34,
        "Sensitive metrics use threshold 0.4 and include only samples whose final insertion–deletion "
        "probability difference exceeds that threshold. Consequently, sensitive n varies by method.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    ax.text(
        0.02,
        0.23,
        "Pointing Game counts a hit when a maximum attribution value lies inside the target COCO box "
        "or segmentation mask; ties follow the official EAGLE evaluator.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_protocol_page(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title("Coverage, Protocols, and Limitations", fontsize=20, weight="bold", pad=20)
    columns = ["Method", "Observed", "Evaluation target", "Perturbation / status"]
    rows = [
        ["EAGLE", "250", "Canonical target token", "64-step official faithfulness; superpixel attribution"],
        ["TAM*", "249 + 1 imputed", "Canonical target token", "64-step official faithfulness; positional map caveat"],
        ["LLaVA-CAM", "250", "Canonical target token", "64-step official faithfulness; dense CAM"],
        ["Ours†", "250", "Summed yes-token mass", "28×28 patches; 8 inserted/deleted per regular step"],
        ["IGOS++", "0", "—", "Configured but attribution/evaluation not run"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        bbox=[0.01, 0.56, 0.98, 0.28],
        colWidths=[0.15, 0.17, 0.25, 0.43],
    )
    style_table(table, font_size=8.5)
    notes = [
        "All completed methods use the same canonical 250 COCO validation images.",
        "All means and standard errors were recomputed from per-image outputs using the same aggregation code.",
        "TAM's synthetic 250th row equals the corresponding observed-249 mean for every metric; it is not a generated saliency map.",
        "Our method's final partial step changes the remaining 1–8 patches whenever the patch count is not divisible by eight.",
        "IGOS++ must be run before a numerical row can be added. No value was inferred or copied from the paper.",
    ]
    y = 0.48
    for note in notes:
        ax.text(0.04, y, "• " + note, fontsize=10, va="top", wrap=True)
        y -= 0.075
    ax.text(
        0.02,
        0.06,
        "Generated " + datetime.now().astimezone().isoformat(timespec="seconds"),
        fontsize=8.5,
        color="#59636E",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load_results()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "COCO-250 explanation method comparison"
        metadata["Subject"] = "EAGLE, TAM, LLaVA-CAM, IGOS++ status, and our method"
        add_title_page(pdf, results)
        add_main_charts(pdf, results)
        add_detailed_tables(pdf, results)
        add_protocol_page(pdf)
    print(OUTPUT)


if __name__ == "__main__":
    main()
