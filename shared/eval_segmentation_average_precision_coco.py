#!/usr/bin/env python3
"""Per-image segmentation AUPRC for COCO attribution maps.

This adapts the repository-root ``segmentation_average_precision.py`` prototype
to the benchmark's actual data formats. The metric is unchanged in spirit:
resize each attribution map with bicubic interpolation, min-max normalize it,
and compute foreground (aupr1) and background (aupr0) precision-recall AUC
against a binary segmentation mask. COCO polygon/RLE masks come directly from
the shared manifest, so separate PNG mask files are not needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve

from eval_point_game_coco import add_value, segmentation_to_mask


EPS = 1e-8


def image_name(value: Any) -> str:
    return os.path.basename(str(value).strip())


def resize_bicubic(score_map: np.ndarray, height: int, width: int) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float32)
    if score.ndim != 2:
        score = score.squeeze()
    if score.ndim != 2:
        raise ValueError(f"score map must be two-dimensional after squeeze, got {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("score map contains NaN or infinity")
    tensor = torch.from_numpy(score)[None, None]
    if score.shape != (height, width):
        tensor = F.interpolate(
            tensor,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
    return tensor[0, 0].numpy()


def normalize_unit_interval(score_map: np.ndarray) -> np.ndarray:
    minimum = float(np.min(score_map))
    maximum = float(np.max(score_map))
    return (score_map - minimum) / (maximum - minimum + EPS)


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    if mask.shape != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="nearest")
    return (tensor[0, 0].numpy() > 0).astype(np.uint8)


def binary_auprc(scores: np.ndarray, target: np.ndarray) -> float:
    y_score = np.asarray(scores, dtype=np.float32).reshape(-1)
    y_true = np.asarray(target, dtype=np.uint8).reshape(-1)
    if y_score.shape != y_true.shape:
        raise ValueError(f"score/target shape mismatch: {y_score.shape} vs {y_true.shape}")
    if not np.any(y_true):
        raise ValueError("AUPRC target has no positive pixels")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def load_patch_csvs(paths: list[str], score_column: str) -> tuple[pd.DataFrame, str]:
    if not paths:
        raise ValueError("no patch CSV files were supplied")
    required = {"image_path", "center_patch_index", "source_grid_h", "source_grid_w"}
    headers = [set(pd.read_csv(path, nrows=0).columns) for path in paths]
    common_columns = set.intersection(*headers)
    missing = required.difference(common_columns)
    if missing:
        raise KeyError(f"patch CSV is missing columns: {sorted(missing)}")
    if not score_column:
        score_column = (
            "attribution_score"
            if "attribution_score" in common_columns
            else "patched_yes_probability"
        )
    if score_column not in common_columns:
        raise KeyError(
            f"score column {score_column!r} is absent from at least one CSV; "
            f"common columns: {sorted(common_columns)}"
        )
    use_columns = sorted(required | {score_column})
    frames = [pd.read_csv(path, usecols=use_columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.copy()
    frame["_image_name"] = frame["image_path"].map(image_name)
    return frame, score_column


def patch_map(rows: pd.DataFrame, score_column: str) -> np.ndarray:
    grid_h_values = rows["source_grid_h"].dropna().astype(int).unique()
    grid_w_values = rows["source_grid_w"].dropna().astype(int).unique()
    if len(grid_h_values) != 1 or len(grid_w_values) != 1:
        raise ValueError("each image must have exactly one source grid shape")
    grid_h, grid_w = int(grid_h_values[0]), int(grid_w_values[0])
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    grouped = rows.groupby("center_patch_index", sort=False)[score_column].mean()
    for patch_index, score in grouped.items():
        patch_index = int(patch_index)
        row, column = divmod(patch_index, grid_w)
        if not (0 <= row < grid_h and 0 <= column < grid_w):
            raise ValueError(f"patch index {patch_index} is outside {grid_h}x{grid_w}")
        grid[row, column] = float(score)
    return grid


def dense_map(explanation_dir: Path, name: str) -> np.ndarray:
    npy_root = explanation_dir / "npy" if (explanation_dir / "npy").is_dir() else explanation_dir
    path = npy_root / f"{Path(name).stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing dense map: {path}")
    return np.load(path)


def eagle_map(explanation_dir: Path, name: str) -> np.ndarray:
    stem = Path(name).stem
    json_path = explanation_dir / "json" / f"{stem}.json"
    npy_path = explanation_dir / "npy" / f"{stem}.npy"
    if not json_path.is_file():
        raise FileNotFoundError(f"missing EAGLE JSON: {json_path}")
    if not npy_path.is_file():
        raise FileNotFoundError(f"missing EAGLE superpixel masks: {npy_path}")
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    superpixel_masks = np.load(npy_path)
    attribution_map, _ = add_value(superpixel_masks, saved)
    return attribution_map


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "standard_error": standard_error(values),
        "n": len(values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.eval_list).resolve()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("--eval-list must contain a JSON list")
    selected = records[args.begin : args.end if args.end >= 0 else None]
    if not selected:
        raise ValueError("selected manifest range is empty")

    patch_frame: pd.DataFrame | None = None
    score_column: str | None = None
    explanation_dir: Path | None = None
    if args.map_source == "patch":
        patch_frame, score_column = load_patch_csvs(args.csv, args.score_column)
    else:
        explanation_dir = Path(args.explanation_dir).resolve()
        if not explanation_dir.is_dir():
            raise FileNotFoundError(f"explanation directory does not exist: {explanation_dir}")

    output_rows: list[dict[str, Any]] = []
    foreground_values: list[float] = []
    background_values: list[float] = []
    failures: list[str] = []

    for relative_index, record in enumerate(selected):
        dataset_index = args.begin + relative_index
        name = image_name(record["image_path"])
        try:
            native_height = int(record["height"])
            native_width = int(record["width"])
            height = args.resolution if args.resolution > 0 else native_height
            width = args.resolution if args.resolution > 0 else native_width
            ground_truth_native = segmentation_to_mask(
                record.get("segmentation"), native_width, native_height
            )
            ground_truth = resize_binary_mask(ground_truth_native, height, width)
            if not np.any(ground_truth):
                raise ValueError("ground-truth segmentation mask is empty")

            if args.map_source == "patch":
                assert patch_frame is not None and score_column is not None
                rows = patch_frame[patch_frame["_image_name"] == name]
                if rows.empty:
                    raise FileNotFoundError(f"no patch rows for {name}")
                raw_map = patch_map(rows, score_column)
            elif args.map_source == "dense":
                assert explanation_dir is not None
                raw_map = dense_map(explanation_dir, name)
            else:
                assert explanation_dir is not None
                raw_map = eagle_map(explanation_dir, name)

            score_map = normalize_unit_interval(resize_bicubic(raw_map, height, width))
            foreground_ap = binary_auprc(score_map, ground_truth)
            background_ap = binary_auprc(1.0 - score_map, ground_truth == 0)
            foreground_values.append(foreground_ap)
            background_values.append(background_ap)
            output_rows.append(
                {
                    "dataset_sample_index": dataset_index,
                    "image_path": name,
                    "select_category": record.get("select_category", ""),
                    "mask_fraction": float(np.mean(ground_truth)),
                    "segmentation_auprc_foreground": foreground_ap,
                    "segmentation_auprc_background": background_ap,
                    "note": "",
                }
            )
        except Exception as error:
            message = f"{name}: {type(error).__name__}: {error}"
            failures.append(message)
            output_rows.append(
                {
                    "dataset_sample_index": dataset_index,
                    "image_path": name,
                    "select_category": record.get("select_category", ""),
                    "mask_fraction": np.nan,
                    "segmentation_auprc_foreground": np.nan,
                    "segmentation_auprc_background": np.nan,
                    "note": message,
                }
            )
            print(f"[error] {message}")

    summary = {
        "metric": "per-image segmentation precision-recall AUC",
        "map_source": args.map_source,
        "score_column": score_column,
        "interpolation": "torch bicubic, align_corners=False",
        "normalization": "per-image min-max to [0,1]",
        "resolution": (
            [args.resolution, args.resolution]
            if args.resolution > 0
            else "original per-image resolution"
        ),
        "manifest": str(manifest_path),
        "requested": len(selected),
        "scored": len(foreground_values),
        "failed": len(failures),
        "segmentation_auprc_foreground": summarize(foreground_values) if foreground_values else None,
        "segmentation_auprc_background": summarize(background_values) if background_values else None,
        "failures": failures,
    }

    out_path = Path(args.out).resolve()
    summary_path = Path(args.out_summary).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(out_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Scored: {summary['scored']}/{summary['requested']}")
    if foreground_values:
        print(f"Segmentation AUPRC foreground: {summary['segmentation_auprc_foreground']['mean']:.6f}")
        print(f"Segmentation AUPRC background: {summary['segmentation_auprc_background']['mean']:.6f}")
    print(f"Per-image CSV: {out_path}")
    print(f"Summary JSON: {summary_path}")
    if failures and not args.allow_missing:
        raise RuntimeError(
            f"segmentation AUPRC failed for {len(failures)} of {len(selected)} images; "
            "inspect the output CSV or rerun with --allow-missing"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-source", choices=("patch", "dense", "eagle"), required=True)
    parser.add_argument("--eval-list", required=True, help="COCO benchmark manifest with segmentation")
    parser.add_argument("--csv", nargs="+", help="patch attribution CSV files")
    parser.add_argument("--score-column", default="", help="patch score column; auto-detected by default")
    parser.add_argument(
        "--explanation-dir",
        help="dense directory containing npy/, or EAGLE slico directory containing json/ and npy/",
    )
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="fixed square evaluation resolution (default: 224); use 0 for original resolution",
    )
    parser.add_argument("--out", default="./segmentation_auprc_per_image.csv")
    parser.add_argument("--out-summary", default="./segmentation_auprc_summary.json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.resolution < 0:
        parser.error("--resolution must be 0 or a positive integer")
    if args.map_source == "patch" and not args.csv:
        parser.error("--map-source patch requires --csv")
    if args.map_source in ("dense", "eagle") and not args.explanation_dir:
        parser.error(f"--map-source {args.map_source} requires --explanation-dir")
    return args


if __name__ == "__main__":
    run(parse_args())
