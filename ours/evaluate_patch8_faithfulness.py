#!/usr/bin/env python3
"""Evaluate patch-ranked attribution by insertion/deletion in groups of 8 patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from our_method_insertion_online_encoder import (
    MAX_PIXELS,
    MIN_PIXELS,
    SYSTEM_PROMPT,
    prepare_yes_no_prompt_inputs,
    yes_token_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch-grid insertion/deletion evaluation in fixed groups of patches."
    )
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patches-per-step", type=int, default=8)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def output_is_complete(path: Path, expected_steps: int, patches_per_step: int) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        lengths = [
            len(saved["insertion_score"]),
            len(saved["deletion_score"]),
            len(saved["insertion_word_score"]),
            len(saved["deletion_word_score"]),
            len(saved["region_area"]),
        ]
        return (
            lengths == [expected_steps] * 5
            and int(saved["patches_per_step"]) == patches_per_step
            and abs(float(saved["region_area"][-1]) - 1.0) < 1e-12
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return False


def load_groups(path: Path) -> List[List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[int(row["dataset_sample_index"])].append(row)
    return [grouped[index] for index in sorted(grouped)]


def validate_group(rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("empty attribution group")
    image_names = {row["image_path"] for row in rows}
    if len(image_names) != 1:
        raise ValueError(f"group contains multiple images: {sorted(image_names)}")
    grid_h = int(rows[0]["source_grid_h"])
    grid_w = int(rows[0]["source_grid_w"])
    expected = grid_h * grid_w
    indices = [int(row["center_patch_index"]) for row in rows]
    ranks = [int(row["rank_by_yes_probability"]) for row in rows]
    if len(rows) != expected or set(indices) != set(range(expected)):
        raise ValueError(
            f"{rows[0]['image_path']}: patch rows do not cover grid {grid_h}x{grid_w}"
        )
    if set(ranks) != set(range(1, expected + 1)):
        raise ValueError(f"{rows[0]['image_path']}: attribution ranks are incomplete")
    fixed_fields = (
        "prompt",
        "yes_token_ids",
        "baseline",
        "resized_height",
        "resized_width",
        "source_grid_h",
        "source_grid_w",
    )
    for field in fixed_fields:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"{rows[0]['image_path']}: inconsistent {field}")


def make_rank_map(rows: List[Dict[str, str]]) -> np.ndarray:
    grid_h = int(rows[0]["source_grid_h"])
    grid_w = int(rows[0]["source_grid_w"])
    result = np.empty((grid_h, grid_w), dtype=np.float32)
    for row in rows:
        result[int(row["center_patch_row"]), int(row["center_patch_col"])] = float(
            row["patched_yes_probability"]
        )
    return result


def apply_patch_indices(
    insertion: np.ndarray,
    deletion: np.ndarray,
    source: np.ndarray,
    baseline: np.ndarray,
    indices: List[int],
    grid_w: int,
    patch_height: int,
    patch_width: int,
) -> None:
    for patch_index in indices:
        row, col = divmod(int(patch_index), grid_w)
        y0, y1 = row * patch_height, (row + 1) * patch_height
        x0, x1 = col * patch_width, (col + 1) * patch_width
        insertion[y0:y1, x0:x1] = source[y0:y1, x0:x1]
        deletion[y0:y1, x0:x1] = baseline[y0:y1, x0:x1]


@torch.inference_mode()
def evaluate_group(
    rows: List[Dict[str, str]],
    processor,
    model,
    coco_root: Path,
    patches_per_step: int,
) -> Dict[str, Any]:
    validate_group(rows)
    first = rows[0]
    image_path = coco_root / first["image_path"]
    resized_h = int(first["resized_height"])
    resized_w = int(first["resized_width"])
    grid_h = int(first["source_grid_h"])
    grid_w = int(first["source_grid_w"])
    if resized_h % grid_h or resized_w % grid_w:
        raise ValueError(
            f"{first['image_path']}: resized dimensions do not divide patch grid"
        )
    patch_height = resized_h // grid_h
    patch_width = resized_w // grid_w
    if (patch_height, patch_width) != (28, 28):
        raise ValueError(
            f"{first['image_path']}: expected 28x28 merged patches, got "
            f"{patch_height}x{patch_width}"
        )

    source_image = Image.open(image_path).convert("RGB").resize((resized_w, resized_h))
    source = np.asarray(source_image, dtype=np.uint8).copy()
    baseline_value = 255 if first["baseline"] == "white" else 0
    baseline = np.full_like(source, baseline_value)
    insertion = baseline.copy()
    deletion = source.copy()

    ranked = sorted(rows, key=lambda row: int(row["rank_by_yes_probability"]))
    ranked_indices = [int(row["center_patch_index"]) for row in ranked]
    total_patches = len(ranked_indices)
    yes_token_ids = [int(value) for value in json.loads(first["yes_token_ids"])]
    prompt = first["prompt"]
    device = model.device

    result: Dict[str, Any] = {
        "image_path": first["image_path"],
        "dataset_sample_index": int(first["dataset_sample_index"]),
        "metric_target": "summed_yes_token_probability",
        "ranking_field": "rank_by_yes_probability",
        "perturbation_unit": "one_merged_vision_patch",
        "patch_height": patch_height,
        "patch_width": patch_width,
        "grid_height": grid_h,
        "grid_width": grid_w,
        "total_patches": total_patches,
        "patches_per_step": patches_per_step,
        "baseline": first["baseline"],
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "yes_token_ids": yes_token_ids,
        "insertion_score": [],
        "deletion_score": [],
        "insertion_word_score": [],
        "deletion_word_score": [],
        "region_area": [],
        "patches_changed_per_step": [],
    }

    for begin in tqdm(
        range(0, total_patches, patches_per_step),
        desc=f"patch8 {first['image_path']}",
        leave=False,
    ):
        batch = ranked_indices[begin : begin + patches_per_step]
        apply_patch_indices(
            insertion,
            deletion,
            source,
            baseline,
            batch,
            grid_w,
            patch_height,
            patch_width,
        )
        insertion_batch = prepare_yes_no_prompt_inputs(
            processor, Image.fromarray(insertion), prompt, SYSTEM_PROMPT, device
        )
        deletion_batch = prepare_yes_no_prompt_inputs(
            processor, Image.fromarray(deletion), prompt, SYSTEM_PROMPT, device
        )
        insertion_score = yes_token_probability(model, insertion_batch, yes_token_ids)
        deletion_score = yes_token_probability(model, deletion_batch, yes_token_ids)
        changed = min(begin + patches_per_step, total_patches)
        result["insertion_score"].append(insertion_score)
        result["deletion_score"].append(deletion_score)
        result["insertion_word_score"].append([insertion_score])
        result["deletion_word_score"].append([deletion_score])
        result["region_area"].append(changed / total_patches)
        result["patches_changed_per_step"].append(len(batch))

    return result


def main() -> None:
    args = parse_args()
    if args.patches_per_step <= 0:
        raise ValueError("--patches-per-step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "json"
    npy_dir = args.output_dir / "npy"
    json_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for input_csv in args.input_csv:
        groups.extend(load_groups(input_csv))
    sample_indices = [int(rows[0]["dataset_sample_index"]) for rows in groups]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("input CSV files contain overlapping dataset samples")
    groups.sort(key=lambda rows: int(rows[0]["dataset_sample_index"]))
    for rows in groups:
        validate_group(rows)
        stem = Path(rows[0]["image_path"]).stem
        npy_path = npy_dir / f"{stem}.npy"
        try:
            existing_map = np.load(npy_path, mmap_mode="r")
            map_complete = existing_map.ndim == 2 and existing_map.size > 0
            del existing_map
        except (OSError, EOFError, ValueError):
            map_complete = False
        if not map_complete:
            atomic_write_npy(npy_path, make_rank_map(rows))

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    input_names = "+".join(path.name for path in args.input_csv)
    for rows in tqdm(groups, desc=f"evaluation {input_names}"):
        expected_steps = math.ceil(len(rows) / args.patches_per_step)
        output_path = args.output_dir / "json" / Path(rows[0]["image_path"]).with_suffix(
            ".json"
        ).name
        if output_is_complete(output_path, expected_steps, args.patches_per_step):
            continue
        result = evaluate_group(
            rows, processor, model, args.coco_root, args.patches_per_step
        )
        atomic_write_json(output_path, result)
        print(
            f"[done] {result['image_path']} patches={result['total_patches']} "
            f"steps={len(result['region_area'])}",
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
